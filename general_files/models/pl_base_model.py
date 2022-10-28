import math
from typing import Any, List
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import functional as F
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoModelWithLMHead,
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from general_files.utils.common_util import Result


# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup,  # not supported for now
}


class BasePLModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer
        self.stage = 'train'
        if "dropout" not in config:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(config.dropout)
        self.model_mode = {
            "base": AutoModel,
            "sequence-classification": AutoModelForSequenceClassification,
            "question-answering": AutoModelForQuestionAnswering,
            "pretraining": AutoModelForPreTraining,
            "token-classification": AutoModelForTokenClassification,
            "language-modeling": AutoModelForCausalLM,
            "seq2seq": AutoModelForSeq2SeqLM,
            "base-lm_head": AutoModelWithLMHead,
        }

    def training_step(self, batch: Any, batch_idx: int):
        outputs = self(**batch)
        if outputs['loss'] != outputs['loss']:
            raise Exception("Loss为Nan，请先检查数据正确性！")
        self.log("loss", outputs['loss'], prog_bar=False, logger=True, sync_dist=True, on_step=True,
                         on_epoch=True, rank_zero_only=True)
        for key in outputs.keys():
            if '_loss' in key:
                key_loss = round(float(torch.clamp(outputs[key].cpu(), max=99, min=0)), 3)
                self.log("train/" + key, key_loss, prog_bar=False, logger=True, sync_dist=True, on_step=True,
                         on_epoch=True, rank_zero_only=True)
        if 'lm_loss' in outputs:
            ppl = round(float(torch.clamp(torch.exp(outputs['lm_loss']).cpu(), max=99, min=0)), 3)
            self.log("ppl", ppl, prog_bar=False, logger=True, sync_dist=True, rank_zero_only=True)
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        
        self.log("lr", round(lr_scheduler.get_last_lr()[-1], 6), prog_bar=True, logger=True, on_step=True,
                 rank_zero_only=True, sync_dist=True)
        self.log("current_epoch", self.current_epoch)
        self.log("progress", self.global_step / self.total_steps())
        return {
            'loss': outputs['loss'],
        }

    def validation_step(self, batch: Any, batch_idx: int):
        outputs = self(**batch)
        self.log("val/step_loss", outputs['loss'], prog_bar=False)
        return {"val_step_loss": outputs['loss']}

    def validation_epoch_end(self, outputs: List[Any]):
        mean_loss = torch.stack([x['val_step_loss'] for x in outputs]).mean().item()
        log = {
            'val_loss': mean_loss,
        }
        self.log('val_loss', mean_loss)
        return log

    def prepare_other_features_for_generation(self, batch):
        ignore_keys = ['input_ids', 'labels', 'decoder_input_ids']
        other_features = dict()
        for key in batch.keys():
            if key not in ignore_keys:
                try:
                    other_features[key] = torch.LongTensor(batch[key]).to(self.device)
                except ValueError as e:
                    other_features[key] = batch[key]
        return other_features

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.config.scheduler]
        total_steps = self.total_steps()
        if self.config.warmup_ratio > 0:
            warmup_steps = self.config.warmup_ratio * total_steps
        else:
            warmup_steps = self.config.warmup_steps

        if self.config.scheduler != "constant":
            scheduler = get_schedule_func(self.opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        else:
            scheduler = get_schedule_func(self.opt, num_warmup_steps=warmup_steps)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.config.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.config.lr,
                scale_parameter=False,
                relative_step=False,
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.lr,
                eps=self.config.adam_epsilon,
            )
        # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.lr)
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()
        return [optimizer], [scheduler]

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.config.want_gpu_num)
        effective_batch_size = self.config.train_batch_size * self.config.accumulate_grad_batches * num_devices
        return (self.config.dataset_size / effective_batch_size) * self.config.max_epochs

    def NLLLoss(self, logits, labels):
        loss_fct = torch.nn.NLLLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return loss

    def CrossEntropyLoss(self, logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return loss

    @staticmethod
    def softmax(input, dim=-1):
        return F.softmax(input, dim=dim)

    def freeze_weights(self, layer):
        for name, value in layer.named_parameters():
            value.requires_grad = False

    def init_pretrained_model(self, pretrained_model_class, freeze=False, only_structure=None):
        """
        初始化预训练模型并调整模型词表大小
        :param pretrained_model_class:
        :return: Model
        """
        config = AutoConfig.from_pretrained(
            self.config.pretrain_model,
            cache_dir=self.config.cache_dir,
            **self.config.model_hyparameters if self.config.model_hyparameters else {}
        )
        model = pretrained_model_class.from_config(config)
        only_structure = only_structure if only_structure is not None else self.config.only_structure
        if only_structure:
            model.resize_token_embeddings(self.tokenizer.vocab_size)
            model.init_weights()
        else:
            model = model.from_pretrained(self.config.pretrain_model.split(":")[-1],
                                          config=config,
                                          cache_dir=self.config.cache_dir)
            model.resize_token_embeddings(self.tokenizer.vocab_size)
        if freeze:
            self.freeze_weights(model)
        model = model.train()
        return model

    def get_pad_mask(self, input_tensor):
        return input_tensor.eq(self.tokenizer.pad_token_id).to(input_tensor.device)

    @staticmethod
    def get_subsequent_mask(sequence):
        _, len_s = sequence.size()
        subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=sequence.device), diagonal=1).bool()
        return subsequent_mask

    def get_pad_and_subsequent_mask(self, src, tgt):
        """
        获取pad mask和tgt的subsequent mask, src的注意力mask为全1
        :param src:
        :param tgt:
        :return:
        """
        mask_pack = Result()
        src_key_padding_mask = self.get_pad_mask(src)
        tgt_key_padding_mask = self.get_pad_mask(tgt)
        tgt_mask = self.get_subsequent_mask(tgt)
        mask_pack.add(src_mask=None,
                      tgt_mask=tgt_mask,
                      src_key_padding_mask=src_key_padding_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask)
        return mask_pack

    def get_embedding_layer(self, hidden_size=None, vocab_size=None, pretrain_embedding=None):
        class Embeddings(nn.Module):
            def __init__(self, hidden_size, vocab_size, pretrain_embedding=None):
                super(Embeddings, self).__init__()
                if pretrain_embedding is not None:
                    self.lut = pretrain_embedding
                    self.hidden_size = pretrain_embedding.embedding_dim if isinstance(pretrain_embedding,
                                                                                      nn.Embedding) else pretrain_embedding.word_embeddings.embedding_dim
                elif hidden_size is not None and vocab_size is not None:
                    self.lut = nn.Embedding(vocab_size, hidden_size)
                    self.hidden_size = hidden_size
                else:
                    raise Exception("Embedding的参数为空")

            def forward(self, x):
                return self.lut(x) * math.sqrt(self.hidden_size)

        return Embeddings(hidden_size, vocab_size, pretrain_embedding)

    def get_attn(self, query, key, value, mask=None):
        """
        点积注意力，mask为value的padding mask，mask中为1的部分将被置为-1e9
        :param query: batch_size, src_seq_len, hidden_size
        :param key: batch_size, tgt_seq_len, hidden_size
        :param value: batch_size, tgt_seq_len, hidden_size
        :param mask: batch_size, src_seq_len, tgt_seq_len
        :param dropout:
        :return:
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).repeat(1, scores.size(1), 1) == 1, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
