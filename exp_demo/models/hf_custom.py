
import torch
from general_files.models.pl_base_model import BasePLModel
from general_files.utils.common_util import Result
import importlib
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import general_files.modules.similarity_calculator as sim_cal
from general_files.modules.info_nce import InfoNCE


class ModelNet(BasePLModel):
    def __init__(self, config, tokenizer, as_pipeline=False):
        super(ModelNet, self).__init__(config, tokenizer)
        if ':' in self.config.pretrain_model:
            model_processor_name = self.config.pretrain_model.split(':')[0]
            module_path = config.logger_project + '.modules.' + model_processor_name
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError as r:
                raise ValueError(f"Please add a processor for this model: {model_processor_name}\n"
                                 f"Error module path：{module_path}")
            processor_name = 'CustomModel'
            processor_class = getattr(module, processor_name)
            pretrain_model = self.config.pretrain_model.split(':')[-1]
            self.backbone = processor_class.from_pretrained(pretrain_model,
                                                            cache_dir=self.config.cache_dir,
                                                            hyparam=config,
                                                            tokenizer=tokenizer,)
            self.backbone.resize_token_embeddings(self.tokenizer.vocab_size)
            self.backbone = self.backbone.train()
        else:
            self.model_type = self.model_mode[config.hf_model_type if not as_pipeline else config.pipline_model_type]
            self.backbone = self.init_pretrained_model(self.model_type, as_pipeline)  # 实例化对象
    
        if self.config.get("use_param_noise", False):
            for name, para in self.backbone.named_parameters():
                self.backbone.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * config.noise_lambda * torch.std(para)

        if "bow_loss" in self.config.get("loss"):
            self.bow_layer = nn.Linear(
                self.backbone.config.d_model, self.config.vocab_size)
            self.bow_attn_layer = nn.Linear(
                2 * self.backbone.config.d_model, self.backbone.config.d_model)
        
    def forward(self,
                # batch, seq_len
                input_ids,
                # batch, seq_len
                labels=None,
                # 其他参数或特征
                past_result=None,  # 生成的时候借用此可以省去每次编码
                **other_features  # 如果有其他特征参数，建议加decoder前缀，保持框架一致性
                ):
        result = Result()
        other_features['decoder_stage'] = self.stage
        outputs = self.backbone(input_ids=input_ids,
                                output_hidden_states=True,
                                output_attentions=True,
                                labels=torch.where(labels == self.tokenizer.pad_token_id, -100, labels) if labels is not None else None,
                                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                                return_dict=False,
                                use_cache=True,
                                **other_features,
                                )
        lm_logits = outputs[0]
        decoder_last_hidden_state = outputs[1]
        encoder_last_hidden_state = outputs[2]
        model_result = outputs[-1]

        if len(self.config.get("loss")) < 1:
            raise Exception("请至少选择一个损失函数！")
        loss = 0
        ###############################################
        # 计算词袋损失
        ###############################################
        if "bow_loss" in self.config.get("loss") and self.stage != "test":
            bow_logits = []
            for b, ids in enumerate(other_features["decoder_seg_label_idx"]):
                batch_bow_logits = []
                for i, idx in enumerate(ids):
                    curr_logit = decoder_last_hidden_state[b, idx, :].unsqueeze(0)
                    bow_label_dist = curr_logit.repeat(
                        encoder_last_hidden_state[b].shape[0], 1)
                    bow_out_logit = self.bow_attn_layer(torch.cat(
                        [bow_label_dist, encoder_last_hidden_state[b]], dim=-1
                    ))
                    bow_out_logit = torch.tanh(bow_out_logit).sum(0)

                    bow_output = self.bow_layer(bow_out_logit)
                    batch_bow_logits.append(bow_output)
                bow_logits.append(batch_bow_logits)

            bow_labels = other_features["decoder_seg_bows"]
            bow_loss = self.compute_bow_loss(bow_logits, bow_labels)
            result.add(bow_loss=bow_loss)
            loss += bow_loss

        if "cosine_loss" in self.config.get("loss") and self.stage != "test":
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            label_dis = self.backbone.shared(labels)

            sim = sim_cal.CosineSimilarity()

            sim_score = -(sim(decoder_last_hidden_state, label_dis) - 1)
            weighted_lm_logits = torch.mul(sim_score.unsqueeze(-1).repeat(1, 1, lm_logits.shape[-1]), lm_logits)
            cosine_loss = loss_fct(weighted_lm_logits.view(-1, weighted_lm_logits.size(-1)), torch.where(labels == self.tokenizer.pad_token_id, -100, labels).view(-1))

            # cosine_loss = outputs[-1]['recontra_loss']
            result.add(cosine_loss=cosine_loss)
            loss += cosine_loss

        if "infonce_loss" in self.config.get("loss") and self.stage != "test":
            info_nce_loss = 0
            label_dis = self.backbone.shared(labels)
            cal_info_nce = InfoNCE(negative_mode='paired', use_weighted=self.config.get("use_weighted_info_nce"))
            for bth in range(labels.shape[0]):
                positive_example = label_dis[bth]
                negative_examples = torch.cat([label_dis[bth + 1:], label_dis[:bth]], dim=0).transpose(0, 1)
                info_nce_loss += cal_info_nce(decoder_last_hidden_state[bth].squeeze(1), positive_example, negative_examples)
            info_nce_loss = info_nce_loss / (labels.shape[0] - 1)
            result.add(info_nce_loss=info_nce_loss)
            loss += info_nce_loss

        if "sent_level_infonce_loss" in self.config.get("loss") and self.stage != "test":
            label_dis = self.backbone.shared(labels)
            info_nce_loss = 0
            cal_info_nce = InfoNCE(negative_mode='paired', use_weighted=self.config.get("use_weighted_info_nce"))
            positive_example = label_dis.mean(dim=1)
            negative_examples = []
            for bth in range(labels.shape[0]):
                negative_examples.append(
                    torch.cat([label_dis[bth + 1:].mean(dim=1), label_dis[:bth].mean(dim=1)], dim=0).unsqueeze(0))
            negative_examples = torch.cat(negative_examples, dim=0)
            info_nce_loss += cal_info_nce(decoder_last_hidden_state.mean(dim=1), positive_example, negative_examples)
            info_nce_loss = info_nce_loss / (labels.shape[0] - 1)
            result.add(info_nce_loss=info_nce_loss)
            loss += info_nce_loss

        if "lm_loss" in self.config.get("loss") and labels is not None:
            result.add(labels=labels)
            if self.config.get("use_know_aware") and 'copy' in self.config.pretrain_model:
                final_logit = 0.5 * F.softmax(lm_logits, dim=-1) + 0.5 * F.softmax(model_result['know_dist'], dim=-1)
                nll_loss_fun = nn.NLLLoss(ignore_index=self.tokenizer.pad_token_id)
                lm_loss = nll_loss_fun(torch.log(final_logit).view(-1, final_logit.shape[-1]), labels.view(-1))

            else:
                lm_loss = self.CrossEntropyLoss(logits=lm_logits, labels=labels)
            result.add(lm_loss=lm_loss)
            loss += lm_loss

        if self.stage != "test" and (
                loss != loss or isinstance(loss, int)
        ):
            raise Exception("Loss为Nan或无梯度，请先检查数据正确性以及超参中 Loss 是否正确选择！")

        result.add(loss=loss)
        return result

    def compute_bow_loss(self, bow_logits, bow_labels):
        bow_loss = 0
        count = 0
        for b, logits in enumerate(bow_logits):
            for i, logit in enumerate(logits):
                logit = logit.repeat(len(bow_labels[b][i]), 1)
                bow_label = (
                    torch.LongTensor(bow_labels[b][i]).to(logit.device)
                    if not isinstance(bow_labels[b][i], torch.Tensor)
                    else bow_labels[b][i].type_as(torch.LongTensor()).to(logit.device)
                )
                b_loss = self.CrossEntropyLoss(logit, bow_label)
                bow_loss += b_loss
                count += 1
        return bow_loss / count if count > 0 else bow_loss


   