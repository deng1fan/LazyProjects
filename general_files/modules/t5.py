import copy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import T5PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from general_files.utils.common_util import Result, get_logger

log = get_logger(__name__)


class CustomModel(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    ###############################################
    # 0、修改初始化
    ###############################################
    def __init__(self, config, hyparam, tokenizer):
        super().__init__(config)
        self.model_dim = config.d_model
        self.hyparam = hyparam
        self.tokenizer = tokenizer
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, self.hyparam.vocab_size, bias=True)
        
        self.dropout = nn.Dropout(self.hyparam.dropout)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        
        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    ###############################################
    # 1、修改原始输入参数
    ###############################################
    def forward(
        self,
        # batch, seq_len
        input_ids=None,
        # batch, seq_len
        labels=None,
        # 无关紧要
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # 其他参数或特征
        **other_features  # 如果有其他特征参数，必须加decoder前缀，否则该参数会经过encoder的forward函数
    ):
        ###############################################
        # 2、初始化返回值以及返回类型
        ###############################################
        result = Result()
        if other_features["decoder_stage"] == "test":
            return_dict = True

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        ###############################################
        # 编码
        ###############################################
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        encoder_last_hidden_state = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            encoder_last_hidden_state = encoder_last_hidden_state.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )
                
        ###############################################
        # 解码
        ###############################################
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        decoder_last_hidden_state = decoder_outputs[0]

        ###############################################
        # 模型输出层
        ###############################################
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            tied_decoder_output = decoder_last_hidden_state * (self.model_dim**-0.5)
            
        lm_logits = self.lm_head(tied_decoder_output)

        ###############################################
        # 计算Loss
        ###############################################
        loss = 0
        
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = (
                loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                if loss == 0
                else loss
                + loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            )
            result.add(lm_loss=loss.detach())
 

        ###############################################
        # 3、包装返回值
        ###############################################
        if other_features["decoder_stage"] != "test" and (
            loss != loss or isinstance(loss, int)
        ):
            raise Exception("Loss为Nan或无梯度，请先检查数据正确性以及超参中 Loss 是否正确选择！")
        if other_features["decoder_stage"] != "test":
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs + (result,)
            return ((loss,) + output) if loss is not None else output
        else:
            return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

    def CrossEntropyLoss(self, logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return loss

        
    ###############################################
    # 5、修正未被向量化的参数扩充
    ###############################################
    def expand_custom_inputs(self, custom_inputs):
        expand_size = self.hyparam.num_return_sequences * self.hyparam.beam_size
        expanded_return_idx = (
            torch.arange(custom_inputs.shape[0])
            .view(-1, 1)
            .repeat(1, expand_size)
            .view(-1)
            .to(custom_inputs.device)
        )
        return custom_inputs.index_select(0, expanded_return_idx)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        other_features = {}
        for k in kwargs.keys():
            if isinstance(kwargs[k], torch.Tensor):
                other_features[k] = self.expand_custom_inputs(kwargs[k])
            else:
                # todo 待将普通数组进行扩充
                other_features[k] = kwargs[k]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            **other_features,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            log.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()
