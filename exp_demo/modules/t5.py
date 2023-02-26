import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack
from general_files.utils.common_util import Result, get_logger
from general_files.modules.t5 import CustomModel as T5Model
from general_files.modules.attention import ScaledDotProductAttention
import general_files.modules.similarity_calculator as sim_cal

log = get_logger(__name__)


class CustomModel(T5Model):

    ###############################################
    # 0、修改初始化
    ###############################################
    def __init__(self, config, hyparam, tokenizer):
        super().__init__(config, hyparam, tokenizer)
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
        self.lm_head = nn.Linear(
            config.d_model, self.hyparam.vocab_size, bias=True)
        self.init_weights()

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
            ###############################################
            # 添加token type 注入
            ###############################################
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
            encoder_last_hidden_state = encoder_last_hidden_state.to(
                self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(
                    self.decoder.first_device)
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
            return_dict=return_dict,
        )
        decoder_last_hidden_state = decoder_outputs[0]

        ###############################################
        # 模型输出层
        ###############################################
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            self.know_attn_layer = self.know_attn_layer.to(
                self.encoder.first_device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            tied_decoder_output = decoder_last_hidden_state * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(tied_decoder_output)

        ###############################################
        # 3、包装返回值
        ###############################################
        if other_features["decoder_stage"] in ["train", "valid"]:
            output = (lm_logits,) + (decoder_last_hidden_state, encoder_last_hidden_state) + \
                     decoder_outputs[1:] + encoder_outputs + (result,)
            return output
        else:
            return Seq2SeqLMOutput(
                loss=None,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
