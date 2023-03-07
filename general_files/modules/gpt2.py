import torch
import torch.nn as nn
from typing import Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import GPT2Model, GPT2PreTrainedModel
from general_files.utils.common_util import Result, get_logger

log = get_logger(__name__)

###############################################
# 0、修改类名
###############################################
class CustomModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    ###############################################
    # 1、修改初始化
    ###############################################
    def __init__(self, config, hyparam, tokenizer):
        super().__init__(config)
        self.model_dim = config.n_embd
        self.hyparam = hyparam
        self.tokenizer = tokenizer
                
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True


    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()


    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        other_features = {}
        for k in kwargs.keys():
            if isinstance(kwargs[k], torch.Tensor) and k.startswith("decoder_"):
                other_features[k] = self.expand_custom_inputs(kwargs[k])
            else:
                # todo 待将普通数组进行扩充
                other_features[k] = kwargs[k]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            **other_features,
        }


    ###############################################
    # 2、修改原始输入参数
    ###############################################
    def forward(
        self,
        # batch, seq_len
        input_ids=None,
        # batch, seq_len
        labels=None,
        # 无关紧要
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # 其他参数或特征
        **other_features  # 如果有其他特征参数，必须加decoder前缀，否则该参数会经过encoder的forward函数
    ):
        ###############################################
        # 3、初始化返回值以及返回类型
        ###############################################
        result = Result()
        if other_features["decoder_stage"] == "test":
            return_dict = True

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        ###############################################
        # 4、包装返回值
        ###############################################
        if other_features["decoder_stage"] != "test" and (
            loss != loss or isinstance(loss, int)
        ):
            raise Exception("Loss为Nan或无梯度，请先检查数据正确性以及超参中 Loss 是否正确选择！")
        if other_features["decoder_stage"] != "test":
            output = (lm_logits,) + transformer_outputs[1:] + (result,)
            return ((loss,) + output) if loss is not None else output
        else:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )

    
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

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
