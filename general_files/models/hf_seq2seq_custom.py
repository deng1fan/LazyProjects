import torch
import torch.nn as nn
from general_files.models.base_files.pl_base_model import BasePLModel
from rich.console import Console
from general_files.utils.common_util import Result
from pytorch_lightning.utilities import rank_zero_only
import importlib

class ModelNet(BasePLModel):
    def __init__(self, config, tokenizer):
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
            self.model_type = self.model_mode[config.hf_model_type]
            self.backbone = self.init_pretrained_model(self.model_type)  # 实例化对象
        if self.config.use_bow:
            self.bow_layer = nn.Linear(self.backbone.config.d_model, self.tokenizer.vocab_size)
            self.bow_attn_layer = nn.Linear(2 * self.backbone.config.d_model, self.backbone.config.d_model)

        if self.config.use_param_noise:
            for name, para in self.backbone.named_parameters():
                self.backbone.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * config.noise_lambda * torch.std(para)

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
                                **other_features,
                                )
        if labels is not None and self.stage == 'train':
            loss = outputs[0]
            result.add(loss=loss)
            result.add(labels=labels)
            if isinstance(outputs[-1], Result):
                model_result = outputs[-1]
                result.merge_or_update(model_result)
        return result


    @rank_zero_only
    def print_model_introduction(self):
        # 新建模型的话，需要补充模型的使用说明，例如模型的输入输出形式等
        console = Console(color_system='256', style="cyan")
        # 打印模型使用说明
        console.print("[bold]························································", justify='center')
        console.print("[bold cyan]")
        console.print("[bold green]模型草图!", justify='center')
        console.print(
            "[bold cyan]            \u250C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510", justify='center')
        console.print("[bold cyan]            \u2502CrossEntropyLoss\u2502", justify='center')
        console.print(
            "[bold cyan]            \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518", justify='center')
        console.print("[bold cyan]                    \u2502", justify='center')
        console.print("[bold cyan]                    \u2502", justify='center')
        console.print(
            "[bold cyan]      labels  \u25C0\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u25B6 generated_ids", justify='center')
        console.print("[bold cyan]")
        console.print("[bold cyan]                                   \u25B2", justify='center')
        console.print("[bold cyan]                                   \u2502", justify='center')
        console.print("[bold cyan]                                   \u2502", justify='center')
        console.print(
            "[bold cyan]\u250C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510             \u250C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510", justify='center')
        console.print("[bold cyan]\u2502             \u2502             \u2502             \u2502", justify='center')
        console.print(
            "[bold cyan]\u2502   Encoder   \u2502\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u25B6\u2502   Decoder   \u2502", justify='center')
        console.print("[bold cyan]\u2502             \u2502             \u2502             \u2502", justify='center')
        console.print(
            "[bold cyan]\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518             \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518", justify='center')
        console.print("[bold cyan]       \u25B2                           \u25B2", justify='center')
        console.print("[bold cyan]       \u2502                           \u2502", justify='center')
        console.print("[bold cyan]       \u2502                           \u2502", justify='center')
        console.print("[bold cyan]       \u2502                           \u2502", justify='center')
        console.print("[bold cyan]", justify='center')
        console.print("[bold cyan]   input_ids              decoder_input_ids", justify='center')
        console.print("[bold cyan]", justify='center')
        console.print("[bold cyan]", justify='center')
        console.print("[bold cyan]1. 极其简单的HuggingFace模型", justify='center')
        console.print("[bold]························································", justify='center')

        pass
