import torch
import torch.nn as nn
from general_files.models.base_files.pl_base_model import BasePLModel
from rich.console import Console
from general_files.utils.common_util import Result
from pytorch_lightning.utilities import rank_zero_only


class ModelNet(BasePLModel):
    def __init__(self, config, tokenizer):
        super(ModelNet, self).__init__(config, tokenizer)
        self.model_type = self.model_mode[config.hf_model_type]
        self.backbone = self.init_pretrained_model(self.model_type)  # 实例化对象

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
        outputs = self.backbone(input_ids=input_ids,
                                labels=torch.where(labels == self.tokenizer.pad_token_id, -100, labels) if labels is not None else None,
                                attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
                                )
        logits = torch.log_softmax(outputs['logits'], dim=-1)
        result.add(logits=logits)
        if labels is not None:
            result.add(labels=labels)
            loss = self.NLLLoss(logits=logits, labels=labels)
            result.add(loss=loss, lm_loss=loss)
        return result

    @rank_zero_only
    def print_model_introduction(self):
        # 新建模型的话，需要补充模型的使用说明，例如模型的输入输出形式等
        console = Console(color_system='256', style="cyan")
        # 打印模型使用说明
        console.print("[bold]························································", justify='center')
        console.print("[bold cyan]")
        console.print("[bold green]模型示意图!", justify='center')
        console.print(
            "[bold cyan]            \u250C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
        console.print("[bold cyan]            \u2502CrossEntropyLoss\u2502")
        console.print(
            "[bold cyan]            \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")
        console.print("[bold cyan]                    \u2502")
        console.print("[bold cyan]                    \u2502")
        console.print(
            "[bold cyan]      labels  \u25C0\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u25B6 generated_ids")
        console.print("[bold cyan]")
        console.print("[bold cyan]                                   \u25B2")
        console.print("[bold cyan]                                   \u2502")
        console.print("[bold cyan]                                   \u2502")
        console.print(
            "[bold cyan]\u250C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510             \u250C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
        console.print("[bold cyan]\u2502             \u2502             \u2502             \u2502")
        console.print(
            "[bold cyan]\u2502   Encoder   \u2502\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u25B6\u2502   Decoder   \u2502")
        console.print("[bold cyan]\u2502             \u2502             \u2502             \u2502")
        console.print(
            "[bold cyan]\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518             \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")
        console.print("[bold cyan]       \u25B2                           \u25B2")
        console.print("[bold cyan]       \u2502                           \u2502")
        console.print("[bold cyan]       \u2502                           \u2502")
        console.print("[bold cyan]       \u2502                           \u2502")
        console.print("[bold cyan]")
        console.print("[bold cyan]   input_ids              decoder_input_ids")
        console.print("[bold cyan]")
        console.print("[bold cyan]注意：", justify='center')
        console.print("[bold cyan]1. HuggingFace模型套件", justify='center')
        console.print("[bold cyan]2. 用户需要在自定义的、基于HF模型的文件里边客制化训练步骤", justify='center')
        console.print("[bold]························································", justify='center')

        pass