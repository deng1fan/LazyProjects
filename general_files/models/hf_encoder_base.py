import torch
from general_files.models.base_files.pl_base_model import BasePLModel
from rich.console import Console
from general_files.utils.common_util import Result


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
                                output_hidden_states=True,
                                labels=labels,
                                attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
                                )

        result.add(logits=outputs["logits"])
        if "loss" in outputs:
            result.add(loss=outputs["loss"])
        else:
            result.add(predict_labels=torch.argmax(outputs["logits"], dim=-1).squeeze(-1))
        return result

    def generate_step(self, input_ids, decoder_input_ids, past_result, **other_features):
        return self(input_ids,
                    decoder_input_ids=decoder_input_ids,
                    past_result=past_result,
                    **other_features)

    def print_model_introduction(self):
        # 新建模型的话，需要补充模型的使用说明，例如模型的输入输出形式等
        console = Console(color_system='256', style="cyan")
        # 打印模型使用说明
        console.print("[bold]························································", justify='center')
        console.print("[bold green]模型说明!", justify='center')
        console.print("[bold cyan]1. 极其简单的HuggingFace Encoder模型，支持分类任务", justify='center')
        console.print("[bold cyan]2. 最普通的Seq2Seq架构，输入没有经过任何处理直接送进模型，得到输出", justify='center')
        console.print("[bold]························································", justify='center')
        pass
