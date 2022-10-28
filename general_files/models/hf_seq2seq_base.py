
"""
Author: D-Yifan 553192215@qq.com
Date: 2022-08-17 13:06:56
LastEditors: D-Yifan 553192215@qq.com
LastEditTime: 2022-10-13 15:08:29
FilePath: general_files/models/hf_seq2seq_base.py
Description: 

Copyright (c) 2022 by D-Yifan 553192215@qq.com, All Rights Reserved. 
"""
import torch
import torch.nn as nn
from general_files.models.pl_base_model import BasePLModel
from rich.console import Console
from general_files.utils.common_util import Result
from pytorch_lightning.utilities import rank_zero_only


class ModelNet(BasePLModel):
    def __init__(self, config, tokenizer, as_pipeline=False):
        super(ModelNet, self).__init__(config, tokenizer)
        self.model_type = self.model_mode[config.hf_model_type if not as_pipeline else config.pipline_model_type]
        self.backbone = self.init_pretrained_model(self.model_type)  # 实例化对象

        if self.config.use_param_noise:
            for name, para in self.backbone.named_parameters():
                self.backbone.state_dict()[name][:] += (
                    (torch.rand(para.size()) - 0.5)
                    * config.noise_lambda
                    * torch.std(para)
                )

    def forward(
        self,
        # batch, seq_len
        input_ids,
        # batch, seq_len
        labels=None,
        # 其他参数或特征
        past_result=None,  # 生成的时候借用此可以省去每次编码
        **other_features  # 如果有其他特征参数，建议加decoder前缀，保持框架一致性
    ):
        result = Result()
        outputs = self.backbone(
            input_ids=input_ids,
            labels=torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
            if labels is not None
            else None,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        logits = torch.log_softmax(outputs["logits"], dim=-1)
        result.add(logits=logits)
        if labels is not None:
            result.add(labels=labels)
            loss = self.NLLLoss(logits=logits, labels=labels)
            result.add(loss=loss, lm_loss=loss)
        return result

   