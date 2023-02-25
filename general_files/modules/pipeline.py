"""
Author: appleloveme 553192215@qq.com
Date: 2022-08-19 15:06:33
LastEditors: appleloveme 553192215@qq.com
LastEditTime: 2022-10-16 20:40:12
FilePath: /faith_dial/general_files/modules/pipeline.py
Description: 

Copyright (c) 2022 by appleloveme 553192215@qq.com, All Rights Reserved. 
"""
import torch.nn as nn
import torch
from general_files.utils.common_util import (
    set_config_gpus,
    init_context,
    RedisClient,
)

class Pipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = set_config_gpus(config)
        self.default_device = config.default_device
        if not config.pipline_model:
            raise Exception("缺少pipline_model参数！")
        model, tokenizer = init_context(config, as_pipeline=True, init_data=False)
        self.tokenizer = tokenizer
        self.model = model.to(self.default_device).eval()

    def forward(self, input_text, input_ids=None, **other_features):
        if not input_ids:
            input_text = [input_text] if isinstance(input_text, str) else input_text
            input_ids = self.tokenizer(input_text, only_input_ids=True)[0]
        input_ids = torch.LongTensor([input_ids]).to(self.default_device)
        other_features["decoder_stage"] = "test"
        generated_ids = self.model.backbone.generate(
            input_ids=input_ids,
            num_beams=self.config.beam_size,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            use_cache=True,
            repetition_penalty=0.9,
            early_stopping=True,
            **other_features,
        )
        generated_sentences = [
            {
                "seqs": self.tokenizer.decode(sent, skip_special_tokens=True),
                "seqs_with_special_tokens": self.tokenizer.decode(
                    sent,
                    skip_special_tokens=False,
                    ignore_tokens=[self.tokenizer.pad_token],
                ),
            }
            for sent in generated_ids
        ]
        return generated_sentences

    def done(self):
        self.model = self.model.to("cpu")
        ###############################################
        # 删除Redis的Gpu占用记录
        ###############################################
        if self.config.task_id:
            redis_client = RedisClient()
            redis_client.deregister_gpus(self.config)
