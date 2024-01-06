<!--
 * @Author: Deng Yifan 553192215@qq.com
 * @Date: 2022-08-26 14:02:16
 * @LastEditors: Deng Yifan 553192215@qq.com
 * @LastEditTime: 2022-08-26 16:54:56
 * @FilePath: /dg_templete/README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by Deng Yifan 553192215@qq.com, All Rights Reserved. 
-->

![](https://img.shields.io/badge/License-GNU%20General%20Public%20License%20v3.0-green)
![](https://img.shields.io/badge/Python-3.8-blue)
![](https://img.shields.io/badge/知乎-邓什么邓-orange)


# Towards Faithful Dialogs via Focus Learning 论文代码正在整理中，先贴出FCE 的核心计算代码片段


论文核心代码：
```
    class CosineSimilarity(torch.nn.Module):
        def forward(self, tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
    cal_sim = CosineSimilarity()
    knowledge_emb = model.get_input_embeddings()(
        knowledges
    )
    sim_dist = -cal_sim(knowledge_emb, labels_emb)
    sim_score = -torch.log(sim_dist + 1 + self.config.get("fce_lamda", 0.01))+ 1
    
    weighted_lm_logits = torch.mul(sim_score.unsqueeze(-1).repeat(1, 1, logits.shape[-1]), logits)
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    fce_loss = loss_fct(weighted_lm_logits.view(-1, weighted_lm_logits.size(-1)),
                                   torch.where(labels == self.tokenizer.pad_token_id, -100, labels).view(-1))
```


