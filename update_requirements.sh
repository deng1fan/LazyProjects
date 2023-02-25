#!/bin/bash
###
 # @Author: Deng Yifan 553192215@qq.com
 # @Date: 2022-09-18 12:23:05
 # @LastEditors: Deng Yifan 553192215@qq.com
 # @LastEditTime: 2022-09-18 12:25:14
 # @FilePath: /faith_dial/update_requirements.sh
 # @Description: 
 # 
 # Copyright (c) 2022 by Deng Yifan 553192215@qq.com, All Rights Reserved. 
### 

# 首次请安装pipreqs
# pip install pipreqs

# 为保险起见，记得将原来的备份
pipreqs ./ --encoding=utf8 --force