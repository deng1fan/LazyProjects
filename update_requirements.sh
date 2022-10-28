#!/bin/bash
###
# @Author: Deng Yifan 553192215@qq.com
# @Date: 2022-09-18 12:23:05
# @LastEditors: appleloveme 52270975+appleloveme@users.noreply.github.com
# @LastEditTime: 2022-10-28 21:22:33
# @FilePath: /dg_templete/update_requirements.sh
# @Description:
#
# Copyright (c) 2022 by Deng Yifan 553192215@qq.com, All Rights Reserved.
###

# 首次请安装pipreqs
# pip install pipreqs

# 为保险起见，记得将原来的备份
pipreqs ./ --encoding=utf8 --force