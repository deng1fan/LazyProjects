###
# @Author: appleloveme 52270975+appleloveme@users.noreply.github.com
# @Date: 2022-10-28 11:51:10
# @LastEditors: appleloveme 52270975+appleloveme@users.noreply.github.com
# @LastEditTime: 2022-10-28 21:21:32
# @FilePath: /dg_templete/gpu_watch.sh
# @Description:
#
# Copyright (c) 2022 by appleloveme 52270975+appleloveme@users.noreply.github.com, All Rights Reserved.
###

# (pip 安装)首次使用请打开此选项进行安装
# pip install gpu-watchmen -i https://pypi.org/simple

# (源码安装  推荐)
# git clone https://github.com/Spico197/watchmen.git [your local path]
# 首次使用请打开此选项进行安装
# pip install -r requirements.txt
# cd ..
# pip install -e .
# cd watchmen
echo 'Open the following link to local browser: http://192.168.80.1:62333'
nohup python -m watchmen.server 1>watchmen.log 2>&1 &
