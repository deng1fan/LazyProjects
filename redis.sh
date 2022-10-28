###
# @Author: Deng Yifan 553192215@qq.com
# @Date: 2022-09-23 14:36:26
# @LastEditors: appleloveme 52270975+appleloveme@users.noreply.github.com
# @LastEditTime: 2022-10-28 21:21:08
# @FilePath: /dg_templete/redis.sh
# @Description:
#
# Copyright (c) 2022 by Deng Yifan 553192215@qq.com, All Rights Reserved.
###


base_path=$(pwd)
export PYTHONPATH=$base_path

nohup python $base_path/general_files/utils/others/redis_client/maintain_redis_data.py > $base_path/nohup/redis_nohup.txt 2>&1
