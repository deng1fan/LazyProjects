'''
Author: appleloveme 553192215@qq.com
Date: 2022-11-16 08:31:10
LastEditors: appleloveme 553192215@qq.com
LastEditTime: 2022-12-03 12:56:51
FilePath: /codes_frame/third_party_eval.py
Description: 

Copyright (c) 2022 by appleloveme 553192215@qq.com, All Rights Reserved. 
'''
##########################################################################
#
#
#        ______                  __   ___  __
#        |  _  \                 \ \ / (_)/ _|
#        | | | |___ _ __   __ _   \ V / _| |_ __ _ _ __
#        | | | / _ \ '_ \ / _` |   \ / | |  _/ _` | '_ \
#        | |/ /  __/ | | | (_| |   | | | | || (_| | | | |
#        |___/ \___|_| |_|\__, |   \_/ |_|_| \__,_|_| |_|
#                          __/ |
#                         |___/
#
#
# Github: https://github.com/D-Yifan
# Zhi hu: https://www.zhihu.com/people/deng_yifan
#
##########################################################################

# -*- coding: utf-8 -*-
from omegaconf import DictConfig
import os
import hydra
import importlib

###############################################
# 📍📍📍 设置环境变量
###############################################
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "False"


@hydra.main(version_base="1.2", config_path="configs/", config_name="third_party_eval.yaml")
def eval(config: DictConfig):

    from general_files.utils.model_util import get_eval_metrics

    data_transformer_name = 'trans'
    try:
        module = importlib.import_module(config.script_path.replace('.py', '').replace('/', '.'))
    except ModuleNotFoundError as r:
        raise ValueError(
            f"Please add a third party eval data transformer located at: {config.script_path}.")
    except Exception as r:
        raise Exception('未知错误 %s' % r)
    data_transformer = getattr(module, data_transformer_name)

    outputs = data_transformer()

    get_eval_metrics(outputs, config)


if __name__ == "__main__":

    eval()
