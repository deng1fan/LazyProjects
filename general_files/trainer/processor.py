'''
Author: appleloveme 52270975+appleloveme@users.noreply.github.com
Date: 2022-10-28 11:51:10
LastEditors: appleloveme 52270975+appleloveme@users.noreply.github.com
LastEditTime: 2022-10-28 21:27:04
FilePath: /dg_templete/general_files/trainer/processor.py
Description: 

Copyright (c) 2022 by appleloveme 52270975+appleloveme@users.noreply.github.com, All Rights Reserved. 
'''
import importlib


def get_trainer_processor(config):
    trainer_processor = config.trainer_processor
    module_path = 'general_files.trainer.' + trainer_processor
    processor_name = 'ModelTrainer'
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as r:
        raise ValueError(f"Please add a trainer processor named like this: {trainer_processor}.")
    except Exception as r:
        raise Exception('未知错误 %s' % r)
    processor_class = getattr(module, processor_name)
    return processor_class

