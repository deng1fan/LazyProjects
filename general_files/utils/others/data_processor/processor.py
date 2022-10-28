'''
Author: D-Yifan 553192215@qq.com
Date: 2022-08-19 09:57:04
LastEditTime: 2022-10-28 18:10:53
FilePath: /dg_templete/general_files/utils/others/data_processor/processor.py
Description: 

Copyright (c) 2022 by D-Yifan 553192215@qq.com, All Rights Reserved. 
'''
import importlib


def get_data_processor(config, tokenizer=None, only_test=False):
    module_path = config.logger_project + '.data_processor.' + config.dataset_processor
    processor_name = 'Processor'
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as r:
        raise ValueError(f"Please add a processor for this : {config.dataset_processor}")
    except Exception as r:
        raise Exception('未知错误: %s' % r)
    processor_class = getattr(module, processor_name)
    processor = processor_class(config, tokenizer, only_test)  # 实例化对象
    return processor
