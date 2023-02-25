'''
Author: appleloveme 553192215@qq.com
Date: 2022-11-13 14:59:54
LastEditors: appleloveme 553192215@qq.com
LastEditTime: 2022-11-18 08:12:25
FilePath: /codes_frame/data/wow_faithdial/basic_preprocess.py
Description: 

Copyright (c) 2022 by appleloveme 553192215@qq.com, All Rights Reserved. 
'''
from general_files.utils.others.stanford_nlp.stanfordnlp import StanfordCoreNLP
import os
from tqdm import tqdm
import general_files.utils.common_util as utils
from general_files.utils.common_util import Result, print_error_info
from general_files.utils.others.data_processor.base_data_processor import BaseProcessor
from general_files.utils.data_util import (
    save_as,
    read_by,
)
from data.utils import caller as data_method_caller
from general_files.modules.pipeline import Pipeline
import random


log = utils.get_logger(__name__)


class Processor(BaseProcessor):
    def __init__(self, config, tokenizer, only_test):
        super(Processor, self).__init__(config, tokenizer, only_test)

    def read_data(self, stage):
        data_ckpt_version = self.config.dataset_version
        data_ckpt_split = self.config.get("dataset_split", "")
        data_path = f"{self.public_dataset_path}/preprocessed_data_{data_ckpt_version}_{data_ckpt_split}"
        if not os.path.exists(data_path + ".pt"):
            all_rows = self.preprocess_data(data_path)[stage]
        else:
            all_rows = read_by(data_path + ".pt", f"预处理的{stage}数据集")[stage]
        if self.config.fast_run:
            all_rows = all_rows[:4]
        return self.get_rows(all_rows, stage)

    def data_process(self, data, stage, *args, **kargs):
        return data

    def tokenize_data(self, batch, stage=None):
        result = Result()
        result.merge_or_update(
            self.tokenizer(
                {
                    "input_ids": batch["history"],
                    "labels": batch["response"],
                },
                padding="max_length",
                max_length=self.config.encoder_max_length,
                truncation=True,
                only_input_ids=True,
                add_special_tokens=True,
            )
        )
        return result

    def map_column(self, test_output):
        # test_output：Dataset类型，使用rename_column修改列名
        # test_output = test_output.rename_column('response', '')
        if "target" in test_output.column_names:
            test_output = test_output.map(
                lambda batch: {
                    "reference": batch["response"],
                    "bert_score_reference": batch["knowledge"].lower(),
                    "f1_reference": batch["knowledge"].lower(),
                    "q2_reference": batch["knowledge"].lower(),
                },
                desc="数据集映射",
            )
        return test_output

    def preprocess_data(self, data_path):
        """
        原始数据集转换为Query生成模型训练所需的格式
        """
        wow_data = read_by(data_path.replace(
            self.config.dataset, 'wizard_of_wikipedia') + '.pt', "原始 WOW 数据集")
        faith_dial_data = read_by(data_path.replace(
            self.config.dataset, 'faith_dial') + '.pt', "原始 FaithDial 数据集")

        for stage in ["train", "valid", "test"]:
            wow_all_rows = wow_data[stage]
            for dialog_id, dialog in enumerate(tqdm(wow_all_rows, desc=f"调整{stage}数据集格式")):
                for uttr_id, uttr in enumerate(dialog["utterances"]):
                    ###############################################
                    # 数据集合并
                    ###############################################
                    wow_data[stage][dialog_id]["utterances"][uttr_id]['dataset'] = 'wow'
            wow_keys = uttr.keys()

            faith_dial_rows = faith_dial_data[stage]
            for dialog_id, dialog in enumerate(tqdm(faith_dial_rows, desc=f"合并{stage}数据集")):
                for uttr_id, uttr in enumerate(dialog["utterances"]):
                    ###############################################
                    # 数据集合并
                    ###############################################
                    faith_dial_data[stage][dialog_id]["utterances"][uttr_id]['dataset'] = 'faith_dial'
                    pop_keys = []
                    for key in uttr.keys():
                        if key not in wow_keys:
                            pop_keys.append(key)
                    for key in pop_keys:
                        faith_dial_data[stage][dialog_id]["utterances"][uttr_id].pop(
                            key)

            wow_data[stage].extend(faith_dial_data[stage])
            random.shuffle(wow_data[stage])

        all_data = wow_data
        save_as(all_data, data_path, data_name="预处理数据集")

        return all_data
