'''
Author: D-Yifan https://github.com/D-Yifan
Date: 2022-10-28 11:55:26
LastEditors: appleloveme 553192215@qq.com
LastEditTime: 2022-10-28 22:12:09
FilePath: /dg_templete/data/utils.py
Description: 

Copyright (c) 2022 by D-Yifan https://github.com/D-Yifan, All Rights Reserved. 
'''

from general_files.utils.common_util import Result
from general_files.utils.data_util import flat, replace_word
from nltk.tokenize import sent_tokenize
import re
from typing import Any, Dict, Iterable, Sequence, Union
import datasets
import numpy as np
import transformers
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from tqdm import tqdm
import logging

logging.getLogger("utils").setLevel(logging.WARNING)


def caller(methods, result, *args, **kwargs):
    result = Result() if result is None else result
    for method in methods:
        result = globals().get(method)(result, *args, **kwargs)
    return result

# 📢  自定义的数据处理方法参数列表需与此保持一致
def clean_text(uttr, *args, **kwargs):
    uttr["response"] = replace_word(
        uttr["response"]
        .replace("U.S.", "America")
        .replace("..", ",")
        .replace(",.", ",")
        .replace(",,", ",")
        .replace("OK.", "OK,")
        .replace("Ok.", "OK,")
        .replace("+", "and")
        .replace("\t", "")
        .replace("?.", ",")
        .replace("!.", ",")
        .replace("\\", "")
        .replace('."', '".')
    )
    # 去掉ghost字符，我也不知道为啥会有这种，不加某些样本无法正确切分
    uttr["knowledge"] = replace_word(
        uttr["knowledge"]
        .replace("U.S.", "America")
        .replace("..", ",")
        .replace(",,", ",")
        .replace(",.", ",")
        .replace("OK.", "OK,")
        .replace("Ok.", "OK,")
        .replace("+", "and")
        .replace("\t", "")
        .replace("?.", ",")
        .replace("!.", ",")
        .replace("\\", "")
        .replace('."', '".')
        .replace("''", '"')
        .replace("Super Smash Bros. Brawl", "Super Smash Bros Brawl")
        .replace('rural."', "rural.")
        .replace("Super Smash Bros. ", "Super Smash Bros ")
        .replace(
            "mental suffering; mental torment",
            "mental suffering and mental torment",
        )
        .replace('torment."', "torment.")
    )

    return uttr


