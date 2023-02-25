'''
Author: appleloveme 553192215@qq.com
Date: 2022-10-30 22:11:56
LastEditors: appleloveme 553192215@qq.com
LastEditTime: 2022-12-05 10:03:40
FilePath: /codes_frame/custom_test_outputs/data_transformer.py
Description: 

Copyright (c) 2022 by appleloveme 553192215@qq.com, All Rights Reserved. 
'''

import jsonlines

def transform(config):
    generated_data_path = "/custom_test_outputs/generated_maxHist1_maxLen100_p0.6.jsonl"
    generated_sents = []
    responses = []
    knows = []
    with open(generated_data_path, 'r+', encoding='utf-8') as f:
        for row in jsonlines.Reader(f):
            generated_sent = row['generated_response'][0]
            gold_sent = row['response']
            know = row['knowledge'].lower()
            # know = row['knowledge']
            knows.append(know)
            responses.append(gold_sent)
            generated_sents.append(generated_sent)

    outputs = {
        "generated_seqs": generated_sents,
        "reference": responses,
        "bert_score_reference": knows,
        "f1_reference": knows,
        "q2_reference": knows,
    }

    return outputs