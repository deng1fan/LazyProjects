# encoding=utf-8
import os
from tqdm import tqdm
import general_files.utils.common_util as utils
from general_files.utils.common_util import Result
from general_files.utils.others.data_processor.base_data_processor import BaseProcessor
from general_files.utils.data_util import save_as, read_by, flat


log = utils.get_logger(__name__)


class Processor(BaseProcessor):
    def __init__(self, config, tokenizer, model, only_test):
        super(Processor, self).__init__(config, tokenizer, model, only_test)

    def read_data(self, stage):
        if self.config.tokenize_method == 'auto':
            data_path = f'{self.config.data_path}{self.config.pretrain_model}_data_pre.pt'
        else:
            data_path = f'{self.config.data_path}{self.config.tokenize_method}_data_pre.pt'
        if not os.path.exists(data_path):
            all_rows = self.preprocess_data(data_path)[stage]
        else:
            all_rows = read_by(data_path, f"预处理的{stage}数据集")[stage]
        if self.config.fast_run:
            all_rows = all_rows[:4]
        return self.get_rows(all_rows, stage)

    def get_rows(self, all_rows, stage):
        rows = Result()
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        sep_token = self.tokenizer.sep_token
        user_token = '<user>'
        bot_token = '<bot>'
        knowledge_token = '<knowledge>'

        for dialog in tqdm(all_rows, desc="格式化输入输出"):
            for uttr in dialog['utterances']:
                ###############################################
                # 基础数据处理
                ###############################################
                all_history = []
                for i, h in enumerate(uttr['history']):
                    if i % 2 == 0:
                        all_history.append("<user> " + h)
                    else:
                        all_history.append("<bot> " + h)
                history = ' '.join(flat(all_history))

                knowledge = uttr['knowledge']
                response = uttr['response']

                ###############################################
                # 构建模型输入输出格式
                ###############################################
                input = ' '.join(flat([
                    knowledge,
                    history,
                ]))
                target = response

                row = Result(
                    source=input,
                    target=target,
                    # >>> other_features <<<
                    response=response,
                    knowledge=knowledge,
                    history=history,
                )
                rows.append_values(row)
        return rows

    def tokenize_data(self, batch, stage=None):
        result = Result()
        result.merge_or_update(self.tokenizer(
            {
                "input_ids": batch['source'],
                "labels": batch['target'],
                "decoder_response": batch['response'],
                "decoder_knowledge": batch['knowledge'],
                "decoder_history": batch['history'],
            },
            padding='max_length',
            max_length=self.config.encoder_max_length,
            truncation=True,
            only_input_ids=True,
            add_special_tokens=True,
        ))
        return result

    def map_column(self, test_output):
        # test_output：Dataset类型，使用rename_column修改列名
        # test_output = test_output.rename_column('response', '')
        if 'target' in test_output.column_names:
            test_output = test_output.map(
                lambda batch: {'reference': batch['response'],
                               'bert_score_reference': batch['knowledge'],
                               'f1_reference': batch['knowledge'],
                               },
                desc='数据集映射')
        return test_output

    def preprocess_data(self, data_path):
        """
        原始数据集转换为Query生成模型训练所需的格式
        """
        # 数据集下载地址：http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
        ori_data = {
            "train": read_by(self.config.ori_data_path + "train.json", data_name="训练集"),
            "valid": read_by(self.config.ori_data_path + "valid_random_split.json", data_name="验证集"),
            "test": read_by(self.config.ori_data_path + "test_random_split.json", data_name="测试集"),
        }
        processed_data = {
            "train": [],
            "valid": [],
            "test": [],
        }
        for stage in ["train", "valid", "test"]:
            data = ori_data[stage]
            for i, item in enumerate(tqdm(data, desc=f"预处理{stage}数据集：")):
                dialog_history = []
                utterances = []
                for j, dialog in enumerate(item['dialog']):
                    response = dialog['text'].replace(".", " . ").replace(",", " , ").replace(":", " : ").replace(
                        "!", " ! ").replace("?", " ? ").replace("'", " ' ")
                    if len(dialog_history) < 1:
                        dialog_history.append('__topic__:' + item['chosen_topic'])
                    history = dialog_history.copy()
                    dialog_history.append(response)
                    speaker = "Wizard" if "Wizard" in dialog["speaker"] else "Apprentice"
                    is_wizard = True if speaker == "Wizard" else False
                    if is_wizard:
                        if len(list(dialog['checked_sentence'].values())) < 1:
                            topic = list(dialog['checked_passage'].values())[0]
                            passage = [topic]
                            for p in dialog['retrieved_passages']:
                                if topic in p:
                                    passage = p[topic]
                            knowledge = passage[0]
                        else:
                            knowledge = list(dialog['checked_sentence'].values())[0]

                    uttr = {
                        "history": history,
                        "knowledge": knowledge,
                        "response": response,
                    }
                    utterances.append(uttr)

                processed_data[stage].append({
                    "utterances": utterances
                })
        data_path = ''.join(data_path.split(".")[:-1])
        if self.config.tokenize_method == 'auto':
            save_as(processed_data, data_path, data_name="预处理数据集")
        else:
            save_as(processed_data, data_path, data_name="预处理数据集")
        return processed_data