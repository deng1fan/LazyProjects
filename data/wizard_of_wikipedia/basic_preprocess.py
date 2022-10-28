from general_files.utils.others.stanford_nlp.stanfordnlp import StanfordCoreNLP
import spacy
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


log = utils.get_logger(__name__)


class Processor(BaseProcessor):
    def __init__(self, config, tokenizer, only_test):
        super(Processor, self).__init__(config, tokenizer, only_test)

    def read_data(self, stage):
        data_ckpt_version = self.config.dataset_version
        data_path = f"{self.public_dataset_path}/preprocessed_data_{data_ckpt_version}"
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
                    "bert_score_reference": batch["knowledge"],
                    "f1_reference": batch["knowledge"],
                },
                desc="数据集映射",
            )
        return test_output

    def preprocess_data(self, data_path):
        """
        原始数据集转换为Query生成模型训练所需的格式
        """
        all_data = {
            "train": read_by(
                self.public_dataset_path + "/train.json", data_name="训练集"
            ),
            "valid": read_by(
                self.public_dataset_path + f"/valid_{self.config.dataset_split}_split.json", data_name="验证集"
            ),
            "test": read_by(
                self.public_dataset_path + f"/test_{self.config.dataset_split}_split.json", data_name="测试集"
            ),
        }
        processed_data = {
            "train": [],
            "valid": [],
            "test": [],
        }

        process_flow = [
            "clean_text",
            # "your-custom-process-flow" # 你可以在这里添加自定义的数据处理流程，名称对应于 utils 中的方法名
        ]

        ###############################################
        # 读取预处理进度
        ###############################################
        data_ckpt_path = data_path + "_ckpt"
        if os.path.exists(data_ckpt_path + ".pt"):
            data_ckpt = read_by(data_ckpt_path + ".pt", data_name="数据集预处理ckpt")
            sents_static = data_ckpt.sents_static
            processed_data = data_ckpt["processed_data"]
        else:
            sents_static = Result(
                responses=[],
                knowledges=[],
            )
            data_ckpt = Result(
                stage=["train", "valid", "test"],
                start_index=0,
                processed_data=processed_data,
                sents_static=sents_static,
            )

        error_count = 0
        for stage in data_ckpt.stage:
            data = all_data[stage][data_ckpt.start_index :]
            for i, item in enumerate(tqdm(data, desc=f"预处理{stage}数据集")):
                utterances = []
                dialog_history = []
                for dialog_idx, dialog in enumerate(item["dialog"]):
                    ###############################################
                    # 格式化原数据结构
                    ###############################################
                    response = dialog["text"]
                    if len(dialog_history) < 1:
                        dialog_history.append("__topic__:" + item["chosen_topic"])
                    history = dialog_history.copy()
                    dialog_history.append(response)
                    speaker = (
                        "Wizard" if "Wizard" in dialog["speaker"] else "Apprentice"
                    )
                    is_wizard = True if speaker == "Wizard" else False
                    if is_wizard:
                        if len(list(dialog["checked_sentence"].values())) < 1:
                            topic = list(dialog["checked_passage"].values())[0]
                            passage = [topic]
                            for p in dialog["retrieved_passages"]:
                                if topic in p:
                                    passage = p[topic]
                            knowledge = passage[0]
                        else:
                            knowledge = list(dialog["checked_sentence"].values())[0]
                    else:
                        continue
                                        

                    ###############################################
                    # 封装新的数据结构
                    ###############################################
                    uttr = Result(
                        history=history,
                        knowledge=knowledge,
                        response=response,
                    )
                    ###############################################
                    # 数据预处理
                    ###############################################
                    try:
                        uttr = data_method_caller(process_flow, uttr)
                    except Exception as e:
                        print_error_info(e)
                        error_count += 1
                        continue

                    utterances.append(uttr)

                    sents_static.get("responses").append(uttr['response'])
                    sents_static.get("knowledges").append(uttr['knowledge'])

                processed_data[stage].append(
                    {"dialog_idx": i, "utterances": utterances}
                )

                ###############################################
                # 保存数据集预处理进度
                ###############################################
                if i != 0 and i % 200 == 0:
                    current_stage = data_ckpt.stage[data_ckpt.stage.index(stage):]
                    data_ckpt.merge_or_update(
                        Result(
                            stage=current_stage,
                            start_index=i,
                            processed_data=processed_data,
                            sents_static=sents_static,
                        )
                    )
                    save_as(data_ckpt, data_ckpt_path, data_name="数据集预处理ckpt")

            data_ckpt["start_index"] = 0

        save_as(processed_data, data_path, data_name="预处理数据集")

        os.remove(data_ckpt_path + ".pt")
        print(f"预处理完成，共有{error_count}条数据出错")
        
        nlp_model.close()
        
        return processed_data
