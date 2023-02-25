from tqdm import tqdm
import general_files.utils.common_util as utils
from general_files.utils.common_util import Result
from general_files.utils.data_util import flat
import spacy
from data.wizard_of_wikipedia.basic_preprocess import Processor as BasicProcessor

log = utils.get_logger(__name__)
class Processor(BasicProcessor):
    def __init__(self, config, tokenizer, only_test):
        super(Processor, self).__init__(config, tokenizer, only_test)
        self.spacy_model = spacy.load("en_core_web_sm")
        # self.tokenizer.set_special_token(eos_token="<eos>", bos_token="<bos>")

    def get_rows(self, all_rows, stage):
        rows = Result()
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        bot_token = "<bot>"
        knowledge_token = "<knowledge>"

        for dialog in tqdm(all_rows, desc=f"格式化 {stage} 输入输出"):
            for uttr in dialog["utterances"]:
                ###############################################
                # 基础数据处理
                ###############################################
                all_history = []
                if "__topic__:" in uttr["history"][0]:
                    all_history.append("<user> " + uttr["history"].pop(0))
                for i, h in enumerate(uttr["history"][:: -1]):
                    if i % 2 == 0:
                        all_history.append("<user> " + h)
                    else:
                        all_history.append("<bot> " + h)
                all_history.append(all_history.pop(0))
                history = flat(all_history[-self.config.history_len:][:: -1])

                knowledge = flat([knowledge_token, uttr["knowledge"]])
                response = uttr["response"]

                ###############################################
                # 构建模型输入输出格式
                ###############################################
                text_map = {
                    "k": knowledge,
                    "h": history,
                    "r": response,
                }
                input = flat([text_map[p] for p in self.config.input_shape.split('-')])
                input = flat([bos_token, input, eos_token])

                target = flat([text_map[p] for p in self.config.target_shape.split('-')])
                target = flat([target, eos_token])

                decoder_input = flat([bot_token, target])

                row = Result(
                    source=input,
                    target=target,
                    # >>> other_features <<<
                    decoder_input=decoder_input,
                    response=response,
                    knowledge=uttr["knowledge"],
                    history=history,
                )
                rows.append_values(row)
        return rows

    def tokenize_data(self, batch, stage=None):
        result = Result()
        # 合并同类编码方式
        result.merge_or_update(
            self.tokenizer(
                # key 对应编码之后的字段名，value 对应原始数据中的字段名
                {
                    "input_ids": batch["source"],
                    "decoder_input_ids": batch["decoder_input"],
                    "labels": batch["target"],
                    "decoder_response": batch["response"],
                    "decoder_knowledge": batch["knowledge"],
                    "decoder_history": batch["history"],
                },
                padding="max_length",
                max_length=self.config.encoder_max_length,
                truncation=True,
                only_input_ids=True,
                add_special_tokens=False,
            )
        )
        # >>> other_features <<<
        # 📢  可以按照自己需求定义更加自由的编码方式
        # 📢  例如：不进行 pad，只编码
        # result.merge_or_update(
        #     self.tokenizer(
        #         {
        #             "decoder_other_features": batch["other_features"],
        #         },
        #         truncation=True,
        #         only_input_ids=True,
        #         add_special_tokens=False,
        #     )
        # )
        # 📢  例如：只进行 pad，不编码
        # result.merge_or_update(
        #     self.tokenizer.pad(
        #         {
        #             "decoder_other_features": batch["other_features"],
        #         },
        #         max_length=self.config.encoder_max_length,
        #         truncation=True,
        #     )
        # )
        # 📢  例如：直接使用原始数据
        # result.add(decoder_other_features=batch["other_features"])
        return result
