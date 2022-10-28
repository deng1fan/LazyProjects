
import os
import sys
import torch.nn as nn
from transformers import BertTokenizer, AutoTokenizer
from general_files.utils import common_util as utils
from general_files.utils.common_util import Result
from general_files.utils.others.data_processor.processor import get_data_processor


log = utils.get_logger(__name__)


class Tokenizer(nn.Module):
    def __init__(self, config):
        super(Tokenizer, self).__init__()
        self.config = config
        self.unk_token_num = 50
        if config.tokenize_method == "auto":
            self.tokenizer = self.get_tokenizer_from_pretrained(config.pretrain_model.split(':')[-1])
        else:
            self.tokenizer = None
        self.init_dict()

    def init_dict(self):
        if self.tokenizer is not None:
            if self.tokenizer.cls_token and not self.tokenizer.bos_token:
                self.tokenizer.bos_token = self.tokenizer.cls_token
            if self.tokenizer.sep_token and not self.tokenizer.eos_token:
                self.tokenizer.eos_token = self.tokenizer.sep_token
            additional_flags = ['<mask>']
            if not self.tokenizer.pad_token:
                self.pad_token = '<pad>'
                additional_flags.append('<pad>')
                self.tokenizer.pad_token = '<pad>'
            else:
                self.pad_token = self.tokenizer.pad_token

            if not self.tokenizer.unk_token:
                self.unk_token = '<unk>'
                additional_flags.append('<unk>')
                self.tokenizer.unk_token = '<unk>'
            else:
                self.unk_token = self.tokenizer.unk_token

            if not self.tokenizer.bos_token:
                self.bos_token = '<bos>'
                additional_flags.append('<bos>')
                self.tokenizer.bos_token = '<bos>'
            else:
                self.bos_token = self.tokenizer.bos_token
            self.start_token = self.tokenizer.bos_token

            if not self.tokenizer.eos_token:
                self.eos_token = '<eos>'
                additional_flags.append('<eos>')
            else:
                self.eos_token = self.tokenizer.eos_token
            self.end_token = self.tokenizer.eos_token

            if not self.tokenizer.sep_token:
                self.sep_token = '<sep>'
                additional_flags.append('<sep>')
                self.tokenizer.sep_token = '<sep>'
            else:
                self.sep_token = self.tokenizer.sep_token

            self.tokenizer.add_special_tokens(
                {'additional_special_tokens': additional_flags})
            self.vocab_size = len(self.tokenizer)

            self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.convert_tokens_to_ids(
                '<eos>')
            self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.convert_tokens_to_ids(
                '<pad>')
            self.unk_token_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else self.tokenizer.convert_tokens_to_ids(
                '<unk>')
            self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.convert_tokens_to_ids(
                '<bos>')
            self.sep_token_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.convert_tokens_to_ids('<sep>')
            self.end_token_id = self.eos_token_id
            self.start_token_id = self.bos_token_id
            return None

        dict_path = self.config.custom_dict_path
        if not os.path.exists(f'{dict_path}vocab.txt'):
            # Init data
            log.info("缺少数据字典！需要生成数据字典！")
            # data_processor = get_data_processor(self.config)
            # all_sents = data_processor.get_all_sents()
            # tokenized_sents = self.encode(all_sents)
            # save_column_to_train(tokenized_sents, self.config.general_files_path)
            # print("请先使用glove文件中的Bash脚本训练，训练完成后使用以下命令将词表和词向量移到项目的数据集目录中")
            # print(f"mv {self.config.general_files_path}/utils/glove/vectors.txt {dict_path}vectors.txt")
            # print(f"mv {self.config.general_files_path}/utils/glove/vocab.txt {dict_path}vocab.txt")
            # print('移动完成后，请重新运行程序以加载词表')
            sys.exit(0)
        else:
            words = []
            with open(f'{dict_path}vocab.txt', 'r') as f:
                for line in f:
                    word = line.split(' ')[0].replace('\n', '').strip()
                    if word != '':
                        words.append(word)
            if '<mask>' not in words:
                words.insert(0, '<mask>')
                words.insert(0, '<sep>')
                words.insert(0, '<bos>')
                words.insert(0, '<eos>')
                words.insert(0, '<unk>')
                words.insert(0, '<pad>')
                with open(f'{dict_path}vocab.txt', 'w') as f:
                    f.truncate(0)
                    for token in words:
                        f.write(token + ' 9999999' + '\n')
            self.word_dict = {w: i for i, w in enumerate(words)}

        self.vocab_size = len(self.word_dict)
        self.pad_token_id = self.word_dict['<pad>']
        self.unk_token_id = self.word_dict['<unk>']
        self.eos_token_id = self.word_dict['<eos>']
        self.bos_token_id = self.word_dict['<bos>']
        self.start_token_id = self.bos_token_id
        self.end_token_id = self.eos_token_id
        self.sep_token_id = self.word_dict['<sep>']
        self.start_token = '<bos>'
        self.end_token = '<eos>'
        self.eos_token = '<eos>'
        self.bos_token = '<bos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.sep_token = '<sep>'
        self.decode_dict = {k: v for v, k in self.word_dict.items()}

    def set_special_token(self, bos_token=None, eos_token=None, sep_token=None, pad_token=None):
        if bos_token:
            self.bos_token = bos_token
            if bos_token not in self.tokenizer.vocab:
                self.tokenizer.add_special_tokens(
                    {'additional_special_tokens': [bos_token]})
            self.bos_token_id = self.tokenizer.convert_tokens_to_ids(bos_token)
            self.start_token = self.bos_token
            self.start_token_id = self.bos_token_id
            if self.tokenizer:
                self.tokenizer.bos_token = bos_token
        if eos_token:
            self.eos_token = eos_token
            if eos_token not in self.tokenizer.vocab:
                self.tokenizer.add_special_tokens(
                    {'additional_special_tokens': [eos_token]})
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids(eos_token)
            self.end_token = self.eos_token
            self.end_token_id = self.eos_token_id
            if self.tokenizer:
                self.tokenizer.eos_token = eos_token
        if sep_token:
            self.sep_token = sep_token
            if sep_token not in self.tokenizer.vocab:
                self.tokenizer.add_special_tokens(
                    {'additional_special_tokens': [sep_token]})
            self.sep_token_id = self.tokenizer.convert_tokens_to_ids(sep_token)
            if self.tokenizer:
                self.tokenizer.sep_token = sep_token
        if pad_token:
            self.pad_token = pad_token
            if pad_token not in self.tokenizer.vocab:
                self.tokenizer.add_special_tokens(
                    {'additional_special_tokens': [pad_token]})
            self.pad_token_id = self.tokenizer.convert_tokens_to_ids(pad_token)
            if self.tokenizer:
                self.tokenizer.pad_token = pad_token
        self.vocab_size = len(self.tokenizer)

    def add_special_tokens(self, special_tokens):
        if self.tokenizer is None:
            additional_special_tokens = special_tokens
            insert_tokens = []
            for st in additional_special_tokens:
                if st not in self.word_dict:
                    self.word_dict[st] = len(self.word_dict)
                    insert_tokens.append(st)
            if len(insert_tokens) > 0:
                self.vocab_size = len(self.word_dict)
                self.decode_dict = {k: v for v, k in self.word_dict.items()}
                # 保存到txt
                with open(f'{self.config.custom_dict_path}vocab.txt', 'a') as f:
                    for token in insert_tokens:
                        f.write(token + ' 9999999' + '\n')
        else:
            special_tokens = list(set(special_tokens + self.tokenizer.all_special_tokens)) 
            self.tokenizer.add_special_tokens(
                {'additional_special_tokens': special_tokens})
            self.vocab_size = len(self.tokenizer)

    def forward(self, inputs,
                padding='do_not_pad',
                max_length=-1,
                only_input_ids=False,
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=False,
                *args, **kwargs):
        if self.tokenizer:
            if max_length < 0:
                max_length = None
            if isinstance(inputs, list):
                results = ()
                for inp in inputs:
                    if isinstance(inp[0], list):
                        tokenized_inputs = [self.tokenizer(i,
                                                          padding=padding,
                                                          max_length=max_length,
                                                          truncation=truncation,
                                                          add_special_tokens=add_special_tokens,
                                                          return_offsets_mapping=return_offsets_mapping,
                                                          *args, **kwargs).data if i != [] else [] for i in inp]
                        if only_input_ids:
                            tokenized_inputs = [ti['input_ids'] if ti != [] else [] for ti in tokenized_inputs]
                    else:
                        tokenized_inputs = self.tokenizer(inp,
                                                          padding=padding,
                                                          max_length=max_length,
                                                          truncation=truncation,
                                                          add_special_tokens=add_special_tokens,
                                                          return_offsets_mapping=return_offsets_mapping,
                                                          *args, **kwargs).data
                        if only_input_ids:
                            tokenized_inputs = tokenized_inputs['input_ids']
                    results += (tokenized_inputs,)
                return results

            elif isinstance(inputs, dict):
                results = Result()
                for key in inputs.keys():
                    if isinstance(inputs[key][0], list):
                        tokenized_inputs = [self.tokenizer(inp,
                                                          padding=padding,
                                                          max_length=max_length,
                                                          truncation=truncation,
                                                          add_special_tokens=add_special_tokens,
                                                          return_offsets_mapping=return_offsets_mapping,
                                                          *args, **kwargs).data if inp != [] else [] for inp in inputs[key]]
                        if only_input_ids:
                            tokenized_inputs = [ti['input_ids'] if ti != [] else [] for ti in tokenized_inputs]
                    else:
                        tokenized_inputs = self.tokenizer(inputs[key],
                                                          padding=padding,
                                                          max_length=max_length,
                                                          truncation=truncation,
                                                          add_special_tokens=add_special_tokens,
                                                          return_offsets_mapping=return_offsets_mapping,
                                                          *args, **kwargs).data
                        if only_input_ids:
                            tokenized_inputs = tokenized_inputs['input_ids']
                    results[key] = tokenized_inputs
                return results
            else:
                raise Exception(
                    f"Tokenizer.forward(): 不支持的输入类型！期望获取dict或list类型，但是获取的是{str(type(inputs))}")

    def pad(self, inputs,
            pad_token=None,
            max_length=-1,
            truncation=False,
            *args, **kwargs):
        if not pad_token:
            pad_token = self.pad_token
        pad_id = self.convert_tokens_to_ids(pad_token)
        if isinstance(inputs, list):
            results = []
            for input_to_pad in inputs:
                if max_length < 0:
                    max_length = max([len(inp) for inp in input_to_pad])
                result = []
                for item in input_to_pad:
                    pad_len = max_length - len(item)
                    if pad_len > 0:
                        pad_item = item + [pad_id] * pad_len
                    elif truncation:
                        pad_item = item[:max_length]
                    else:
                        pad_item = item
                    result.append(pad_item)
                results.append(result)
            return results

        elif isinstance(inputs, dict):
            results = Result()
            for key in inputs.keys():
                if max_length < 0:
                    max_length = max([len(inp) for inp in inputs[key]])
                result = []
                for item in inputs[key]:
                    pad_len = max_length - len(item)
                    if pad_len > 0:
                        pad_item = item + [pad_id] * pad_len
                    elif truncation:
                        pad_item = item[:max_length]
                    else:
                        pad_item = item
                    result.append(pad_item)
                results[key] = result
            return results

        else:
            raise Exception(
                f"Tokenizer.pad(): 不支持的输入类型！期望获取dict或list类型，但是获取的是{str(type(inputs))}")

    def decode(self, sent_ids, skip_special_tokens=False, ignore_tokens=None, *args, **kwargs):
        if self.tokenizer:
            decode_sents = self.tokenizer.decode(sent_ids, skip_special_tokens=skip_special_tokens, *args, **kwargs)
            ###############################################
            # 对忽略字符进行替代
            ###############################################
            if ignore_tokens:
                for i, w in enumerate(ignore_tokens):
                    decode_sents = decode_sents.replace(w, '')
        else:
            sent = []
            stop_id = self.word_dict[self.end_token]
            for w in sent_ids:
                if skip_special_tokens:
                    if w == stop_id:
                        break
                    if w == self.pad_token_id:
                        continue
                word = self.decode_dict[w]
                sent.append(word)
            decode_sents = ' '.join(sent)
            ###############################################
            # 对忽略字符进行替代
            ###############################################
            if ignore_tokens:
                for i, w in enumerate(ignore_tokens):
                    decode_sents = decode_sents.replace(w, '')
        return decode_sents

    def convert_tokens_to_ids(self, token, *args, **kwargs):
        if self.tokenizer is not None:
            return self.tokenizer.convert_tokens_to_ids(token, *args, **kwargs)
        token_id = self.unk_token_id if token not in self.word_dict else self.word_dict[token]
        return token_id

    def convert_ids_to_tokens(self, token_id, *args, **kwargs):
        if self.tokenizer is not None:
            return self.tokenizer.convert_ids_to_tokens(token_id, *args, **kwargs)
        token = self.decode_dict[self.unk_token_id] if token_id not in self.decode_dict else self.decode_dict[token_id]
        return token

    def update_dict(self):
        self.vocab_size = len(self.word_dict)
        self.decode_dict = {k: v for v, k in self.word_dict.items()}

    def is_oov(self, word):
        if word == '' or word == ' ':
            return False
        if self.config.tokenize_method == "auto":
            if self.tokenizer.convert_tokens_to_ids(word) == self.tokenizer.unk_token_id:
                return True
        else:
            if word not in self.word_dict:
                return True
        return False

    def __len__(self):
        return self.vocab_size

    def get_tokenizer_from_pretrained(self, pretrain_model):
        if pretrain_model in ['fnlp/bart-base-chinese', 'fnlp/bart-large-chinese', 'fnlp/cpt-large',
                              'fnlp/cpt-base']:
            # 一些特殊情况处理
            tokenizer = BertTokenizer.from_pretrained(pretrain_model, cache_dir=self.config.cache_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrain_model, cache_dir=self.config.cache_dir)
        return tokenizer
