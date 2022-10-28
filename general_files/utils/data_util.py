from nltk.corpus import stopwords
import rich.tree
from sklearn.model_selection import train_test_split
from general_files.utils.others.data_processor.processor import get_data_processor
import pandas as pd
from datasets import Dataset
import torch
from rich.console import Console
import jieba.analyse as analyse
import os
import json
import pickle
import matplotlib.pyplot as plt
import jsonlines
import pprint
from pytorch_lightning.utilities import rank_zero_only
import spacy
import random
import itertools
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# 英文表达常见缩写
CONJUNCTIONS_WORDS_MAP = {
    "isn't": "is not",
    "Isn't": "Is not",
    "wasn't": "was not",
    "Wasn't": "Was not",
    "aren't": "are not",
    "Aren't": "Are not",
    "weren't": "were not",
    "haven't": "have not",
    "Haven't": "Have not",
    "hasn't": "has not",
    "Hasn't": "Has not",
    "hadn't": "had not",
    "won't": "will not",
    "Won't": "Will not",
    "wouldn't": "would not",
    "don't": "do not",
    "Don't": "Do not",
    "doesn't": "does not",
    "Doesn't": "Does not",
    "didn't": "did not",
    "Didn't": "Did not",
    "can't": "can not",
    "Can't": "Can not",
    "couldn't": "could not",
    "Couldn't": "Could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not",
    "i'm": "i am",
    "I'm": "I am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "It's": "It is",
    "we're": "we are",
    "We're": "We are",
    "they're": "they are",
    "They're": "They are",
    "i've": "i have",
    "I've": "I have",
    "you've": "you have",
    "You've": "You have",
    "we've": "we have",
    "We've": "We have",
    "they've": "they have",
    "They've": "They have",
    "i'd": "i would",
    "I'd": "I would",
    "you'd": "you would",
    "You'd": "You would",
    "he'd": "he would",
    "He'd": "He would",
    "she'd": "she would",
    "She'd": "She would",
    "we'd": "we would",
    "We'd": "We would",
    "they'd": "they would",
    "i'll": "i will",
    "I'll": "I will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "We'll": "We will",
    "they'll": "they will",
    "They'll": "They will",
    "isnt": "is not",
    "Isnt": "Is not",
    "wasnt": "was not",
    "arent": "are not",
    "werent": "were not",
    "havent": "have not",
    "hasnt": "has not",
    "hadnt": "had not",
    "wont": "will not",
    "wouldnt": "would not",
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "cant": "can not",
    "couldnt": "could not",
    "shouldnt": "should not",
    "mightnt": "might not",
    "mustnt": "must not",
    "There's": "There is",
    "there's": "there is",
    "There're": "There are",
    "there're": "there are",
    "What's": "What is",
    "what's": "what is",
    "What're": "What are",
    "what're": "what are",
    "Who's": "Who is",
    "who's": "who is",
    "Who're": "Who are",
    "who're": "who are",
    "Where's": "Where is",
    "where's": "where is",
    "Where're": "Where are",
    "where're": "where are",
    "How's": "How is",
    "how's": "how is",
    "How're": "How are",
    "how're": "how are",
    "I'ma": "I am going to",
    "I'lla": "I will",
    "we're": "we are",
    "We're": "We are",
    "they're": "they are",
    "They're": "They are",
    "When's": "When is",
    "when's": "when is",
    "Let's": "Let us",
    "let's": "let us",
    "Who've": "Who have",
    "who've": "who have",
    "My name's": "My name is",
    "my name's": "my name is",
    "Your name's": "Your name is",
    "your name's": "your name is",
    "His name's": "His name is",
    "his name's": "his name is",
    "Her name's": "Her name is",
    "her name's": "her name is",
    "Its name's": "Its name is",
    "its name's": "its name is",
    "Our name's": "Our name is",
    "our name's": "our name is",
    "Their name's": "Their name is",
    "their name's": "their name is",
    "What're": "What are",
    "what're": "what are",
    "What've": "What have",
    "what've": "what have",
    "What'll": "What will",
    "what'll": "what will",
    "What'd": "What did",
    "what'd": "what did",

}


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


@rank_zero_only
def pp(input_obj):
    pprint.pprint(input_obj)


def get_tokenized_data(config, tokenizer, only_test=False):
    """
    获取tokenized数据集的入口函数
    :param config: dict
    :param tokenizer: tokenizer
    :param model: model
    :return: train_data_tokenized, valid_data_tokenized, test_data_tokenized
    """
    data_processor = get_data_processor(config, tokenizer, only_test)
    return data_processor.get_dataset()


@rank_zero_only
def print_dataset_overview(
    train_data_tokenized, valid_data_tokenized, test_data_tokenized
):
    style = "bold cyan"
    tree = rich.tree.Tree(
        "Dataset Overview", style=style, highlight=True, guide_style=style
    )
    if train_data_tokenized is not None:
        branch = tree.add("Train Dataset", style=style, guide_style=style)
        branch.add(
            "Size: " + str(len(train_data_tokenized)), style=style, guide_style=style
        )
        branch.add(
            "Columns: " + "、".join(train_data_tokenized.column_names),
            style=style,
            guide_style=style,
        )

    if valid_data_tokenized is not None:
        branch = tree.add("Valid Dataset", style=style, guide_style=style)
        branch.add(
            "Size: " + str(len(valid_data_tokenized)), style=style, guide_style=style
        )
        branch.add(
            "Columns: " + "、".join(valid_data_tokenized.column_names),
            style=style,
            guide_style=style,
        )

    if test_data_tokenized is not None:
        branch = tree.add("Test Dataset", style=style, guide_style=style)
        branch.add(
            "Size: " + str(len(test_data_tokenized)), style=style, guide_style=style
        )
        branch.add(
            "Columns: " + "、".join(test_data_tokenized.column_names),
            style=style,
            guide_style=style,
        )
    rich.print(tree)


def read_by(path, data_name=""):
    """读取文件，根据文件后缀名自动选择读取方法
        目前支持保存类型有：‘pkl’、‘txt’、‘pt’、‘json’, 'jsonl'

    Args:
        data_name: str, 打印提示时需要，便于控制台查看保存的文件是什么文件, 默认为空

    Returns:
        data：Object
    """
    if not os.path.exists(path):
        log.info(f"文件路径似乎并不存在....")
        raise FileNotFoundError(path)
    log.info(f"正在加载文件 {data_name} from {path}")
    if ".pkl" in path:
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif ".json" in path:
        with open(path, "r") as f:
            # 读取json数据
            data = json.load(f)
    elif ".jsonl" in path:
        data = []
        with open(path, "rb") as f:
            for item in jsonlines.Reader(f):  # 每一行读取后都是一个json，可以按照key去取对应的值
                data.append(item)
    elif ".pt" in path:
        data = torch.load(path)
    elif ".txt" in path:
        data = []
        for item in open(path):
            item = item.replace("\n", "").strip()
            if item == "":
                continue
            data.append(item)
    log.info(f"成功加载 {data_name}!")
    return data


def save_as(data, save_path, file_format="pt", data_name="", protocol=4):
    """将参数中的文件对象保存为指定格式格式文件
        目前支持保存类型有：‘pkl’、‘txt’、‘pt’、‘json’, 'jsonl'
        默认为‘pt’

    Args:
        data: obj, 要保存的文件对象
        save_path: str, 文件的保存路径，应当包含文件名和后缀名
        data_name: str, 打印提示时需要，便于控制台查看保存的文件是什么文件, 默认为空
        protocol: int, 当文件特别大的时候，需要将此参数调到4以上, 默认为4
        file_format: str, 要保存的文件类型，支持‘pkl’、‘txt’、‘pt’、‘json’、‘jsonl’

    Returns:
        None
    """
    parent_path = "/".join(save_path.split("/")[:-1])
    if not os.path.exists(parent_path):
        log.info(f"保存路径的父文件夹（{parent_path}）不存在，将自动创建....")
        os.makedirs(parent_path)
    save_path = save_path + f".{file_format}"
    log.info(f"正在保存文件 {data_name} 到 {save_path}")
    if file_format == "pkl":
        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol=protocol)
    elif file_format == "txt":
        if not isinstance(data, list):
            data = [data]
        with open(save_path, "w") as f:
            for line in data:
                f.write(str(line) + "\n")
    elif file_format == "json":
        with open(save_path, "w") as f:
            json.dump(data, f)
    elif file_format == "jsonl":
        with jsonlines.open(save_path, mode="w") as writer:
            writer.write_all(data)
    elif file_format == "pt":
        torch.save(data, save_path)
    else:
        raise Exception(f"请添加针对{file_format}类型文件的保存方法！")
    log.info(f"保存 {data_name} 成功!")
    return None



def read_txt_by_line(file_path, data_name=None):
    """
    读取txt文件
    Args:
        file_path: str, 读取的目标文件保存路径
        data_name: str, 读取的文件名，打印提示时需要，便于控制台查看读取的文件是什么文件, 默认为空

    Returns:
        data: obj, 读取的文件对象

    """
    all_lines = set()
    for item in open(file_path):
        item = item.replace("\n", "").strip()
        if item == "":
            continue
        all_lines.add(item)
    if data_name is not None:
        log.info(f"Load {data_name} from {file_path}")
    return all_lines


def split_data(ori_data, random_seed, valid_size=0.1, test_size=0.1):
    train_data, test_data = train_test_split(
        ori_data, test_size=test_size, random_state=random_seed
    )
    train_data, valid_data = train_test_split(
        train_data, test_size=valid_size, random_state=random_seed
    )
    return train_data, valid_data, test_data


def concatenate_multi_datasets(dataset1, dataset2):
    return Dataset.from_pandas(
        pd.concat([pd.DataFrame(dataset1), pd.DataFrame(dataset2)], axis=1)
    )


def max_lens(X):
    """
    max_lens
    """
    if not isinstance(X[0], list):
        return [len(X)]
    elif not isinstance(X[0][0], list):
        return [len(X), max(len(x) for x in X)]
    elif not isinstance(X[0][0][0], list):
        return [len(X), max(len(x) for x in X), max(len(x) for xs in X for x in xs)]
    else:
        raise ValueError("Data list whose dim is greater than 3 is not supported!")


def list2tensor(X, pad_token_id):
    """
    list2tensor
    """
    size = max_lens(X)
    if len(size) == 1:
        tensor = torch.tensor(X)
        return tensor

    tensor = torch.zeros(size, dtype=torch.long)
    lengths = torch.zeros(size[:-1], dtype=torch.long)
    if len(size) == 2:
        for i, x in enumerate(X):
            x = torch.tensor(x)
            l = x.ne(pad_token_id).int().sum().item()
            tensor[i] = x
            lengths[i] = l
    else:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                x = torch.tensor(x)
                l = x.ne(pad_token_id).int().sum().item()
                tensor[i, j] = x
                lengths[i, j] = l

    return tensor, lengths


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.lt(lengths.unsqueeze(-1))
    # mask = mask.repeat(*lengths.size(), 1).lt(lengths.unsqueeze(-1))
    return mask


def dict_list_to_tensor(ori_list):
    keys = ori_list[0].keys()
    res = dict()
    for f in ori_list:
        for key in keys:
            if key not in res:
                res[key] = []
            item = [f[key]] if isinstance(f[key], int) else f[key]
            res[key].append(item)

    for key in keys:
        try:
            res[key] = torch.LongTensor(res[key])
        except ValueError:
            continue
    return res


def generate_square_subsequent_mask(seq_tensor):
    """
    生成decoder的上三角矩阵
    :param seq_tensor: batch_size, seq_len
    :return:
    """
    seq_len = seq_tensor.sahpe[1]
    mask = (
        torch.triu(torch.ones((seq_len, seq_len), device=seq_tensor.device)) == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(pad_token_id=0, src=None, tgt=None):
    """
    生成padding矩阵和attention矩阵
    :param pad_token_id: 默认为0
    :param src: batch_size, seq_len
    :param tgt: batch_size, seq_len
    :return:
    """
    src_mask = tgt_mask = src_padding_mask = tgt_padding_mask = None
    if src is not None:
        src_seq_len = src.shape[1]
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(
            torch.bool
        )
        src_padding_mask = (src == pad_token_id).transpose(0, 1)
    if tgt is not None:
        tgt_mask = generate_square_subsequent_mask(tgt)
        tgt_padding_mask = (tgt == pad_token_id).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def strip_str(input):
    """
    适用于中文去空格
    """
    return str(input).replace(" ", "").replace("\n", "").replace("\t", "").strip()


@rank_zero_only
def print_sample_data(
    tokenizer,
    data_tokenized_list,
    data_tokenized_name_list,
    config=None,
    experiment=None,
):
    console = Console(color_system="256", style="cyan")
    show_columns = ["input_ids", "labels"]
    log_data = {}
    for i, data_tokenized in enumerate(data_tokenized_list):
        if data_tokenized is None:
            continue
        console.print(
            f"[bold red]{data_tokenized_name_list[i]}数据格式展示", justify="center"
        )
        log_sub_data = {}
        index = random.randint(0, len(data_tokenized))
        for c in show_columns:
            if c in data_tokenized.column_names:
                input_to_print = data_tokenized[index][c]
                input_to_print = (
                    input_to_print
                    if (config.data_mode == "classification" and c == "labels")
                    else tokenizer.decode(
                        input_to_print, ignore_tokens=[tokenizer.pad_token]
                    )
                )
                console.print(f"[bold orange1]{c}格式 --> [bold cyan]{input_to_print}")
                if experiment and config.logger == "comet":
                    log_sub_data[f"{c}格式"] = input_to_print
        log_data[data_tokenized_name_list[i]] = log_sub_data
    if experiment and config.logger == "comet":
        # 保存到comet
        experiment.log_asset_data(log_data, name=f"data input format.json")


def extract_zh_keywords_by_tf_idf(text, top_k=5):
    keywords = analyse.extract_tags(text, topK=top_k, withWeight=False)
    return keywords


def extract_zh_keywords_by_textrank(text, top_k=5):
    # 使用jieba的实现方式
    keywords = analyse.textrank(text, topK=top_k, withWeight=False)
    return keywords


def extract_en_keywords_by_rare_nltk(text):
    ## Rake
    from rake_nltk import Rake

    rake_nltk_var = Rake(max_length=2)
    rake_nltk_var.extract_keywords_from_text(text)
    keywords = rake_nltk_var.get_ranked_phrases()
    return keywords


def extract_en_keywords_by_keybert(
    full_text, keyphrase_ngram_range=(1, 3), highlight=False, top_n=10
):
    from keybert import KeyBERT

    kw_model = KeyBERT(model="all-mpnet-base-v2")
    keywords = kw_model.extract_keywords(
        full_text,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words="english",
        highlight=highlight,
        top_n=top_n,
    )

    keywords_list = list(dict(keywords).keys())
    return keywords_list


def extract_en_keywords_by_yake(
    text, max_ngram_size=2, deduplication_threshold=0.9, top_k=5, language="en"
):
    """
    yake使用文本统计特征方法从文章中选择最重要的关键字。
    :param text:
    :param max_ngram_size: 最大关键词语长度
    :param deduplication_threshold:  设置在关键词中是否可以重复单词
    :return:
    """
    ## yake
    import yake

    custom_kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_threshold,
        top=top_k,
        features=None,
    )
    keywords = custom_kw_extractor.extract_keywords(text)
    return keywords


def extract_en_keywords_by_sklearn_tfidf(corpus, top_n=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    # 首先，構建語料庫corpus
    transfer = TfidfVectorizer(stop_words=stopwords.words("english"))
    corpus = transfer.fit_transform(corpus)

    words = transfer.get_feature_names()  # 所有文本的关键字
    weight = corpus.toarray()
    corpus_kw = []
    for w in weight:
        # 排序
        kw = []
        loc = np.argsort(-w)
        for i in range(top_n):
            kw.append({words[loc[i]]: w[loc[i]]})
        corpus_kw.append(kw)
    return corpus_kw


def extract_en_keywords_by_rare_nltk(text):
    """
    spaCy是一个集成化的工业级自然语言处理工具，主要功能包括分词、词性标注、词干化、命名实体识别、名词短语提取等等。
    doc.ents的输出可能是1-gram, 2-gram, 3-gram等，无法人工调控。
    :param text:
    :return:
    """
    ## spaCy
    spacy_nlp = spacy.load("en_core_web_sm")
    doc = spacy_nlp(text)
    return doc.ents


def number_of_certain_probability(sequence, probability):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(sequence, probability):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def flat(input_list, connect_str=" "):
    """
    展平列表
    """
    if not connect_str:
        return list(
            itertools.chain(*[i if isinstance(i, list) else [i] for i in input_list])
        )
    else:
        return connect_str.join(
            list(
            itertools.chain(*[i if isinstance(i, list) else [i] for i in input_list])
        ))


def rfind_list(input_list, target):
    """
    实现列表的rfind方法，即返回符合条件的最后一个
    """
    return len(input_list) - input_list[-1::-1].index(target) - 1

# 绘制箱线图
def xxt(data):
    plt.boxplot(
        x=data,  # 指定绘图数据
        patch_artist=True,  # 要求用自定义颜色填充盒形图，默认白色填充
        showmeans=True,  # 以点的形式显示均值
        flierprops={
            "marker": "o",
            "markerfacecolor": "red",
            "color": "black",
        },  # 设置异常值属性，点的形状、填充色和边框色
        meanprops={"marker": "D", "markerfacecolor": "indianred"},  # 设置均值点的属性，点的形状、填充色
    )
    plt.show()


class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, collate_fn, config):
        super().__init__()
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.collate_fn = collate_fn
        self.config = config

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            pin_memory=self.config.dataloader_pin_memory,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            pin_memory=self.config.dataloader_pin_memory,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=self.collate_fn,
        )


def replace_word(texts):
    if isinstance(texts, str):
        for k, v in CONJUNCTIONS_WORDS_MAP.items():
            texts = texts.replace(k, v)
    elif isinstance(texts, list):
        for i, text in enumerate(texts):
            for k, v in CONJUNCTIONS_WORDS_MAP.items():
                texts[i] = text.replace(k, v)
    else:
        raise ValueError("texts must be str or list")
    return texts


if __name__ == "__main__":
    keyword = extract_zh_keywords_by_tf_idf(
        "这个 我 专门 了解 过 的 。 先 把 真 睫毛 涂 一 层 睫毛 膏 卷 起来 定型 ， 然后 把 假 睫毛 剪 下来 揉 软 ， 再 把 睫毛 贴 上 胶水 贴 到 眼皮 上 。",
        top_k=3,
    )
    print()
