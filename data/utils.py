"""
Author: Deng Yifan 553192215@qq.com
Date: 2022-07-07 08:15:04
LastEditors: Deng Yifan 553192215@qq.com
LastEditTime: 2022-09-19 17:21:56
FilePath: /faith_dial/faith_dial/utils.py
Description: 

Copyright (c) 2022 by Deng Yifan 553192215@qq.com, All Rights Reserved. 
"""
from general_files.utils.common_util import Result
from general_files.utils.data_util import flat, replace_word
from nltk.tokenize import sent_tokenize
import re
import logging

logging.getLogger("utils").setLevel(logging.WARNING)


def caller(methods, result, *args, **kwargs):
    result = Result() if result is None else result
    for method in methods:
        result = globals().get(method)(result, *args, **kwargs)
    return result


def clean_text(uttr, *args, **kwargs):
    if "response" in uttr:
        uttr["response"] = uttr["response"].replace('. . ', '', 1)
        uttr["response"] = uttr["response"][::-
                                            1].replace('. . . ', ' . ', 1)[::-1]
        uttr["response"] = uttr["response"].replace("that . s ", "that is").replace(" . . . ", " ")\
            .replace(" . . ", " , ").replace("U.S.", "America").replace("..", ",").replace(",.", ",")\
            .replace(",,", ",").replace("[sic]", "").replace("OK.", "OK,").replace("Ok.", "OK,")\
            .replace("+", "and").replace("\t", "").replace("?.", ",").replace("!.", ",")\
            .replace("( )", "").replace("\\", "").replace('."', '".')
    if "knowledge" in uttr:
        uttr["knowledge"] = uttr["knowledge"].replace("U.S.", "America").replace("..", ",").replace("[sic]", "")\
            .replace(",,", ",").replace(",.", ",").replace("OK.", "OK,").replace("Ok.", "OK,")\
            .replace("+", "and").replace("\t", "").replace("?.", ",").replace("!.", ",").replace("( )", "")\
            .replace("\\", "").replace('."', '".').replace("''", '"')\
            .replace("Super Smash Bros. Brawl", "Super Smash Bros Brawl").replace('rural."', "rural.")\
            .replace("Super Smash Bros. ", "Super Smash Bros ").replace(
                "mental suffering; mental torment",
                "mental suffering and mental torment",
        ).replace('torment."', "torment.")
    return uttr


def contrast_response_by_baseline(uttr, *args, **kwargs):
    generator = kwargs['generator']
    all_history = []
    for i, h in enumerate(uttr["history"]):
        if i % 2 == 0:
            all_history.append("<user> " + h)
        else:
            all_history.append("<bot> " + h)
    history = flat(all_history[-1:])
    input = "<knowledge> " + uttr['knowledge'] + " " + history
    contrastive_response = generator(input)[0]['seqs']
    contrastive_response = replace_word(contrastive_response)
    uttr.add(contrastive_response=contrastive_response)
    return uttr


def standford_nlp_seg(uttr, *args, **kwargs):
    config = kwargs['config']
    nlp_model = kwargs['nlp_model']
    if "knowledge" in uttr:
        know_segments = trans_sent_to_segs(
            uttr["knowledge"], nlp_model, config)
        uttr.add(know_segments=know_segments)
    segments = trans_sent_to_segs(uttr["response"], nlp_model, config)
    if "contrastive_response" in uttr:
        contrastive_response_segments = trans_sent_to_segs(
            uttr["contrastive_response"], nlp_model, config)
        uttr.add(contrast_response_segments=contrastive_response_segments)
    uttr.add(segments=segments,)
    return uttr


def trans_sent_to_segs(sentence, nlp_model, config):
    sub_ori_sents = sent_tokenize(sentence)
    sub_sents = []
    ###############################################
    # 按照逗号分割
    ###############################################
    for j, s in enumerate(sub_ori_sents):
        if "," in s or "!" in s or "?" in s or ' . ' in s or (". " in s and s[-1] != "."):
            ns = re.split(r"([,!.?])", s)
            for k, subs in enumerate(ns):
                if subs in [",", "!", ".", "?"]:
                    ns.pop(k)
                if k + 1 <= len(ns) - 1:
                    ns[k] = (ns[k] + ns[k + 1]).strip()
                    ns.pop(k + 1)
            sub_sents.extend(ns)
        else:
            sub_sents.append(s)
    ###############################################
    # 二次切分句子
    ###############################################
    try:
        segments = []
        for s in sub_sents:
            if len(s.strip().split(" ")) > 1:
                segments.extend(cut_sent(s.strip(), nlp_model))
            elif s != "":
                segments.append(s)
    except Exception as e:
        segments = sub_sents
        print("错误数据：", sentence)
        with open(config.public_data_path + "/error_sents.txt", encoding="utf-8", mode="a") as file:
            file.write(sentence + "\n")

    return segments


# 注意切分后的子句是按照stanfordnlp词切分后的情况
def cut_sent(sentence, nlp_model):
    result = nlp_model.dependency_parse(sentence)
    d = {}
    # 从i指向j的映射统计
    for i in range(1, len(result)):
        d[result[i][2]] = result[i][1]

    # 先将result内部的根节点统计出来
    # 所有有被指向的节点都是根节点
    roots = []
    for i in range(1, len(result)):
        roots.append(result[i][1])

    # 消除重复的根节点
    roots = list(set(roots))
    roots.sort()
    # 存储分割结果
    segs = []

    for root in roots:
        # print(root)
        # 当前根节点的分割方案
        children_root = []

        # 判断一下这个根节点是否因为是别的节点的孩子而已经被分割到别的组里了
        flag = False
        for seg in segs:
            if root in seg:
                flag = True
                break
        if flag:
            continue
        children_root.append(root)

        # 下面开始遍历根节点root的所有叶子节点，然后判断是否符合所定义的三种条件
        for re in result:
            # re的父节点不是root，因此直接略过不要
            if re[1] == 0:
                continue

            # 将已经分割好的排除出去
            f1 = False
            for seg in segs:
                if re[2] in seg:
                    f1 = True
            if f1:
                continue

            # 到了这里，就说明所代表的节点是当前根节点的孩子节点
            if root > re[2]:
                # 下面开始在当前re节点和当前根节点里面寻找那两种模式
                # 开始判断第一种模式：就是节点re和root之间的所有节点都是root节点的函数
                pattern = False
                for i in range(re[2], root):
                    try:
                        if d[i] != root:
                            pattern = True
                            break
                    except KeyError:
                        continue

                if pattern:  # 说明在re[2]和root之间有不是root孩子节点的节点
                    # 下面开始判断是否是第二种模式
                    flag = True
                    for i in range(re[2], root):
                        try:
                            if d[i] != i + 1:
                                flag = False
                                break
                        except KeyError:
                            continue

                    if flag:  # 此时说明之间的所有 id 都可以加入到当前root的分割里面

                        for ch in range(re[2], root):
                            children_root.append(ch)

                else:  # 说明在re[2]和root之间都是root的孩子节点，将之间的东西全部加入到当前的分割里面
                    for ch in range(re[2], root):
                        children_root.append(ch)
            else:
                # 下面开始root比当前节点小的情况的处理，开始识别两种模式
                # 现在开始判断第一种模式
                pattern = False
                for i in range(root + 1, re[2] + 1):
                    try:
                        if d[i] != root:
                            pattern = True
                            break
                    except KeyError:
                        continue

                if pattern:  # 此时说明两者之间有不是root孩子节点的节点
                    # 下面开始第二种模式的判断
                    flag = True
                    for i in range(re[2], root, -1):
                        try:
                            if d[i] != i - 1:
                                flag = False
                                break
                        except KeyError:
                            continue

                    if flag:  # 此时说明符合那个第二种模式的
                        for ch in range(root, re[2] + 1):
                            children_root.append(ch)

                else:  # 此时两者之间都是root的孩子节点，将他们加入到当前的分割里面
                    for ch in range(root, re[2] + 1):
                        children_root.append(ch)

        children_root = list(set(children_root))
        segs.append(children_root)

    # 下面开始将分割方案里面长度为1或者2的 合并一些
    def merge(segs):

        done = False
        # 还要先判断一下segs内部是不是有长度不大于2的EDU才行
        for i in range(len(segs) - 1):
            if len(segs[i]) <= 2:
                done = True
                break

        while done and len(segs) > 1:
            segs[i + 1].extend(segs[i])
            # 下面将已经被合并的EDU弹出去
            segs.pop(i)

            for i in range(len(segs) - 1):
                if len(segs[i]) <= 2:
                    done = True
                    break
                else:
                    done = False
        # 最后还要再判断一次segs的最后一个是否是长度不大于2的
        if len(segs) > 1 and len(segs[-1]) <= 2:
            segs[-2].extend(segs[-1])
            segs.pop(-1)

        return segs

    segs = merge(segs)

    l1 = [re[2] for re in result]
    l1.sort()
    l2 = []
    for seg in segs:
        l2.extend(seg)
    l2.sort()
    # 一些依存句法没有切分到的句子
    other = list(set(l1) ^ set(l2))
    all_segs = set([item for sublist in segs for item in sublist])
    while len(set(l1) - all_segs) > 0:
        for oth in other:
            for i in range(len(segs)):
                # print(segs[i])
                if oth not in all_segs and (
                    (oth - 1 in segs[i]) or (oth + 1) in segs[i]
                ):
                    segs[i].append(oth)
                    all_segs.add(oth)
                    break

    sort_segs = []
    for i in range(len(segs)):
        segs[i].sort()
        sort_segs.append(segs[i])

    sent_token_list = nlp_model.word_tokenize(sentence)
    seg_sentence = []
    for seg in segs:
        seg_token = []
        for token_idx in seg:
            token = sent_token_list[token_idx - 1]
            seg_token.append(token)
        seg_str = " ".join(seg_token)
        seg_sentence.append(seg_str)

    ###############################################
    # 修正分词
    ###############################################
    ori_segs = sentence.split(" ")
    re_segments = []
    last_index = 0
    for index_seg, o_s in enumerate(seg_sentence):
        pre_os = "".join(seg_sentence[: index_seg + 1])
        for index, seg in enumerate(ori_segs):
            if seg == "":
                continue
            pre_seg = "".join(ori_segs[: index + 1])
            # https://blog.csdn.net/m0_51981035/article/details/122892547
            if (
                pre_os.strip()
                .replace(" ", "")
                .replace("\u0020", "")
                .replace("\u3000", "")
                .replace("\u00A0", "")
                == pre_seg.strip()
                .replace(" ", "")
                .replace("\u0020", "")
                .replace("\u3000", "")
                .replace("\u00A0", "")
                and len(ori_segs[last_index: index + 1]) >= 1
            ):
                re_segments.append(" ".join(ori_segs[last_index: index + 1]))
                last_index = index + 1
    if (
        sentence.replace(" ", "").strip()
        != "".join(re_segments).replace(" ", "").strip()
    ):
        print()
    assert (
        sentence.replace(" ", "").strip()
        == "".join(re_segments).replace(" ", "").strip()
    )
    return seg_sentence
