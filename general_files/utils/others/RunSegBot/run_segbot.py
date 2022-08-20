import re
from nltk.tokenize import word_tokenize
import pickle
import numpy as np
import torch
from solver import TrainSolver
from model import PointerNetworks
from tqdm import tqdm
import json

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"RE_DIGITS":1,"UNKNOWN":2,"PADDING":0}
        self.word2count = {"RE_DIGITS":1,"UNKNOWN":1,"PADDING":1}
        self.index2word = {0: "PADDING", 1: "RE_DIGITS", 2: "UNKNOWN"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.strip('\n').strip('\r').split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



def mytokenizer(inS,all_dict):

    #repDig = re.sub(r'\d+[\.,/]?\d+','RE_DIGITS',inS)
    repDig = re.sub(r'\d*[\d,]*\d+', 'RE_DIGITS', inS)
    toked = word_tokenize(repDig)
    or_toked = word_tokenize(inS)
    re_unk_list = []
    ori_list = []

    for (i,t) in enumerate(toked):
        if t not in all_dict and t not in ['RE_DIGITS']:
            re_unk_list.append('UNKNOWN')
            ori_list.append(or_toked[i])
        else:
            re_unk_list.append(t)
            ori_list.append(or_toked[i])

    labey_edus = [0]*len(re_unk_list)
    labey_edus[-1] = 1




    return ori_list,re_unk_list,labey_edus



def get_mapping(X,Y,D):

    X_map = []
    for w in X:
        if w in D:
            X_map.append(D[w])
        else:
            X_map.append(D['UNKNOWN'])

    X_map = np.array([X_map])
    Y_map = np.array([Y])



    return X_map,Y_map





def seg_cut(inputstring):


    all_voc = r'all_vocabulary.pickle'
    voca = pickle.load(open(all_voc, 'rb'))
    voca_dict = voca.word2index

    ori_X, X, Y = mytokenizer(inputstring, voca_dict)

    X_in, Y_in = get_mapping(X, Y, voca_dict)

    mymodel = PointerNetworks(voca_size =2, voc_embeddings=np.ndarray(shape=(2,300), dtype=float),word_dim=300, hidden_dim=10,is_bi_encoder_rnn=True,rnn_type='GRU',rnn_layers=3,
                 dropout_prob=0.5,use_cuda=False,finedtuning=True,isbanor=True)

    mymodel = torch.load(r'trained_model.torchsave', map_location=lambda storage, loc: storage)
    mymodel.use_cuda = False

    mymodel.eval()#.to("cuda:0")
    mysolver = TrainSolver(mymodel, '', '', '', '', '',
                           batch_size=1, eval_size=1, epoch=10, lr=1e-2, lr_decay_epoch=1, weight_decay=1e-4,
                           use_cuda=False)

    test_batch_ave_loss, test_pre, test_rec, test_f1, visdata = mysolver.check_accuracy(X_in, Y_in)

    start_b = visdata[3][0]
    end_b = visdata[2][0] + 1
    segments = []

    for i, END in enumerate(end_b):
        START = start_b[i]
        segments.append(' '.join(ori_X[START:END]))

    return segments

def read_by(path):
    with open(path, 'r') as f:
        # 读取json数据
        data = json.load(f)
    return data


def preprocess_data(data_path):
    """
    原始数据集转换为Query生成模型训练所需的格式
    """
    # 数据集下载地址：http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
    all_data = {
        "train": read_by("/home/dengyf/code/exp_1/exp_1/data/wow/ori_data/" + "train.json"),
        "valid": read_by("/home/dengyf/code/exp_1/exp_1/data/wow/ori_data/" + "valid_random_split.json"),
        "test": read_by("/home/dengyf/code/exp_1/exp_1/data/wow/ori_data/" + "test_random_split.json"),
    }
    processed_data = {
        "train": [],
        "valid": [],
        "test": [],
    }


    for stage in ["train", "valid", "test"]:
        data = all_data[stage]
        for i, item in enumerate(tqdm(data, desc=f"预处理{stage}数据集：")):
            for j, dialog in enumerate(item['dialog']):
                response = dialog['text'].replace(".", " . ").replace(",", " , ").replace(":", " : ").replace("!", " ! ").replace("?", " ? ").replace("'", " ' ")
                data[i]['dialog'][j]['response_seg'] = seg_cut(response)
        with open("/home/dengyf/code/exp_1/exp_1/data/wow/ori_data/" + f"{stage}_seg.json", 'w') as f:
            json.dump(data, f)







if __name__ == '__main__':

    preprocess_data("/home/dengyf/code/exp_1/exp_1/data/wow/cut_train.pt")