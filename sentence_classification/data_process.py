import os
import re
import json
import hanlp
import numpy as np
import pandas as pd

from ..data_process.word2vec import get_w2v_vocab, get_w2v_vector
from models import MethodE


class DataProcessConfig:
    def __init__(self, method):
        self.labels = ['', '政策目标', '申请审核程序', '资金管理-资金来源', '资金管理-管理原则', '监管评估-监督管理',
                       '监管评估-考核评估', '政策内容-人才培养', '政策内容-资金支持', '政策内容-技术支持', '政策内容-公共服务',
                       '政策内容-组织建设', '政策内容-目标规划', '政策内容-法规管制', '政策内容-政策宣传', '政策内容-税收优惠',
                       '政策内容-金融支持', '政策内容-政府采购', '政策内容-对外承包', '政策内容-公私合作', '政策内容-海外合作']
        self.label_num = len(self.labels)
        self.label2idx = {}
        self.idx2label = {}
        self.dev_rate = 0.2
        self.test_rate = 0.2

        self.method = method

        self.raw_data_path = '../data_process/datasets/sentence_classification.json'

        self.processed_data_path = os.path.join(os.getcwd(), 'data/')
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
        self.train_path = os.path.join(self.processed_data_path, 'train.json')
        self.dev_path = os.path.join(self.processed_data_path, 'dev.json')
        self.test_path = os.path.join(self.processed_data_path, 'test.json')

        self.preprocess = False


class DataProcess:
    def __init__(self, config):
        self.config = config

        if self.config.preprocess:
            self.word_w2v = get_w2v_vocab()
            self.word_w2v = dict([(word, index) for index, word in enumerate(self.word_w2v)])
            self.lattice_cutter = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
            self.stopwords = self._stopwordslist('../data_process/utils/cn_stopwords.txt')

    def _stopwordslist(self, stop_word_path):
        stopwords = [line.strip() for line in open(stop_word_path, encoding='UTF-8').readlines()]
        return stopwords

    def _load_data(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def _normalization(self, sample):
        def _get_lattice_word(sample):
            def is_all_chinese(word_str):
                for c in word_str:
                    if not '\u4e00' <= c <= '\u9fa5':
                        return False
                return True

            def is_not_stopword(word):
                return True if word not in self.stopwords else False

            def lattice_cut(text):
                index = 0
                word_list = []
                for word in self.lattice_cutter(text):
                    word_len = len(word)
                    if word_len > 1 and is_all_chinese(word) and is_not_stopword(word):
                        word_list.append((word, index, index + word_len - 1))
                    index += word_len
                return word_list

            def lattice_to_vec(lattice_word):
                lattice_vec = []
                for lword in lattice_word:
                    if lword[0] in self.word_w2v:
                        lword_index = self.word_w2v[lword[0]]
                        lattice_vec.append(lword_index)
                return lattice_vec

            cut_func = [lattice_cut]
            lattice_word = set()
            for func in cut_func:
                words = func(sample['text'])
                lattice_word |= set(words)
            lattice_word = [w for w in lattice_word]
            lattice_word.sort(key=lambda x: len(x[0]))
            lattice_word.sort(key=lambda x: x[2])
            sample['lattice'] = lattice_word
            sample['lattice_vec'] = lattice_to_vec(lattice_word)
            return sample
        return _get_lattice_word(sample)

    def preprocess(self):
        raw_path = self.config.raw_data_path
        data = self._load_data(raw_path)
        norm_data = []
        for sample in data:
            norm_data.append(self._normalization(sample))

        # 划分数据集
        print('test')
        # TODO: 仲未test！！！

    def get_data(self, data_type):
        if data_type == 'train':
            path = self.config.train_path
        elif data_type == 'dev':
            path = self.config.dev_path
        else:
            path = self.config.test_path
        with open(path, 'r') as f:
            data = np.array(json.load(fp=f))
        return data

