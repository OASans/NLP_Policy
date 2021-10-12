import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import re
import json
import hanlp
import random
import numpy as np
import pandas as pd

from NLP_Policy.data_process.word2vec import get_w2v_vocab



class DataProcessConfig:
    def __init__(self):
        self.preprocess = False

        self.labels = ['', '政策目标', '申请审核程序', '资金管理-资金来源', '资金管理-管理原则', '监管评估-监督管理',
                       '监管评估-考核评估', '政策内容-人才培养', '政策内容-资金支持', '政策内容-技术支持', '政策内容-公共服务',
                       '政策内容-组织建设', '政策内容-目标规划', '政策内容-法规管制', '政策内容-政策宣传', '政策内容-税收优惠',
                       '政策内容-金融支持', '政策内容-政府采购', '政策内容-对外承包', '政策内容-公私合作', '政策内容-海外合作']
        self.dev_rate = 0.2
        self.test_rate = 0.2

        self.raw_data_path = '../data_process/datasets/sentence_classification.json'
        self.processed_data_path = os.path.join(os.getcwd(), 'data/')
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
        self.total_path = os.path.join(self.processed_data_path, 'total.json')
        self.train_path = os.path.join(self.processed_data_path, 'train.json')
        self.dev_path = os.path.join(self.processed_data_path, 'dev.json')
        self.test_path = os.path.join(self.processed_data_path, 'test.json')
        self.label2idx_path = os.path.join(self.processed_data_path, 'label2idx.json')
        self.idx2label_path = os.path.join(self.processed_data_path, 'idx2label.json')


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

                cut_results = self.lattice_cutter(text)
                for word in cut_results['tok/fine']:
                    word_len = len(word)
                    if word_len > 1 and is_all_chinese(word) and is_not_stopword(word):  # 去掉非全汉字的词
                        word_list.append((word, index, index + word_len - 1))
                    index += word_len
                return word_list

            def lattice_to_token(lattice_words):
                lattice_tokens = []
                for lword in lattice_words:
                    if lword[0] in self.word_w2v:
                        lword_index = self.word_w2v[lword[0]]
                        lattice_tokens.append(lword_index)
                return lattice_tokens

            # 分类任务中暂时先考虑一个分词工具
            cut_func = [lattice_cut]
            lattice_word = set()
            for func in cut_func:
                words = func(sample['sentence'])
                lattice_word |= set(words)
            lattice_word = [w for w in lattice_word]
            lattice_word.sort(key=lambda x: len(x[0]))
            lattice_word.sort(key=lambda x: x[2])

            return {'sid': sample['sid'], 'sentence': sample['sentence'], 'sentence_type': sample['sentence_type'], 'lattice': lattice_word,
                    'lattice_token': lattice_to_token(lattice_word), 'sentence_num': sample['sentence_num']}
        return _get_lattice_word(sample)

    def data_split(self, total_data):
        """
        randomly split data into train, dev, and test set
        """

        def get_split_index(seed=9):
            """
            获得数据集的划分index
            """
            # 固定随机种子
            total_num = total_data.shape[0]
            dev_num = int(total_num * self.config.dev_rate)
            test_num = int(total_num * self.config.test_rate)

            total_range_list = [i for i in range(total_num)]
            random.seed(seed)
            dev_index_list = random.sample(total_range_list, dev_num)

            total_range_list = [i for i in total_range_list if i not in dev_index_list]
            random.seed(seed)
            test_index_list = random.sample(total_range_list, test_num)

            train_index_list = [i for i in total_range_list if i not in test_index_list]

            return train_index_list, dev_index_list, test_index_list
        train_index_list, dev_index_list, test_index_list = get_split_index()
        train_data = total_data[train_index_list]
        dev_data = total_data[dev_index_list]
        test_data = total_data[test_index_list]
        return train_data, dev_data, test_data

    def _label2idx(self):
        label_num = len(self.config.labels)
        label2idx = {self.config.labels[i]: i for i in range(label_num)}
        idx2label = {i: self.config.labels[i] for i in range(label_num)}
        return label2idx, idx2label

    def preprocess(self):
        raw_path = self.config.raw_data_path
        data = self._load_data(raw_path)
        norm_data = []
        for sample in data:
            norm_data.append(self._normalization(sample))
        norm_data = np.array(norm_data)


        # 划分数据集，存json
        train_data, dev_data, test_data = self.data_split(norm_data)
        label2idx, idx2label = self._label2idx()
        with open(self.config.train_path, 'w') as f:
            json.dump(norm_data.tolist(), f)
        with open(self.config.train_path, 'w') as f:
            json.dump(train_data.tolist(), f)
        with open(self.config.dev_path, 'w') as f:
            json.dump(dev_data.tolist(), f)
        with open(self.config.test_path, 'w') as f:
            json.dump(test_data.tolist(), f)
        with open(self.config.label2idx_path, 'w') as f:
            json.dump(label2idx, f)
        with open(self.config.idx2label_path, 'w') as f:
            json.dump(idx2label, f)

    def get_data(self, data_type):
        if data_type == 'label':
            path1 = self.config.label2idx_path
            path2 = self.config.idx2label_path
            with open(path1, 'r') as f:
                label2idx = json.load(fp=f)
            with open(path2, 'r') as f:
                idx2label = json.load(fp=f)
            return label2idx, idx2label
        if data_type == 'train':
            path = self.config.train_path
        elif data_type == 'dev':
            path = self.config.dev_path
        else:
            path = self.config.test_path
        with open(path, 'r') as f:
            data = np.array(json.load(fp=f))
        return data

