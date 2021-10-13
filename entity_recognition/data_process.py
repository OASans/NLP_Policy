import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import itertools
import spacy
import random
import hanlp
import numpy as np
from spacy.tokens import Span
from spacy.util import filter_spans


class DataProcessConfig:
    def __init__(self):
        self.preprocess = False

        self.label_dict = {'发布地区': 'AREA', '制定部门': 'RELDE', '政策文号': 'NUMB', '政策名称': 'TITLE',
                           '执行部门': 'EXECDE', '发布时间': 'RELT', '执行期限': 'VALIDT'}
        self.ner_tagging = 'BIO'
        self.dev_rate = 0.2
        self.test_rate = 0.2

        self.raw_data_path = '../data_process/datasets/entity.json'
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
            self.spacy_nlp = spacy.blank("zh")
            self.lattice_cutter = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
            self.stopwords = self._stopwordslist('../data_process/utils/cn_stopwords.txt')

    def _load_data(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def _stopwordslist(self, stop_word_path):
        stopwords = [line.strip() for line in open(stop_word_path, encoding='UTF-8').readlines()]
        return stopwords

    def _normalization(self, sample):
        def _get_bio(sentence, entity_list):
            sentence_len = len(sentence)
            bio_label = ['O'] * sentence_len
            for entity in entity_list:
                bio_label[entity[0]] = 'B-{}'.format(self.config.label_dict[entity[2]])
                for i in range(entity[0] + 1, entity[1] + 1):
                    bio_label[i] = 'I-{}'.format(self.config.label_dict[entity[2]])
            return bio_label

        def _get_lattice_word(sentence):
            def is_not_stopword(word):
                return True if word not in self.stopwords else False

            def lattice_cut(text):
                index = 0
                word_list = []

                cut_results = self.lattice_cutter(text)
                for word in cut_results['tok/coarse']:
                    word_len = len(word)
                    if word_len > 1 and is_not_stopword(word):  # 去掉非全汉字的词
                        word_list.append((word, index, index + word_len - 1))
                    index += word_len
                return word_list

            # 分类任务中暂时先考虑一个分词工具
            cut_func = [lattice_cut]
            lattice_word = set()
            for func in cut_func:
                words = func(sentence)
                lattice_word |= set(words)
            lattice_word = [w for w in lattice_word]
            lattice_word.sort(key=lambda x: len(x[0]))
            lattice_word.sort(key=lambda x: x[2])
            return lattice_word

        text = self.spacy_nlp(sample['sentence'])
        origin_spans = [Span(text, entity[2][0], entity[2][1] + 1, label=entity[1]) for entity in sample['entity_list']]
        filtered_spans = filter_spans(origin_spans)
        upmost_entities = [(s.start, s.end - 1, s.label_) for s in filtered_spans]
        bio_label = _get_bio(sample['sentence'], upmost_entities)
        lattice_word = _get_lattice_word(sample['sentence'])
        return {'sentence': sample['sentence'], 'entity_list': upmost_entities, 'bio_label': bio_label,
                'lattice': lattice_word}

    def _data_split(self, total_data):
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
        cartesian = itertools.product(['B-', 'I-'], list(self.config.label_dict.values()))
        labels = ['O'] + [''.join([label[0], label[1]]) for label in cartesian]
        label_num = len(labels)
        label2idx = {labels[i]: i for i in range(label_num)}
        idx2label = {i: labels[i] for i in range(label_num)}
        return label2idx, idx2label

    def preprocess(self):
        raw_path = self.config.raw_data_path
        data = self._load_data(raw_path)
        norm_data = []
        for sample in data:
            norm_data.append(self._normalization(sample))
        norm_data = np.array(norm_data)

        # 创建BIO标签映射
        label2idx, idx2label = self._label2idx()

        # 划分数据集，存json
        train_data, dev_data, test_data = self._data_split(norm_data)
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

