import collections
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import copy
import itertools
import spacy
import random
import hanlp
import numpy as np
from spacy.tokens import Span
from spacy.util import filter_spans
from torch.utils.data import Dataset

from NLP_Policy.data_process.word2vec import get_w2v_vocab
from NLP_Policy.utils.tokenizer import MyBertTokenizer


class DataProcessConfig:
    def __init__(self):
        self.preprocess = False
        self.debug_mode = False
        self.dev_rate = 0.2
        self.test_rate = 0.2
        self.max_len = 512

        # fixed
        self.raw_data_path = '../../data_process/datasets/entry.json'
        self.ptm_model = 'hfl/chinese-roberta-wwm-ext-large'
        self.processed_data_path = os.path.join(os.getcwd(), 'data/')
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
        self.total_path = os.path.join(self.processed_data_path, 'total.json')
        self.train_path = os.path.join(self.processed_data_path, 'train.json')
        self.dev_path = os.path.join(self.processed_data_path, 'dev.json')
        self.test_path = os.path.join(self.processed_data_path, 'test.json')
        self.label2idx_path = os.path.join(self.processed_data_path, 'label2idx.json')
        self.idx2label_path = os.path.join(self.processed_data_path, 'idx2label.json')

        # relation
        self.relation_dict = {'等于': 'EQUAL', '不等于': 'UNEQUAL', '小于': 'SMALLER', '大于': 'BIGGER', '包含': 'CONTAIN',
                              '不包含': 'UNCONTAIN', '符合': 'ACCORD', '是': 'IS', '否': 'ISNOT', '': 'NOREL'}


class DataProcess:
    def __init__(self, config):
        self.config = config
        self.token_max_len = self.config.max_len  # token的最大长度
        self.max_len = self.config.max_len - 2  # 实际最大长度

        if self.config.preprocess:
            self.spacy_nlp = spacy.blank("zh")
            self.tokenizer = MyBertTokenizer.from_pretrained(config.ptm_model)

    def _load_data(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def _normalization(self, sample):

        # 先不切句子
        # def _split_text(sample):
        #
        #     def split_by_len(sample_len):
        #         if len(sample_len['sentence']) <= self.config.max_len:
        #             return [sample_len]
        #         out_samples = []
        #         right_limit = 0
        #         rest_text = sample_len['sentence']
        #         while len(rest_text) > self.config.max_len:
        #             new_sample = copy.deepcopy(sample_len)
        #             new_sample['entities'] = []
        #             for char_index in range(self.config.max_len - 1, -1, -1):
        #                 if (rest_text[char_index] in ('，', '。', '!', '?')) or char_index == 0:
        #                     if char_index == 0:
        #                         char_index = self.config.max_len - 1
        #                     left_limit = right_limit
        #                     right_limit += char_index + 1
        #                     new_sample['sentence'] = rest_text[:char_index + 1]
        #
        #                     for entity in sample_len['entity_list']:
        #                         if entity[0] >= left_limit and entity[1] < right_limit:
        #                             new_entity = (entity[0] - left_limit, entity[1] - left_limit, entity[2])
        #                             new_sample['entity_list'].append(new_entity)
        #
        #                     rest_text = rest_text[char_index + 1:]
        #                     out_samples.append(new_sample)
        #                     break
        #         else:
        #             left_limit = right_limit
        #             new_sample = copy.deepcopy(sample_len)
        #             new_sample['sentence'] = rest_text
        #             new_sample['entity_list'] = []
        #             for entity in sample_len['entity_list']:
        #                 if entity[0] >= left_limit:
        #                     new_entity = (entity[0] - left_limit, entity[1] - left_limit, entity[2])
        #                     new_sample['entity_list'].append(new_entity)
        #             out_samples.append(new_sample)
        #         return out_samples
        #
        #     new_samples = split_by_len(sample)
        #     new_samples.sort(key=lambda x: x['sub_id'])
        #     for index, ppp in enumerate(new_samples):
        #         ppp['sub_id'] = index
        #     return new_samples

        def _convert_sentence_to_token(sentence):
            tokens = self.tokenizer.my_encode(sentence, max_length=self.token_max_len, add_special_tokens=True,
                                              truncation=True)
            decode2raw, raw2decode = self.tokenizer.get_token_map(sentence)
            return tokens, decode2raw, raw2decode

        # 去掉subject-object对不完整的三元组
        # sample = _drop_incomplete_triple(sample)

        tokens, decode2raw, raw2decode = _convert_sentence_to_token(sample['sentence'])

        text = self.spacy_nlp(sample['sentence'])
        origin_spans = []
        origin_subjects = []
        origin_objects = []
        relation_dict = {}
        object_dict = {}
        for entry in sample['entry_list']:
            if entry[-1][0] != -1 and entry[-2][0] != -1:
                subject = Span(text, entry[-2][0], entry[-2][1] + 1, label='subject')
                object = Span(text, entry[-1][0], entry[-1][1] + 1, label='object')
                origin_spans.append(object)
                origin_spans.append(subject)
                origin_subjects.append(subject)
                origin_objects.append(object)
                relation_dict[subject] = entry[-4]
                object_dict[subject] = object
        filtered_spans = filter_spans(origin_spans)
        legal_subjects = [s for s in filtered_spans if s in origin_subjects]
        legal_objects = [s for s in filtered_spans if s in origin_objects]

        legal_spos = []
        for subject in legal_subjects:
            object = object_dict[subject]
            if object in legal_objects:
                legal_spos.append({'s': (raw2decode[subject.start], raw2decode[subject.end - 1]),
                                   'p': relation_dict[subject],
                                   'o': (raw2decode[object.start], raw2decode[object.end - 1])})

        new_samples = []
        for i, spo in enumerate(legal_spos):
            new_sample = {'sid': sample['sid'], 'sub_id': sample['sid'] + str(i), 'sentence_tokens': tokens,
                          'raw2decode': raw2decode, 'decode2raw': decode2raw, 'spo': spo}
            new_samples.append(new_sample)
        return new_samples

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
        label_num = len(self.config.relation_dict)
        labels = list(self.config.relation_dict.values())
        label2idx = {labels[i]: i for i in range(label_num)}
        idx2label = {i: labels[i] for i in range(label_num)}
        return label2idx, idx2label

    def preprocess(self):
        raw_path = self.config.raw_data_path
        data = self._load_data(raw_path)
        norm_data = []
        for sample in data:
            norm_data.extend(self._normalization(sample))
        norm_data = np.array(norm_data)

        # 划分数据集
        train_data, dev_data, test_data = self._data_split(norm_data)

        # 创建标签映射
        label2idx, idx2label = self._label2idx()

        # 存json
        with open(self.config.total_path, 'w') as f:
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
        return RelationDataSet(data, self.config.debug_mode)


class RelationDataSet(Dataset):
    def __init__(self, data, debug_mode):
        self.data = data
        if debug_mode:
            self.len = 100 if len(self.data) > 100 else len(self.data)
        else:
            self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index]