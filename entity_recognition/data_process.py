import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
        self.ner_tagging = 'BIOES'
        self.num_tags = 29 if self.ner_tagging == 'BIOES' else None

        # fixed
        self.ptm_model = 'hfl/chinese-roberta-wwm-ext-large'
        self.label_dict = {'发布地区': 'AREA', '制定部门': 'RELDE', '政策文号': 'NUMB', '政策名称': 'TITLE',
                           '执行部门': 'EXECDE', '发布时间': 'RELT', '执行期限': 'VALIDT'}
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
        self.token_max_len = self.config.max_len  # token的最大长度
        self.max_len = self.config.max_len - 2  # 实际最大长度

        if self.config.preprocess:
            self.spacy_nlp = spacy.blank("zh")
            self.lattice_cutter = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
            self.word_w2v = get_w2v_vocab()
            self.word_w2v = dict([(word, index) for index, word in enumerate(self.word_w2v)])
            self.stopwords = self._stopwordslist('../data_process/utils/cn_stopwords.txt')
            self.tokenizer = MyBertTokenizer.from_pretrained(config.ptm_model)

    def _load_data(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def _stopwordslist(self, stop_word_path):
        stopwords = [line.strip() for line in open(stop_word_path, encoding='UTF-8').readlines()]
        return stopwords

    def _normalization(self, sample):
        def _get_lattice_word(sentence, raw2decode):
            def is_not_stopword(word):
                return True if word not in self.stopwords else False

            def lattice_cut(text):
                coarse_word_list = []
                fine_word_list = []
                cut_results = self.lattice_cutter(text)
                index = 0
                for word in cut_results['tok/coarse']:
                    word_len = len(word)
                    if word_len > 1 and is_not_stopword(word):
                        coarse_word_list.append((word, index, index + word_len - 1))
                    index += word_len
                index = 0
                for word in cut_results['tok/fine']:
                    word_len = len(word)
                    if word_len > 1 and is_not_stopword(word):
                        fine_word_list.append((word, index, index + word_len - 1))
                    index += word_len
                return coarse_word_list, fine_word_list

            def adjust_lattice(lattice_word):
                lattice_word = [[w[0], raw2decode[w[1]], raw2decode[w[2]]] for w in lattice_word]
                lattice_word.sort(key=lambda x: len(x[0]))
                lattice_word.sort(key=lambda x: x[2])
                return lattice_word

            # 暂时先考虑一个分词工具
            cut_func = [lattice_cut]
            coarse_lattice_word = set()
            fine_lattice_word = set()
            for func in cut_func:
                coarse_words, fine_words = func(sentence)
                coarse_lattice_word |= set(coarse_words)
                fine_lattice_word |= set(fine_words)
            coarse_lattice_word = adjust_lattice(coarse_lattice_word)
            fine_lattice_word = adjust_lattice(fine_lattice_word)
            return coarse_lattice_word, fine_lattice_word

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

        def _convert_lattice_to_token(fine_lattice):
            lattice_tokens = []
            for lword in fine_lattice:
                if lword[0] in self.word_w2v:
                    lword_index = self.word_w2v[lword[0]]
                    lattice_tokens.append([lword_index, lword[1], lword[2]])
            return lattice_tokens

        def _convert_sentence_to_token(sentence):
            tokens = self.tokenizer.my_encode(sentence, max_length=self.token_max_len, add_special_tokens=True,
                                            truncation=True)
            decode2raw, raw2decode = self.tokenizer.get_token_map(sentence)
            return tokens, decode2raw, raw2decode

        tokens, decode2raw, raw2decode = _convert_sentence_to_token(sample['sentence'])

        text = self.spacy_nlp(sample['sentence'])
        origin_spans = [Span(text, entity[2][0], entity[2][1] + 1, label=entity[1]) for entity in sample['entity_list']]
        filtered_spans = filter_spans(origin_spans)
        upmost_entities = [(raw2decode[s.start], raw2decode[s.end - 1], self.config.label_dict[s.label_]) for s in filtered_spans]

        coarse_lattice, fine_lattice = _get_lattice_word(sample['sentence'], raw2decode)
        lattice_tokens = _convert_lattice_to_token(fine_lattice)
        return {'sid': sample['sid'], 'sentence_num': sample['sentence_num'], 'sentence_tokens': tokens,
                'raw2decode': raw2decode, 'decode2raw': decode2raw, 'entity_list': upmost_entities,
                'coarse_lattice': coarse_lattice, 'lattice_tokens': lattice_tokens}

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
        cartesian = itertools.product(['B-', 'I-', 'E-', 'S-'], list(self.config.label_dict.values()))
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

        # 创建BIOES标签映射
        label2idx, idx2label = self._label2idx()

        # 划分数据集，存json
        train_data, dev_data, test_data = self._data_split(norm_data)
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
        return EntityDataSet(data, self.config.debug_mode)


class EntityDataSet(Dataset):
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