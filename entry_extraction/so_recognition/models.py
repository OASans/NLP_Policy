# -*- coding: utf-8 -*-
import sys
sys.path.append('/remote-home/aqshi/NLP_Policy/NLP_Policy')
sys.path.append('/remote-home/aqshi/NLP_Policy')
import torch.nn as nn
from transformers import BertModel
from pytorchcrf import CRF


class ModelConfig:
    def __init__(self):
        self.dropout_rate = 0.4

        # bert
        self.ptm_model = 'hfl/chinese-roberta-wwm-ext-large'
        self.ptm_feat_size = 1024

        # crf
        self.num_tags = None


# Bert
class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.params = {'ptm': [], 'other': []}

        self.bert = BertModel.from_pretrained(config.ptm_model, output_hidden_states=True)
        self.params['ptm'].extend([p for p in self.bert.parameters()])

    def get_params(self):
        return self.params

    def forward(self, inputs):
        text_embedded = self.bert(inputs['sentence_tokens'], inputs['sentence_masks'])['last_hidden_state']
        return text_embedded


# CRF
class Crf(nn.Module):
    def __init__(self, config):
        super(Crf, self).__init__()
        self.params = {'crf': [], 'other': []}
        self.num_tags = config.num_tags

        self.crf = CRF(config.num_tags, batch_first=True)
        self.params['crf'].extend([p for p in self.crf.parameters()])

        self.emission_linear = nn.Linear(config.ptm_feat_size, config.num_tags)
        self.params['other'].extend([p for p in self.emission_linear.parameters()])

    def get_params(self):
        return self.params

    def decode(self, emission, mask):
        """
        emission: B T L F
        """
        emission_shape = emission.shape
        result = self.crf.decode(emission, mask)
        result = result.squeeze(dim=0)
        result = result.tolist()
        return result

    def cal_emission(self, text_vec):
        emission = self.emission_linear(text_vec)
        return emission

    def cal_loss(self, preds, y_true, mask):
        emission = preds['emission']
        _loss = -self.crf(emission, y_true, mask, reduction='token_mean')
        return _loss

    def forward(self, text_vec, mask, en_pred=True):
        emission = self.cal_emission(text_vec)
        if en_pred:
            pred = self.decode(emission, mask)
        else:
            pred = None
        return {'emission': emission,
                'pred': pred}


# %% Model Overview
class Bert_Crf(nn.Module):
    def __init__(self, config):
        super(Bert_Crf, self).__init__()
        self.params = {}
        self.layer_list = []

        self.bert = Bert(config)
        self.layer_list.append(self.bert)

        self.crf = Crf(config)
        self.layer_list.append(self.crf)

    def get_params(self):
        if not self.params:
            for layer in self.layer_list:
                for key, value in layer.get_params().items():
                    if key not in self.params:
                        self.params[key] = []
                    self.params[key].extend(value)
        return self.params

    def cal_loss(self, preds, targets, mask):
        return self.crf.cal_loss(preds, targets, mask)

    def forward(self, inputs):
        text_embedded = self.bert(inputs)
        output = self.crf(text_embedded, inputs['sentence_masks'])
        return output
