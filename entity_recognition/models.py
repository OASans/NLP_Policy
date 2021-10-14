# -*- coding: utf-8 -*-
import torch.nn as nn
from transformers import BertModel
from pytorchcrf import CRF


class ModelConfig:
    def __init__(self):
        self.dropout_rate = 0.4

        # bert
        self.ptm_model = 'hfl/chinese-roberta-wwm-ext-large'

        # crf
        self.in_feat_size = None
        self.num_tags = None


# Bert
class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.params = {'ptm': [], 'other': []}

        self.bert = BertModel.from_pretrained(config.ptm_model, output_hidden_states=True)
        self.params['ptm'].extend([p for p in self.bert.parameters()])

        # self.dropout_rate = config.dropout_rate
        # self.dropout = nn.Dropout(self.dropout_rate)

    def get_params(self):
        return self.params

    def forward(self, inputs):
        text_embedded = self.bert(inputs['X'], inputs['X_mask'])['last_hidden_state']
        text_embedded = text_embedded[:, 1:-1, :]  # TODO: 为啥来着
        # output = self.dropout(text_embedded)
        return text_embedded


# CRF
class Crf(nn.Module):
    def __init__(self, config):
        super(Crf, self).__init__()
        self.params = {'crf': [], 'other': []}
        self.num_tags = config.num_tags

        self.crf = CRF(config.num_tags, batch_first=True)
        self.params['crf'].extend([p for p in self.crf.parameters()])

        self.emission_linear = nn.Linear(config.in_feat_size, config.num_tags)
        self.params['other'].extend([p for p in self.emission_linear.parameters()])

    def get_params(self):
        return self.params

    def decode(self, emission, mask):
        """
        emission: B T L F
        """
        emission_shape = emission.size()
        mask = mask.unsqueeze(dim=1)
        mask = mask.repeat(1, emission_shape[1], 1)
        mask = mask.reshape([-1, mask.size(2)])
        emission = emission.reshape([-1, emission_shape[2], emission.size(3)])
        result = self.crf.decode(emission, mask)
        result = result.reshape([-1, emission_shape[1], mask.size(1)])
        result = result.tolist()
        return result

    def cal_emission(self, text_vec):
        emission = self.emission_linear(text_vec)
        emission = emission.reshape(list(emission.size()[:2]) + [1, self.num_tags])
        emission = emission.permute([0, 2, 1, 3])  # B L T F -> B T L F
        return emission

    def cal_loss(self, preds, targets, mask):
        emission = preds['emission']
        y_true = targets['y_true']
        mask = mask.unsqueeze(dim=1)
        mask = mask.repeat(1, emission.size(1), 1)
        mask = mask.reshape([-1, mask.size(2)])
        emission = emission.reshape([-1, emission.size(2), emission.size(3)])  # B*T L F
        y_true = y_true.reshape([-1, y_true.size(2)])
        _loss = -self.crf(emission, y_true, mask, reduction='token_mean')
        return _loss

    def forward(self, inputs, en_pred=True):
        text_vec, mask = inputs['text_vec'], inputs['mask']
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
        self.bert = Bert(config)
        self.crf = Crf(config)

    def cal_loss(self, preds, targets, mask):
        return self.crf.cal_loss(preds, targets, mask)

    def forward(self, inputs):
        sentence_tokens, sentence_masks = inputs['sentence_tokens'], inputs['sentence_masks']
        text_embedded = self.bert(sentence_tokens)
        output = self.crf(text_embedded)
        return output
