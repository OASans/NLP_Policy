# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from module.encoder import Bert, TextEncoder
from module.attn import RelPositionEmbedding
from module.fusion import FLAT, Dense
from module.decoder import Crf


class ModelConfig:
    def __init__(self):
        # data
        self.max_len = None

        # bert
        self.ptm_model = 'hfl/chinese-roberta-wwm-ext'
        self.ptm_feat_size = 768
        self.num_ptm_layers = 12

        # w2v
        self.w2v = False
        self.w2v_feat_size = 300

        # attn
        self.dim_pos = 160
        self.num_pos = 4  # flat
        self.attn_dropout = 0.1
        self.hidden_dropout = 0.1
        self.intermediate_size = 640

        # fusion
        self.num_flat_layers = 1
        self.num_heads = 8
        self.scaled = False
        self.en_ffd = True
        self.hidden_size = self.dim_pos

        # crf
        self.num_tags = None
        self.focal = False

        # common
        self.layer_norm_eps = 1e-12
        self.dropout = 0.2


# %% Model Overview
class Bert_Crf(nn.Module):
    def __init__(self, config):
        super(Bert_Crf, self).__init__()
        self.params = {}
        self.layer_list = []

        self.bert = Bert(config)
        self.layer_list.append(self.bert)

        self.dense = Dense(config)
        self.layer_list.append(self.dense)

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
        text_embedded = self.dense(text_embedded)
        output = self.crf(text_embedded, inputs['sentence_masks'])
        return output


class Bert_Flat_Crf(nn.Module):
    def __init__(self, config):
        super(Bert_Flat_Crf, self).__init__()
        self.config = config
        self.params = {}
        self.layer_list = []

        self.text_encoder = TextEncoder(config)
        self.layer_list.append(self.text_encoder)

        self.pe = RelPositionEmbedding(config.max_len, config.dim_pos)

        self.fusion = FLAT(config)
        self.layer_list.append(self.fusion)

        self.output = Crf(config)
        self.layer_list.append(self.output)

        self.params = {}

    def get_params(self):
        if not self.params:
            for layer in self.layer_list:
                for key, value in layer.get_params().items():
                    if key not in self.params:
                        self.params[key] = []
                    self.params[key].extend(value)
        return self.params

    def forward(self, inputs, en_decode=True):
        # encoder
        encoder_inputs = {'text': inputs['sentence_tokens'],
                          'mask': inputs['sentence_masks']}
        if self.config.w2v:
            encoder_inputs['pos_emb'] = self.pe
            encoder_inputs['word_text'] = inputs['word_text']
            encoder_inputs['word_mask'] = inputs['word_mask']
            encoder_inputs['word_pos_b'] = inputs['word_pos_b']
            encoder_inputs['word_pos_e'] = inputs['word_pos_e']
            encoder_inputs['word_pos_abs'] = inputs['word_pos_abs']
            encoder_inputs['word_indice'] = inputs['word_indice']
            encoder_inputs['part_size'] = inputs['part_size']
        encoder_outputs = self.text_encoder(encoder_inputs)

        # fusion
        fusion_inputs = {'char_word_vec': encoder_outputs['char_word_vec'],
                         'char_word_mask': encoder_outputs['char_word_mask'],
                         'char_word_s': encoder_outputs['char_word_s'],
                         'char_word_e': encoder_outputs['char_word_e'],
                         'part_size': inputs['part_size'],
                         'pos_emb_layer': self.pe}
        fusion_outputs = self.fusion(fusion_inputs)

        # output
        result = self.output(fusion_outputs['text_vec'], inputs['sentence_masks'])

        return result

    def cal_loss(self, preds, targets, mask):
        return self.output.cal_loss(preds, targets, mask)