# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.nn as nn

from module.encoder import Bert, TextEncoder, RandomEmbedding, RandomTextEncoder
from module.attn import RelPositionEmbedding
from module.fusion import FLAT, Dense, Ordinary_Transf
from module.decoder import Crf


class ModelConfig:
    def __init__(self):
        # data
        self.max_len = None

        # random embedding
        self.char_vocab = None
        self.word_vocab = None
        self.char_vocab_size = None
        self.word_vocab_size = None
        self.random_dim = 768

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

        # loss
        self.focal = False

        # common
        self.layer_norm_eps = 1e-12
        self.dropout = 0.2

    def add_vocab(self):
        processed_data_path = './random_data'
        char2idx_path = os.path.join(processed_data_path, 'char2idx.json')
        word2idx_path = os.path.join(processed_data_path, 'word2idx.json')
        with open(char2idx_path, 'r') as f:
            self.char_vocab = json.load(fp=f)
        with open(word2idx_path, 'r') as f:
            self.word_vocab = json.load(fp=f)
        self.char_vocab_size = len(self.char_vocab)
        self.word_vocab_size = len(self.word_vocab)


# %% Model Overview
class Random_Crf(nn.Module):
    def __init__(self, config):
        super(Random_Crf, self).__init__()
        self.config = config
        self.params = {}
        self.layer_list = []

        self.text_encoder = RandomEmbedding(config, config.char_vocab_size)
        self.layer_list.append(self.text_encoder)

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
        text_embedded = self.text_encoder(inputs['sentence_tokens'])
        text_embedded = self.dense(text_embedded)
        output = self.crf(text_embedded, inputs['sentence_masks'])
        return output


class Random_Flat_Crf(nn.Module):
    def __init__(self, config):
        super(Random_Flat_Crf, self).__init__()
        self.config = config
        self.params = {}
        self.layer_list = []

        self.text_encoder = RandomTextEncoder(config)
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


class Bert_Transformer_Crf(nn.Module):
    """
    Bert-Transformer-CRF
    这里encoder只对字母char进行了encode，使用的是预训练bert，[batch_size, seq_len, 768]
    fusion层是普通Transformer(position encoding使用绝对位置)
    output是crf
    """
    def __init__(self, config):
        super(Bert_Transformer_Crf, self).__init__()
        self.config = config
        self.params = {}
        self.layer_list = []

        # text_embedder, 作为Transformer层的输入，[batch_siez, seq_len, emb_dim], emb_dim=768
        self.text_encoder = TextEncoder(config)
        self.layer_list.append(self.text_encoder)

        self.pe = RelPositionEmbedding(config.max_len, config.dim_pos)

        # Transformer层, [batch_siez, seq_len, emb_dim]->[batch_siez, seq_len, hidden_size]
        self.fusion = Ordinary_Transf(config)
        self.layer_list.append(self.fusion)

        # output: crf
        self.output = Crf(config)
        self.layer_list.append(self.output)

    def get_params(self):
        if not self.params:
            for layer in self.layer_list:
                for key, value in layer.get_params().items():
                    if key not in self.params:
                        self.params[key] = []
                    self.params[key].extend(value)
        return self.params

    def forward(self,  inputs, en_decode=True):
        # encoder
        encoder_inputs = {'text': inputs['sentence_tokens'],
                          'mask': inputs['sentence_masks']}
        encoder_outputs = self.text_encoder(encoder_inputs)

        fusion_inputs = {'text_vec': encoder_outputs['text_vec'],
                         'mask': inputs['sentence_masks'],
                         'pos_emb_layer': self.pe}
        fusion_outputs = self.fusion(fusion_inputs)

        # output
        result = self.output(fusion_outputs['text_vec'], inputs['sentence_masks'])

        return result

    def cal_loss(self, preds, targets, mask):
        return self.output.cal_loss(preds, targets, mask)