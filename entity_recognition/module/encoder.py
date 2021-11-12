# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel
from pytorchcrf import CRF


# %% encoder
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


class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.config = config
        self.params = {'ptm': [], 'other': []}
        self.detach_ptm_flag = False

        self.bert = BertModel.from_pretrained(config.ptm_model, output_hidden_states=True)
        self.params['ptm'].extend([p for p in self.bert.parameters()])

        self.w2v_linear = nn.Linear(config.w2v_feat_size, config.ptm_feat_size)
        self.layer_norm = nn.LayerNorm(config.ptm_feat_size, eps=config.layer_norm_eps)
        self.params['other'].extend([p for p in self.w2v_linear.parameters()])
        self.params['other'].extend([p for p in self.layer_norm.parameters()])

        self.dropout = nn.Dropout(config.dropout)

    def get_params(self):
        return self.params

    def get_bert_vec(self, text, text_mask, text_pos=None):
        if text_pos is None:
            bert_output = self.bert(text, text_mask)
        else:
            bert_output = self.bert(text, text_mask, position_ids=text_pos)
        text_vecs = bert_output['hidden_states']
        text_vecs = list(text_vecs)
        return text_vecs

    def forward(self, inputs):
        text, mask, = inputs['text'], inputs['mask']

        text_vecs = self.get_bert_vec(text, mask, inputs['pos'] if 'pos' in inputs else None)
        char_vec = text_vecs[self.config.num_ptm_layers]
        if self.config.w2v:
            word_text, word_mask, word_indice = inputs['word_text'], inputs['word_mask'], inputs['word_indice']
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pos = torch.arange(0, text.size(1)).long().unsqueeze(dim=0).to(device)  # TODO
            pos = pos * mask.long()
            char_s = char_e = pos
            word_s, word_e, word_abs = inputs['word_pos_b'], inputs['word_pos_e'], inputs['word_pos_abs']
            part_size = inputs['part_size']
            word_vec = self.w2v_linear(word_text)
            word_vec = self.dropout(word_vec)
            word_vec = self.layer_norm(word_vec)

            # # 新增：绝对位置
            # part0, part1_1 = torch.split(pos, part_size[:2], dim=1)
            # part1_2, part2 = torch.split(word_abs, part_size[1:], dim=1)
            # part1 = part1_1 + part1_2
            # char_word_abs = torch.cat([part0, part1, part2], dim=1)

            part0, part1_1 = torch.split(char_vec, part_size[:2], dim=1)
            part1_2, part2 = torch.split(word_vec, part_size[1:], dim=1)
            part1 = part1_1 + part1_2
            char_word_vec = torch.cat([part0, part1, part2], dim=1)

            part0, part1_1 = torch.split(mask, part_size[:2], dim=1)
            part1_2, part2 = torch.split(word_mask, part_size[1:], dim=1)
            part1 = part1_1 + part1_2
            char_word_mask = torch.cat([part0, part1, part2], dim=1)

            part0, part1_1 = torch.split(char_s, part_size[:2], dim=1)
            part1_2, part2 = torch.split(word_s, part_size[1:], dim=1)
            part1 = part1_1 + part1_2
            char_word_s = torch.cat([part0, part1, part2], dim=1)

            part0, part1_1 = torch.split(char_e, part_size[:2], dim=1)
            part1_2, part2 = torch.split(word_e, part_size[1:], dim=1)
            part1 = part1_1 + part1_2
            char_word_e = torch.cat([part0, part1, part2], dim=1)
            return {'char_word_vec': char_word_vec,
                    'char_word_mask': char_word_mask,
                    'char_word_s': char_word_s,
                    'char_word_e': char_word_e}
        else:
            return {'text_vec': char_vec}

