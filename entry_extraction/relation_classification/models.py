# -*- coding: utf-8 -*-
import sys
sys.path.append('/remote-home/aqshi/NLP_Policy/NLP_Policy')
sys.path.append('/remote-home/aqshi/NLP_Policy')

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class ModelConfig:
    def __init__(self):
        self.dropout = 0.4
        self.device = torch.cuda.current_device()
        self.use_cuda = None

        # bert
        self.ptm_model = 'hfl/chinese-roberta-wwm-ext-large'
        self.ptm_feat_size = 1024

        self.embedding_dim = self.ptm_feat_size

        # bilstm 1
        self.lstm1_hidden_dim = 512
        self.lstm1_layer_num = 2

        # bilstm 2
        self.lstm2_layer_num = 2
        self.lstm2_hidden_dim = 256


# %% base classes
# Bert
class Bert(nn.Module):
    def __init__(self, ptm_model):
        super(Bert, self).__init__()
        self.params = {'ptm': [], 'other': []}

        self.bert = BertModel.from_pretrained(ptm_model, output_hidden_states=True)
        self.params['ptm'].extend([p for p in self.bert.parameters()])

    def get_params(self):
        return self.params

    def forward(self, inputs):
        text_embedded = self.bert(inputs['sentence_tokens'], inputs['sentence_masks'])['last_hidden_state']
        return text_embedded


# BiLSTM
class BiLSTM(nn.Module):
    """
    batch_size * seq_len * embed_dim => batch_size * seq_len * hidden_dim
    """
    def __init__(self, input_dim, hidden_dim, layer_num, dropout):
        super(BiLSTM, self).__init__()
        self.params = {'ptm': [], 'other': []}
        self.embedding_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = layer_num

        self.rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2, num_layers=self.num_layers,
                           bidirectional=True, batch_first=True, dropout=dropout)
        self.params['other'].extend([p for p in self.rnn.parameters()])

    def get_params(self):
        return self.params

    def forward(self, words_embedded):
        output, _ = self.rnn(words_embedded)
        # [batch_size, seq_len, hidden_dim]
        return output

# %% Bert-ESIM
# Layer 1: Input Encoding
class InputEncoding(nn.Module):
    """
    batch_size * seq_len * embed_dim => batch_size * seq_len * hidden_dim
    """
    def __init__(self, config):
        super(InputEncoding, self).__init__()
        self.config = config
        self.params = {}
        self.layer_list = []

        self.embedding_layer = Bert(config.ptm_model)
        self.layer_list.append(self.embedding_layer)

        self.bilstm_layer = BiLSTM(config.embedding_dim, config.lstm1_hidden_dim, config.lstm1_layer_num, config.dropout)
        self.layer_list.append(self.bilstm_layer)

    def get_params(self):
        if not self.params:
            for layer in self.layer_list:
                for key, value in layer.get_params().items():
                    if key not in self.params:
                        self.params[key] = []
                    self.params[key].extend(value)
        return self.params

    def forward(self, inputs):
        def get_so_embedded(text_embedded, so_pairs):
            max_len = 0
            batch_size = len(so_pairs)
            s_idxs = []
            o_idxs = []
            for i in range(batch_size):
                s_idx = [idx for idx in range(so_pairs[i][0][0], so_pairs[i][0][1] + 1)]
                o_idx = [idx for idx in range(so_pairs[i][1][0], so_pairs[i][1][1] + 1)]
                s_idxs.append(s_idx)
                o_idxs.append(o_idx)
                max_len = max(max_len, so_pairs[i][0][1] - so_pairs[i][0][0] + 1)
                max_len = max(max_len, so_pairs[i][1][1] - so_pairs[i][1][0] + 1)
            s_embedded = torch.empty((0, max_len, self.config.embedding_dim)).to(self.config.device)
            o_embedded = torch.empty((0, max_len, self.config.embedding_dim)).to(self.config.device)
            s_masks = torch.empty((0, max_len)).to(self.config.device)
            o_masks = torch.empty((0, max_len)).to(self.config.device)
            for i in range(batch_size):
                s_idx = s_idxs[i]
                o_idx = o_idxs[i]
                s = torch.index_select(text_embedded[i], 0, torch.tensor(s_idx).to(self.config.device))
                o = torch.index_select(text_embedded[i], 0, torch.tensor(o_idx).to(self.config.device))
                s = torch.cat((s, torch.zeros(max_len - len(s_idx), self.config.embedding_dim).to(self.config.device)), 0).unsqueeze(0)
                o = torch.cat((o, torch.zeros(max_len - len(o_idx), self.config.embedding_dim).to(self.config.device)), 0).unsqueeze(0)
                s_mask = torch.cat((torch.ones(len(s_idx)), torch.zeros(max_len - len(s_idx))), 0).unsqueeze(0).to(self.config.device)
                o_mask = torch.cat((torch.ones(len(o_idx)), torch.zeros(max_len - len(o_idx))), 0).unsqueeze(0).to(self.config.device)
                s_embedded = torch.cat((s_embedded, s), 0)
                o_embedded = torch.cat((o_embedded, o), 0)
                s_masks = torch.cat((s_masks, s_mask), 0)
                o_masks = torch.cat((o_masks, o_mask), 0)
            return s_embedded, o_embedded, s_masks, o_masks

        text_embedded = self.embedding_layer(inputs)  # batch_size, seq_len, embed_dim

        # batch_size, max_so_len, embed_dim
        s_embedded, o_embedded, s_masks, o_masks = get_so_embedded(text_embedded, inputs['so_pairs'])

        # packed bilstm
        s_lengths = s_masks.sum(dim=1).numpy().tolist() if not self.config.use_cuda else s_masks.sum(dim=1).cpu().numpy().tolist()
        o_lengths = o_masks.sum(dim=1).numpy().tolist() if not self.config.use_cuda else o_masks.sum(dim=1).cpu().numpy().tolist()
        s_packed = nn.utils.rnn.pack_padded_sequence(s_embedded, s_lengths, batch_first=True, enforce_sorted=False)
        o_packed = nn.utils.rnn.pack_padded_sequence(o_embedded, o_lengths, batch_first=True, enforce_sorted=False)
        s_packed_output = self.bilstm_layer(s_packed)
        o_packed_output = self.bilstm_layer(o_packed)
        s_output, _ = nn.utils.rnn.pad_packed_sequence(s_packed_output, batch_first=True)
        # batch_size, max_s_len, hidden_dim
        o_output, _ = nn.utils.rnn.pad_packed_sequence(o_packed_output, batch_first=True)
        # batch_size, max_o_len, hidden_dim

        # s和o会分别被pad pack到它们各自的最大seq_len，所以目前它们在维度1上不一致，要加回0
        batch_size = s_output.shape[0]
        s_max_len = int(max(s_lengths))
        o_max_len = int(max(o_lengths))
        if s_max_len < o_max_len:
            s_output = torch.cat((s_output, torch.zeros(batch_size, o_max_len - s_max_len, self.config.lstm1_hidden_dim).to(self.config.device)), 1)
        elif s_max_len > o_max_len:
            o_output = torch.cat((o_output, torch.zeros(batch_size, s_max_len - o_max_len, self.config.lstm1_hidden_dim).to(self.config.device)), 1)

        return {'s_encoded': s_output, 'o_encoded': o_output, 's_masks': s_masks, 'o_masks': o_masks}


# Layer 2: Local Inference Modeling
def local_inference(inputs):
    """
    inputs['p_local_inference']: batch_size * seq_len * (8*hidden_dim)
    """
    def local_inference_enhancement(a, a_weighted):
        ma = torch.cat([a, a_weighted, a - a_weighted, a * a_weighted], -1)
        return ma

    a = inputs['s_encoded']  # batch_size * seq_len * hidden_dim
    b = inputs['o_encoded']  # batch_size * seq_len * hidden_dim
    mask_a = inputs['s_masks']
    mask_b = inputs['o_masks']
    mask_a = mask_a.masked_fill((1 - mask_a).bool(), -1e15)
    mask_b = mask_b.masked_fill((1 - mask_b).bool(), -1e15)

    e = torch.matmul(a, b.transpose(1, 2))  # batch_size * seq_len * seq_len
    weight_a = F.softmax(e + mask_b.unsqueeze(1), dim=-1)
    weight_b = F.softmax(e.transpose(1, 2) + mask_a.unsqueeze(1), dim=-1)

    a_weighted = torch.matmul(weight_a, b)
    b_weighted = torch.matmul(weight_b, a)
    s_local_inference = local_inference_enhancement(a, a_weighted)  # batch_size * seq_len * (4*hidden_dim)
    o_local_inference = local_inference_enhancement(b, b_weighted)  # batch_size * seq_len * (4*hidden_dim)
    return {'s_local_inference': s_local_inference, 'o_local_inference': o_local_inference,
            's_masks': inputs['s_masks'], 'o_masks': inputs['o_masks']}


# Layer 3: Inference Composition
class InferenceComposition(nn.Module):
    def __init__(self, config):
        super(InferenceComposition, self).__init__()
        self.config = config
        self.params = {'ptm': [], 'other': []}

        self.bilstm_layer = BiLSTM(4*config.lstm1_hidden_dim, config.lstm2_hidden_dim, config.lstm2_layer_num, config.dropout)
        self.linear1 = nn.Linear(4 * config.lstm2_hidden_dim, config.num_tags)
        self.bn1 = nn.BatchNorm1d(config.num_tags)
        self.dropout = nn.Dropout(config.dropout)

        self.params['other'].extend([p for p in self.bilstm_layer.parameters()])
        self.params['other'].extend([p for p in self.linear1.parameters()])

    def get_params(self):
        return self.params

    def pooling(self, va):  # batch_size * seq_len * (2*hidden_dim)
        length = va.shape[1]
        va_ave = F.avg_pool1d(va.transpose(1, 2), kernel_size=length).squeeze(-1)  # batch_size * (2*hidden_dim)
        va_max = F.max_pool1d(va.transpose(1, 2), kernel_size=length).squeeze(-1)  # batch_size * (2*hidden_dim)
        return va_ave, va_max

    def forward(self, inputs):
        ma = inputs['s_local_inference']  # batch_size * seq_len * (4*hidden_dim)
        mb = inputs['o_local_inference']  # batch_size * seq_len * (4*hidden_dim)
        mask_a = inputs['s_masks']
        mask_b = inputs['o_masks']

        a_lengths = mask_a.sum(dim=1).numpy().tolist() if not self.config.use_cuda else mask_a.sum(dim=1).cpu().numpy().tolist()
        b_lengths = mask_b.sum(dim=1).numpy().tolist() if not self.config.use_cuda else mask_b.sum(dim=1).cpu().numpy().tolist()
        s_packed = nn.utils.rnn.pack_padded_sequence(ma, a_lengths, batch_first=True, enforce_sorted=False)
        o_packed = nn.utils.rnn.pack_padded_sequence(mb, b_lengths, batch_first=True, enforce_sorted=False)
        s_packed_output = self.bilstm_layer(s_packed)
        o_packed_output = self.bilstm_layer(o_packed)
        va, _ = nn.utils.rnn.pad_packed_sequence(s_packed_output, batch_first=True)
        # batch_size, max_s_len, hidden_dim
        vb, _ = nn.utils.rnn.pad_packed_sequence(o_packed_output, batch_first=True)
        # batch_size, max_o_len, hidden_dim

        va_ave, va_max = self.pooling(va)
        vb_ave, vb_max = self.pooling(vb)
        v = torch.cat([va_ave, va_max, vb_ave, vb_max], -1)  # batch_size * (4*hidden_dim)

        output = torch.tanh(self.dropout(self.bn1(self.linear1(v))))
        return output


# Bert-ESIM Model Overview
class BertESIM(nn.Module):
    def __init__(self, config):
        super(BertESIM, self).__init__()
        self.config = config
        self.params = {}
        self.layer_list = []

        self.input_encoding = InputEncoding(config)
        self.layer_list.append(self.input_encoding)
        self.inference_composition = InferenceComposition(config)
        self.layer_list.append(self.inference_composition)

        self.softmax = nn.Softmax(dim=1)

    def get_params(self):
        if not self.params:
            for layer in self.layer_list:
                for key, value in layer.get_params().items():
                    if key not in self.params:
                        self.params[key] = []
                    self.params[key].extend(value)
        return self.params

    def forward(self, inputs):
        encode_output = self.input_encoding(inputs)
        local_inference_output = local_inference(encode_output)
        output = self.inference_composition(local_inference_output)
        y_hat = self.softmax(output)
        return y_hat


# %% Bert-LSTM
class BertLSTM(nn.Module):
    """
    batch_size * seq_len * embed_dim => batch_size * seq_len * hidden_dim
    """
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.config = config
        self.params = {}
        self.layer_list = []

        self.embedding_layer = Bert(config.ptm_model)
        self.layer_list.append(self.embedding_layer)

        self.bilstm_layer = BiLSTM(config.embedding_dim, config.lstm1_hidden_dim, config.lstm1_layer_num, config.dropout)
        self.layer_list.append(self.bilstm_layer)

        self.linear = nn.Linear(config.lstm1_hidden_dim * 4, self.config.num_tags)

        self.softmax = nn.Softmax(dim=1)

    def get_params(self):
        if not self.params:
            for layer in self.layer_list:
                for key, value in layer.get_params().items():
                    if key not in self.params:
                        self.params[key] = []
                    self.params[key].extend(value)
            self.params['other'].extend([p for p in self.linear.parameters()])
        return self.params

    def pooling(self, va):  # batch_size * seq_len * (2*hidden_dim)
        length = va.shape[1]
        va_ave = F.avg_pool1d(va.transpose(1, 2), kernel_size=length).squeeze(-1)  # batch_size * (2*hidden_dim)
        va_max = F.max_pool1d(va.transpose(1, 2), kernel_size=length).squeeze(-1)  # batch_size * (2*hidden_dim)
        return va_ave, va_max

    def forward(self, inputs):
        def get_so_embedded(text_embedded, so_pairs):
            max_len = 0
            batch_size = len(so_pairs)
            s_idxs = []
            o_idxs = []
            for i in range(batch_size):
                s_idx = [idx for idx in range(so_pairs[i][0][0], so_pairs[i][0][1] + 1)]
                o_idx = [idx for idx in range(so_pairs[i][1][0], so_pairs[i][1][1] + 1)]
                s_idxs.append(s_idx)
                o_idxs.append(o_idx)
                max_len = max(max_len, so_pairs[i][0][1] - so_pairs[i][0][0] + 1)
                max_len = max(max_len, so_pairs[i][1][1] - so_pairs[i][1][0] + 1)
            s_embedded = torch.empty((0, max_len, self.config.embedding_dim)).to(self.config.device)
            o_embedded = torch.empty((0, max_len, self.config.embedding_dim)).to(self.config.device)
            s_masks = torch.empty((0, max_len)).to(self.config.device)
            o_masks = torch.empty((0, max_len)).to(self.config.device)
            for i in range(batch_size):
                s_idx = s_idxs[i]
                o_idx = o_idxs[i]
                s = torch.index_select(text_embedded[i], 0, torch.tensor(s_idx).to(self.config.device))
                o = torch.index_select(text_embedded[i], 0, torch.tensor(o_idx).to(self.config.device))
                s = torch.cat((s, torch.zeros(max_len - len(s_idx), self.config.embedding_dim).to(self.config.device)), 0).unsqueeze(0)
                o = torch.cat((o, torch.zeros(max_len - len(o_idx), self.config.embedding_dim).to(self.config.device)), 0).unsqueeze(0)
                s_mask = torch.cat((torch.ones(len(s_idx)), torch.zeros(max_len - len(s_idx))), 0).unsqueeze(0).to(self.config.device)
                o_mask = torch.cat((torch.ones(len(o_idx)), torch.zeros(max_len - len(o_idx))), 0).unsqueeze(0).to(self.config.device)
                s_embedded = torch.cat((s_embedded, s), 0)
                o_embedded = torch.cat((o_embedded, o), 0)
                s_masks = torch.cat((s_masks, s_mask), 0)
                o_masks = torch.cat((o_masks, o_mask), 0)
            return s_embedded, o_embedded, s_masks, o_masks

        text_embedded = self.embedding_layer(inputs)  # batch_size, seq_len, embed_dim

        # batch_size, max_so_len, embed_dim
        s_embedded, o_embedded, s_masks, o_masks = get_so_embedded(text_embedded, inputs['so_pairs'])

        # packed bilstm
        s_lengths = s_masks.sum(dim=1).numpy().tolist() if not self.config.use_cuda else s_masks.sum(dim=1).cpu().numpy().tolist()
        o_lengths = o_masks.sum(dim=1).numpy().tolist() if not self.config.use_cuda else o_masks.sum(dim=1).cpu().numpy().tolist()
        s_packed = nn.utils.rnn.pack_padded_sequence(s_embedded, s_lengths, batch_first=True, enforce_sorted=False)
        o_packed = nn.utils.rnn.pack_padded_sequence(o_embedded, o_lengths, batch_first=True, enforce_sorted=False)
        s_packed_output = self.bilstm_layer(s_packed)
        o_packed_output = self.bilstm_layer(o_packed)
        s_output, _ = nn.utils.rnn.pad_packed_sequence(s_packed_output, batch_first=True)
        # batch_size, max_s_len, hidden_dim
        o_output, _ = nn.utils.rnn.pad_packed_sequence(o_packed_output, batch_first=True)
        # batch_size, max_o_len, hidden_dim

        s_ave, s_max = self.pooling(s_output)
        o_ave, o_max = self.pooling(o_output)
        v = torch.cat([s_ave, s_max, o_ave, o_max], -1)  # batch_size * (4*hidden_dim)

        output = self.linear(v)

        y_hat = self.softmax(output)
        return y_hat


# %% Simple-Bert-ESIM
# Layer 1: Simple Input Encoding
class SimpleInputEncoding(nn.Module):
    """
    batch_size * seq_len * embed_dim => batch_size * seq_len * hidden_dim
    """
    def __init__(self, config):
        super(SimpleInputEncoding, self).__init__()
        self.config = config
        self.params = {}
        self.layer_list = []

        self.embedding_layer = Bert(config.ptm_model)
        self.layer_list.append(self.embedding_layer)

    def get_params(self):
        if not self.params:
            for layer in self.layer_list:
                for key, value in layer.get_params().items():
                    if key not in self.params:
                        self.params[key] = []
                    self.params[key].extend(value)
        return self.params

    def forward(self, inputs):
        def get_so_embedded(text_embedded, so_pairs):
            max_len = 0
            batch_size = len(so_pairs)
            s_idxs = []
            o_idxs = []
            for i in range(batch_size):
                s_idx = [idx for idx in range(so_pairs[i][0][0], so_pairs[i][0][1] + 1)]
                o_idx = [idx for idx in range(so_pairs[i][1][0], so_pairs[i][1][1] + 1)]
                s_idxs.append(s_idx)
                o_idxs.append(o_idx)
                max_len = max(max_len, so_pairs[i][0][1] - so_pairs[i][0][0] + 1)
                max_len = max(max_len, so_pairs[i][1][1] - so_pairs[i][1][0] + 1)
            s_embedded = torch.empty((0, max_len, self.config.embedding_dim)).to(self.config.device)
            o_embedded = torch.empty((0, max_len, self.config.embedding_dim)).to(self.config.device)
            s_masks = torch.empty((0, max_len)).to(self.config.device)
            o_masks = torch.empty((0, max_len)).to(self.config.device)
            for i in range(batch_size):
                s_idx = s_idxs[i]
                o_idx = o_idxs[i]
                s = torch.index_select(text_embedded[i], 0, torch.tensor(s_idx).to(self.config.device))
                o = torch.index_select(text_embedded[i], 0, torch.tensor(o_idx).to(self.config.device))
                s = torch.cat((s, torch.zeros(max_len - len(s_idx), self.config.embedding_dim).to(self.config.device)), 0).unsqueeze(0)
                o = torch.cat((o, torch.zeros(max_len - len(o_idx), self.config.embedding_dim).to(self.config.device)), 0).unsqueeze(0)
                s_mask = torch.cat((torch.ones(len(s_idx)), torch.zeros(max_len - len(s_idx))), 0).unsqueeze(0).to(self.config.device)
                o_mask = torch.cat((torch.ones(len(o_idx)), torch.zeros(max_len - len(o_idx))), 0).unsqueeze(0).to(self.config.device)
                s_embedded = torch.cat((s_embedded, s), 0)
                o_embedded = torch.cat((o_embedded, o), 0)
                s_masks = torch.cat((s_masks, s_mask), 0)
                o_masks = torch.cat((o_masks, o_mask), 0)
            return s_embedded, o_embedded, s_masks, o_masks

        text_embedded = self.embedding_layer(inputs)  # batch_size, seq_len, embed_dim

        # batch_size, max_so_len, embed_dim
        s_embedded, o_embedded, s_masks, o_masks = get_so_embedded(text_embedded, inputs['so_pairs'])

        return {'s_encoded': s_embedded, 'o_encoded': o_embedded, 's_masks': s_masks, 'o_masks': o_masks}


# Layer 3: Simple Inference Composition
class SimpleInferenceComposition(nn.Module):
    def __init__(self, config):
        super(SimpleInferenceComposition, self).__init__()
        self.config = config
        self.params = {'ptm': [], 'other': []}

        self.bilstm_layer = BiLSTM(4*config.embedding_dim, config.lstm2_hidden_dim, config.lstm2_layer_num, config.dropout)
        self.linear = nn.Linear(4 * config.lstm2_hidden_dim, config.num_tags)

        self.params['other'].extend([p for p in self.bilstm_layer.parameters()])
        self.params['other'].extend([p for p in self.linear.parameters()])

    def get_params(self):
        return self.params

    def pooling(self, va):  # batch_size * seq_len * (2*hidden_dim)
        length = va.shape[1]
        va_ave = F.avg_pool1d(va.transpose(1, 2), kernel_size=length).squeeze(-1)  # batch_size * (2*hidden_dim)
        va_max = F.max_pool1d(va.transpose(1, 2), kernel_size=length).squeeze(-1)  # batch_size * (2*hidden_dim)
        return va_ave, va_max

    def forward(self, inputs):
        ma = inputs['s_local_inference']  # batch_size * seq_len * (4*hidden_dim)
        mb = inputs['o_local_inference']  # batch_size * seq_len * (4*hidden_dim)
        mask_a = inputs['s_masks']
        mask_b = inputs['o_masks']

        a_lengths = mask_a.sum(dim=1).numpy().tolist() if not self.config.use_cuda else mask_a.sum(dim=1).cpu().numpy().tolist()
        b_lengths = mask_b.sum(dim=1).numpy().tolist() if not self.config.use_cuda else mask_b.sum(dim=1).cpu().numpy().tolist()
        s_packed = nn.utils.rnn.pack_padded_sequence(ma, a_lengths, batch_first=True, enforce_sorted=False)
        o_packed = nn.utils.rnn.pack_padded_sequence(mb, b_lengths, batch_first=True, enforce_sorted=False)
        s_packed_output = self.bilstm_layer(s_packed)
        o_packed_output = self.bilstm_layer(o_packed)
        va, _ = nn.utils.rnn.pad_packed_sequence(s_packed_output, batch_first=True)
        # batch_size, max_s_len, hidden_dim
        vb, _ = nn.utils.rnn.pad_packed_sequence(o_packed_output, batch_first=True)
        # batch_size, max_o_len, hidden_dim

        va_ave, va_max = self.pooling(va)
        vb_ave, vb_max = self.pooling(vb)
        v = torch.cat([va_ave, va_max, vb_ave, vb_max], -1)  # batch_size * (4*hidden_dim)

        output = self.linear(v)
        return output


class SimpleBertESIM(nn.Module):
    def __init__(self, config):
        super(SimpleBertESIM, self).__init__()
        self.config = config
        self.params = {}
        self.layer_list = []

        self.input_encoding = SimpleInputEncoding(config)
        self.layer_list.append(self.input_encoding)
        self.inference_composition = SimpleInferenceComposition(config)
        self.layer_list.append(self.inference_composition)

        self.softmax = nn.Softmax(dim=1)

    def get_params(self):
        if not self.params:
            for layer in self.layer_list:
                for key, value in layer.get_params().items():
                    if key not in self.params:
                        self.params[key] = []
                    self.params[key].extend(value)
        return self.params

    def forward(self, inputs):
        encode_output = self.input_encoding(inputs)
        local_inference_output = local_inference(encode_output)
        output = self.inference_composition(local_inference_output)
        y_hat = self.softmax(output)
        return y_hat
