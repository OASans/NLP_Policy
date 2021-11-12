import torch
import torch.nn as nn

from module.attn import TransfSelfEncoderRel


class FLAT(nn.Module):
    def __init__(self, config):
        super(FLAT, self).__init__()
        self.config = config
        self.params = {'other': []}

        if self.config.ptm_feat_size != self.config.hidden_size:
            self.adapter = nn.Linear(self.config.ptm_feat_size, self.config.hidden_size)
            self.params['other'].extend([p for p in self.adapter.parameters()])
        self.encoder_layers = []
        for _ in range(self.config.num_flat_layers):
            encoder_layer = TransfSelfEncoderRel(self.config)
            self.encoder_layers.append(encoder_layer)
            self.params['other'].extend([p for p in encoder_layer.parameters()])
        self.encoder_layers = nn.ModuleList(self.encoder_layers)

    def get_params(self):
        return self.params

    def forward(self, inputs):
        char_word_vec = inputs['char_word_vec']
        char_word_mask = inputs['char_word_mask']
        char_word_s = inputs['char_word_s']
        char_word_e = inputs['char_word_e']
        part_size = inputs['part_size']

        pos_emb_layer = inputs['pos_emb_layer']
        if self.config.ptm_feat_size != self.config.hidden_size:
            hidden = self.adapter(char_word_vec)
        else:
            hidden = char_word_vec
        pe_ss = pos_emb_layer(char_word_s.unsqueeze(dim=2) - char_word_s.unsqueeze(dim=1))
        pe_se = pos_emb_layer(char_word_s.unsqueeze(dim=2) - char_word_e.unsqueeze(dim=1))
        pe_es = pos_emb_layer(char_word_e.unsqueeze(dim=2) - char_word_s.unsqueeze(dim=1))
        pe_ee = pos_emb_layer(char_word_e.unsqueeze(dim=2) - char_word_e.unsqueeze(dim=1))
        for layer in self.encoder_layers:
            hidden = layer(hidden, [pe_ss, pe_se, pe_es, pe_ee], char_word_mask)
        char_vec, _ = hidden.split([part_size[0] + part_size[1], part_size[2]], dim=1)
        return {'text_vec': char_vec}