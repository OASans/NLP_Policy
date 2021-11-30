from enum import Enum
from xgboost import XGBClassifier
from sklearn.svm import SVC
import torch.nn as nn
import torch
import torch.nn.functional as F


ModelE = Enum('ModelE', ('xgboost', 'lgbm', 'adaboost', 'SVM', 'TextCNN', 'BiLSTM_Attention'))


class ModelConfig:
    def __init__(self):
        # xgboost
        self.lr = 0.01
        self.n_estimators = 100
        self.objective = 'multi:softmax'
        self.max_depth = 8


class XGBoost:
    def __init__(self, model_config):
        self.model = XGBClassifier(learning_rate=model_config.lr, n_estimators=model_config.n_estimators,
                                   objective=model_config.objective, max_depth=model_config.max_depth)

    def __call__(self, X, y):
        self.model.fit(X, y)
        return self.model


class SVM:
    def __init__(self):
        self.model = SVC()

    def __call__(self, X, y):
        self.model.fit(X, y)
        return self.model


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        class_num = 21
        channel_num = 1
        filter_sizes = [3, 4, 5]
        filter_num = 100

        # vocabulary_size = args.vocabulary_size
        embedding_dimension = 300

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.ReLU = nn.ReLU()
        self.max_pool1d = nn.MaxPool1d(kernel_size=200)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [conv(x) for conv in self.convs]
        x = [self.ReLU(item).squeeze(3) for item in x]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        output = self.fc(x)
        return output


class BiLSTM_Attention(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_layers):
        super(BiLSTM_Attention, self).__init__()
        # embedding之后的shape: torch.Size([200, 8, 300])
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings = self.word_embeddings.from_pretrained(
        #     vectors, freeze=False)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embedding_dim,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               batch_first=False,
                               bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(
            num_hiddens * 2, num_hiddens * 2))
        self.u_omega = nn.Parameter(torch.Tensor(num_hiddens * 2, 1))
        self.decoder = nn.Linear(2 * num_hiddens, 21)

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        # inputs的形状是(seq_len,batch_size)
        # embeddings = self.word_embeddings(inputs)
        # 提取词特征，输出形状为(seq_len,batch_size,embedding_dim)
        # rnn.LSTM只返回最后一层的隐藏层在各时间步的隐藏状态。
        inputs = inputs.permute(1, 0, 2)
        outputs, _ = self.encoder(inputs)  # output, (h, c)
        # outputs形状是(seq_len,batch_size, 2 * num_hiddens)
        x = outputs.permute(1, 0, 2)
        # x形状是(batch_size, seq_len, 2 * num_hiddens)

        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束

        feat = torch.sum(scored_x, dim=1)  # 加权求和
        # feat形状是(batch_size, 2 * num_hiddens)
        outs = self.decoder(feat)
        # out形状是(batch_size, 21)
        return outs
