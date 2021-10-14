import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from NLP_Policy.data_process.word2vec import get_w2v_vector


class FittingConfig:
    def __init__(self, unique):
        self.use_cuda = False
        self.w2v = False
        self.batch_size = 16
        self.epochs = 100
        self.lr = {'ptm': 0.00003, 'crf': 0.005, 'others': 0.0001}
        self.early_stop = True
        self.patience = 8
        self.decay = 0.95
        self.num_tags = None

        # fixed
        self.result_data_path = os.path.join(os.getcwd(), 'result/')
        if not os.path.exists(self.result_data_path):
            os.makedirs(self.result_data_path)
        self.result_model_path = os.path.join(self.result_data_path, 'best_model_{}.pt'.format(unique))
        self.result_data_path = os.path.join(self.result_data_path, 'acc_result_{}.json'.format(unique))
        self.result_pic_path = os.path.join(self.result_data_path, 'acc_pic_{}.png'.format(unique))


class ModelFitting:
    def __init__(self, config):
        self.config = config

        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.lr = config.lr
        self.use_cuda = config.use_cuda
        self.early_stop = config.early_stop
        self.patience = config.patience
        self.decay = config.decay

        self.w2v_array = get_w2v_vector() if config.w2v else None
        self.label2idx, self.idx2label = None, None
        self.model = None

        # plot
        self.acc_result = {'dev_p': [], 'dev_r': [], 'dev_f1': [], 'dev_loss': []}

    def collate_fn_test(self, batch):
        # 暂未考虑处理词向量信息，先做character-level
        text = []
        text_mask = []
        max_len = 0
        min_len = 9999
        for sample in batch:
            sample_len = len(sample['sentence_tokens'])
            max_len = max_len if max_len > sample_len else sample_len
            min_len = min_len if min_len < sample_len else sample_len
        for sample in batch:
            text_length = len(sample['sentence_tokens'])
            text.append(sample['sentence_tokens'] + [0] * (max_len - text_length))
            text_mask.append([1] * text_length + [0] * (max_len - text_length))

        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask).float()
        result = {'sentence_tokens': [text, True], 'sentence_masks': [text_mask, True]}
        if self.use_cuda:
            result = dict([(key, value[0].cuda() if value[1] else value[0]) for key, value in result.items()])
        else:
            result = dict([(key, value[0]) for key, value in result.items()])
        return result

    def collate_fn_train(self, batch):
        def _get_bioes(sentence_len, entity_list):
            bioes_label = ['O'] * sentence_len
            for entity in entity_list:
                if entity[0] == entity[1]:
                    bioes_label[entity[0]] = 'S-{}'.format(entity[2])
                    continue
                bioes_label[entity[0]] = 'B-{}'.format(entity[2])
                bioes_label[entity[1]] = 'E-{}'.format(entity[2])
                for i in range(entity[0] + 1, entity[1]):
                    bioes_label[i] = 'I-{}'.format(entity[2])
            bioes_label = [self.label2idx[label] for label in bioes_label]
            return bioes_label

        inputs = self.collate_fn_test(batch)
        y_true = []
        max_len = inputs['sentence_tokens'].shape[1]
        for sample in batch:
            text_length = len(sample['sentence_tokens'])
            bioes_label = _get_bioes(text_length, sample['entity_list'])
            bioes_label = bioes_label + [0] * (max_len - text_length)
            y_true.append(bioes_label)
        y_true = torch.tensor(y_true).long()
        if self.use_cuda:
            y_true = y_true.cuda()
        return inputs, y_true

    def get_collate_fn(self, mode='train'):
        if mode == 'train' or mode == 'dev':
            return self.collate_fn_train
        elif mode == 'test':
            return self.collate_fn_test

    def cal_loss(self, y_hat, y_true):
        return self.model.cal_loss(y_hat, y_true)

    def save_model(self, model):
        print('==================================saving best model...==================================')
        torch.save(model.state_dict(), self.config.result_model_path)

    def save_plotting_data(self):
        with open(self.config.result_data_path, 'w') as outfile:
            json.dump(self.config.result_data_path, outfile)

    def _plotting_data(self, dev_loss):
        self.acc_result['dev_loss'].append(dev_loss)

    def plot_acc(self):
        epochs = np.arange(0, len(self.acc_result['dev_loss']))
        plt.plot(epochs, self.acc_result['dev_loss'], 'darkorange', label='dev_loss')
        plt.legend()
        plt.savefig(self.config.result_pic_path)
        plt.show()
        self.save_plotting_data()

    def train(self, train_inputs):
        self.model = train_inputs['model'] if not self.use_cuda else train_inputs['model'].cuda()
        self.label2idx, self.idx2label = train_inputs['label2idx'], train_inputs['idx2label']
        train_data = train_inputs['train_data']
        dev_data = train_inputs['dev_data']

        train_steps = int((len(train_data) + self.batch_size - 1) / self.batch_size)
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, collate_fn=self.get_collate_fn('train'),
                                      shuffle=True)

        params_lr = []
        for key, value in self.model.get_params().items():
            if key in self.lr:
                params_lr.append({"params": value, 'lr': self.lr[key]})
        optimizer = torch.optim.Adam(params_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decay)

        best_dev_loss = 99999
        last_better_epoch = 0
        for epoch in range(self.epochs):
            for step, (inputs, targets) in enumerate(train_dataloader):
                optimizer.zero_grad()
                self.model.train()
                preds = self.model(inputs)
                loss = self.model.cal_loss(preds, targets, inputs['sentence_masks'])
                loss.backward()
                optimizer.step()
                print('epoch: {}, step: {}/{}, loss: {:.6f}, lr: {:.6f}'.format(epoch, step, train_steps, loss,
                                                                                optimizer.param_groups[0]['lr']))
            # dev
            with torch.no_grad():
                dev_inputs = {'model': self.model, 'data': dev_data}
                eval_result = self.eval(dev_inputs)
                if eval_result['loss'] < best_dev_loss:
                    best_dev_loss = eval_result['loss']
                    last_better_epoch = epoch
                    self.save_model(self.model)
                elif self.early_stop:
                    if epoch - last_better_epoch >= self.patience:
                        print('===============================early stopping...===============================')
                        print('best dev loss: ', best_dev_loss)
                        break
            self._plotting_data(eval_result['loss'].item())
            scheduler.step()

    def eval(self, dev_inputs):
        model = dev_inputs['model']
        dev_data = dev_inputs['data']
        dev_dataloader = DataLoader(dev_data, batch_size=self.batch_size, collate_fn=self.get_collate_fn('dev'))
        result = {}
        total_loss = 0
        batch_num = 0
        with torch.no_grad():
            print('==================================evaluating dev data...==================================')
            for step, (inputs, targets) in enumerate(dev_dataloader):
                model.eval()
                y_hat = model(inputs)
                loss = self.cal_loss(y_hat, targets)
                total_loss += loss.float()
                batch_num += 1
            result['loss'] = total_loss / batch_num
            print('loss: {:.6f}'.format(result['loss']))
        return result

    def test(self, test_inputs):
        print('==================================evaluating test data...==================================')
        model = test_inputs['model']
        model.load_state_dict(torch.load(self.config.result_model_path))
        if self.use_cuda:
            model = model.cuda()
        test_inputs['model'] = model
        result = self.eval(test_inputs)
        return result
