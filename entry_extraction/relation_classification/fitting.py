import sys
sys.path.append('/remote-home/aqshi/NLP_Policy/NLP_Policy')
sys.path.append('/remote-home/aqshi/NLP_Policy')

import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn import metrics


class FittingConfig:
    def __init__(self, unique):
        self.use_cuda = False
        self.batch_size = 10
        self.epochs = 100
        self.lr = {'ptm': 0.00003, 'others': 0.001}
        self.early_stop = True
        self.patience = 8
        self.decay = 0.95
        self.num_tags = None

        # fixed
        self.result_data_path = os.path.join(os.getcwd(), 'result/')
        if not os.path.exists(self.result_data_path):
            os.makedirs(self.result_data_path)
        self.result_model_path = os.path.join(self.result_data_path, 'best_model_{}.pt'.format(unique))
        self.result_acc_path = os.path.join(self.result_data_path, 'acc_result_{}.json'.format(unique))
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

        self.label2idx, self.idx2label = None, None
        self.model = None

        # plot
        self.acc_result = {'dev_p': [], 'dev_r': [], 'dev_f1': [], 'dev_loss': []}

    def collate_fn_test(self, batch):
        # 暂未考虑处理词向量信息，先做character-level
        text = []
        text_mask = []
        so_pair = []
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
            so_pair.append((sample['spo']['s'], sample['spo']['o']))

        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask).float()
        result = {'sentence_tokens': [text, True], 'sentence_masks': [text_mask, True], 'so_pairs': [so_pair, False]}
        if self.use_cuda:
            result = dict([(key, value[0].cuda() if value[1] else value[0]) for key, value in result.items()])
        else:
            result = dict([(key, value[0]) for key, value in result.items()])
        return result

    def collate_fn_train(self, batch):
        inputs = self.collate_fn_test(batch)
        y_true = []
        for sample in batch:
            y_true.append(self.label2idx[sample['spo']['p']])
        y_true = torch.tensor(y_true).long()
        if self.use_cuda:
            y_true = y_true.cuda()
        return inputs, y_true

    def get_collate_fn(self, mode='train'):
        if mode == 'train' or mode == 'dev':
            return self.collate_fn_train
        elif mode == 'test':
            return self.collate_fn_test

    def save_model(self, model):
        print('==================================saving best model...==================================')
        torch.save(model.state_dict(), self.config.result_model_path)

    def _plotting_data(self, dev_p, dev_r, dev_f1, dev_loss):
        self.acc_result['dev_p'].append(dev_p)
        self.acc_result['dev_r'].append(dev_r)
        self.acc_result['dev_f1'].append(dev_f1)
        self.acc_result['dev_loss'].append(dev_loss)

    def save_plotting_data(self):
        with open(self.config.result_acc_path, 'w') as outfile:
            json.dump(self.acc_result, outfile)

    def plot_acc(self):
        epochs = np.arange(0, len(self.acc_result['dev_f1']))
        plt.plot(epochs, self.acc_result['dev_p'], 'steelblue', label='dev_p')
        plt.plot(epochs, self.acc_result['dev_r'], 'darkolivegreen', label='dev_r')
        plt.plot(epochs, self.acc_result['dev_f1'], 'salmon', label='dev_f1')
        plt.plot(epochs, self.acc_result['dev_loss'], 'darkorange', label='dev_loss')
        plt.legend()
        plt.savefig(self.config.result_pic_path)
        plt.show()
        self.save_plotting_data()

    def decode(self, y_hat):
        return y_hat.argmax(dim=1)

    def cal_loss(self, y_hat, y_true):
        loss = nn.CrossEntropyLoss()
        return loss(y_hat, y_true)

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
                if self.use_cuda:
                    torch.cuda.empty_cache()
                optimizer.zero_grad()
                self.model.train()
                y_hat = self.model(inputs)

                loss = self.cal_loss(y_hat, targets)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    preds = self.decode(y_hat)
                    if self.use_cuda:
                        preds = preds.cpu().numpy()
                        targets = targets.cpu().numpy()
                    else:
                        preds = preds.numpy()
                        targets = targets.numpy()
                    acc = metrics.accuracy_score(targets, preds)
                    p = metrics.precision_score(targets, preds, average='micro')
                    r = metrics.recall_score(targets, preds, average='micro')
                    f1 = metrics.f1_score(targets, preds, average='micro')
                    print('epoch: {}, step: {}/{}, loss: {:.6f}, ACC: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(
                        epoch, step, train_steps, loss, acc, p, r, f1))
                print('lr: {:.6f}'.format(optimizer.param_groups[0]['lr']))
            # dev
            with torch.no_grad():
                dev_inputs = {'data': dev_data}
                eval_result = self.eval(dev_inputs)
                if eval_result['loss'] < best_dev_loss:
                    best_dev_loss = eval_result['loss']
                    last_better_epoch = epoch
                    # self.save_model(self.model)  # TODO
                elif self.early_stop:
                    if epoch - last_better_epoch >= self.patience:
                        print('===============================early stopping...===============================')
                        print('best dev loss: ', best_dev_loss)
                        break
            self._plotting_data(eval_result['p'], eval_result['r'], eval_result['f1'], eval_result['loss'].item())
            scheduler.step()
            if self.use_cuda:
                torch.cuda.empty_cache()

    def eval(self, dev_inputs):
        dev_data = dev_inputs['data']
        dev_dataloader = DataLoader(dev_data, batch_size=self.batch_size, collate_fn=self.get_collate_fn('dev'))
        result = {}
        metrics_data = {'loss': 0, 'acc': 0, 'p': 0, 'r': 0, 'f1': 0, 'batch_num': 0}
        with torch.no_grad():
            print('==================================evaluating dev data...==================================')
            for step, (inputs, targets) in enumerate(dev_dataloader):
                if self.use_cuda:
                    torch.cuda.empty_cache()
                self.model.eval()
                y_hat = self.model(inputs)
                loss = self.cal_loss(y_hat, targets)

                preds = self.decode(y_hat)
                if self.use_cuda:
                    preds = preds.cpu().numpy()
                    targets = targets.cpu().numpy()
                else:
                    preds = preds.numpy()
                    targets = targets.numpy()
                acc = metrics.accuracy_score(targets, preds)
                p = metrics.precision_score(targets, preds, average='micro')
                r = metrics.recall_score(targets, preds, average='micro')
                f1 = metrics.f1_score(targets, preds, average='micro')

                metrics_data['loss'] += loss.cpu().float()
                metrics_data['acc'] += acc
                metrics_data['p'] += p
                metrics_data['r'] += r
                metrics_data['f1'] += f1
                metrics_data['batch_num'] += 1
            result['loss'] = metrics_data['loss'] / metrics_data['batch_num']
            result['acc'] = metrics_data['acc'] / metrics_data['batch_num']
            result['p'] = metrics_data['p'] / metrics_data['batch_num']
            result['r'] = metrics_data['r'] / metrics_data['batch_num']
            result['f1'] = metrics_data['f1'] / metrics_data['batch_num']
            print('loss: {:.6f}, acc: {:.4f}, p: {:.4f}, r: {:.4f}, f1: {:.4f}'.format(result['loss'], result['acc'], result['p'], result['r'], result['f1']))
        return result

    def test(self, test_inputs):
        print('==================================evaluating test data...==================================')
        model = test_inputs['model']
        self.label2idx, self.idx2label = test_inputs['label2idx'], test_inputs['idx2label']
        test_data = test_inputs['test_data']
        model.load_state_dict(torch.load(self.config.result_model_path))
        if self.use_cuda:
            model = model.cuda()
        self.model = model
        inputs = {'data': test_data}
        result = self.eval(inputs)
        return result
