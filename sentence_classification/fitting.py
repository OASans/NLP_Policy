import os
import pandas as pd
import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
import torch.nn as nn

from NLP_Policy.data_process.word2vec import get_w2v_vector
from models import ModelE, XGBoost, SVM, TextCNN, BiLSTM_Attention


class FittingConfig:
    def __init__(self, unique):
        self.result_data_path = os.path.join(os.getcwd(), 'result/')
        if not os.path.exists(self.result_data_path):
            os.makedirs(self.result_data_path)
        self.result_model_path = os.path.join(self.result_data_path, 'best_model_{}.pt'.format(unique))
        self.result_data_path = os.path.join(self.result_data_path, 'acc_result_{}.json'.format(unique))
        self.result_pic_path = os.path.join(self.result_data_path, 'acc_pic_{}.png'.format(unique))

        self.w2v = True
        self.merge_label = True  # 合并21类标签为8类
        self.get_label_num = False  # 获取数据集标签数量
        self.batch_size = 64
        self.epochs = 20


class ModelFitting:
    def __init__(self, config, model_config):
        self.config = config
        self.model_config = model_config
        # self.w2v_array = get_w2v_vector() if self.config.w2v else None
        if self.config.w2v:
            self.w2v_array = get_w2v_vector()
            print("get_w2v_vector finished!")
        self.label2idx, self.idx2label = None, None
        self.model = None
        self.early_stop = False
        self.patience = 3

    def _get_X_y(self, data):
        X = np.zeros(shape=(data.shape[0], 302)).astype(float)
        for i in range(data.shape[0]):
            sentence_vec = np.array([0.0] * 300)
            index_in_text = float(data[i]['sid'].split('_')[-1])
            sentence_num = float(data[i]['sentence_num'])
            token_num = float(len(data[i]['lattice_token']))
            for token in data[i]['lattice_token']:
                sentence_vec = sentence_vec + np.array(self.w2v_array[token])
            sentence_vec = sentence_vec / token_num if token_num != 0.0 else sentence_vec
            sentence_vec = np.append(sentence_vec, index_in_text / sentence_num)
            sentence_vec = np.append(sentence_vec, token_num)
            X[i] = sentence_vec

        data = pd.DataFrame(data.tolist())
        y = np.array([self.label2idx[label] for label in data['sentence_type'].values])
        return X, y

    def get_l_num(self, train_data, dev_data, test_data):
        _, train_label = self._get_X_y(train_data)
        _, dev_label = self._get_X_y(dev_data)
        _, test_label = self._get_X_y(test_data)
        label_count = np.zeros(len(self.label2idx))
        for i, set in enumerate([train_label, dev_label, test_label]):
            print(str(i) + ":\n")
            for item in set:
                label_count[item] += 1
            result = {}
            for i in range(len(label_count)):
                label_name = self.idx2label[str(i)]
                result[label_name] = int(label_count[i])
            print(result)
            for key, value in result.items():
                print(key + "：" + str(value))
        exit(0)

    def train(self, train_inputs):
        model = train_inputs['model']
        train_data = train_inputs['train_data']
        dev_data = train_inputs['dev_data']
        test_data = train_inputs['test_data']
        if not self.config.merge_label:
            self.label2idx, self.idx2label = train_inputs['label2idx'], train_inputs['idx2label']
        else:
            self.label2idx = {"":0, '政策目标':1, '申请审核程序':2, '资金管理-资金来源':3, '资金管理-管理原则':3, '监管评估-监督管理':4,
                          '监管评估-考核评估':4, '政策内容-人才培养':5, '政策内容-资金支持':5, '政策内容-技术支持':5,
                          '政策内容-公共服务':5, '政策内容-组织建设':6, '政策内容-目标规划':6, '政策内容-法规管制':6,
                          '政策内容-政策宣传':6, '政策内容-税收优惠':6, '政策内容-金融支持':6, '政策内容-政府采购':7,
                          '政策内容-对外承包':7, '政策内容-公私合作':7, '政策内容-海外合作':7}
            self.idx2label = {0:'', 1:'政策目标', 2:'申请审核程序', 3:'资金管理', 4:'监管评估', 5:'政策内容-供给型',
                          6:'政策内容-环境型', 7:'政策内容-需求型'}

        if self.config.get_label_num:
            self.get_l_num(train_data, dev_data, test_data)

        if model in [ModelE.xgboost, ModelE.SVM]:
            train_X, train_y = self._get_X_y(train_data)
            dev_X, dev_y = self._get_X_y(dev_data)
            test_X, test_y = self._get_X_y(test_data)

            if model is ModelE.xgboost:
                train_X = np.vstack((train_X, dev_X))
                train_y = np.hstack((train_y, dev_y))
                self.model = XGBoost(model_config=self.model_config)
            elif model is ModelE.SVM:
                self.model = SVM()

            clf = self.model(train_X, train_y)
            print('training finished!')

            test_pred = clf.predict(test_X)
            print('accuracy_score: ', metrics.accuracy_score(test_y, test_pred))
            print('f1_score: ', metrics.f1_score(test_y, test_pred, average='micro'))
            print('recall_score:', metrics.recall_score(test_y, test_pred, average='micro'))
            print('precision_score:', metrics.precision_score(test_y, test_pred, average='micro'))
            print('confusion_metrics:', metrics.confusion_matrix(test_y, test_pred))
            print('predict finished!')

        if model in [ModelE.TextCNN, ModelE.BiLSTM_Attention]:
            args = []
            if model is ModelE.TextCNN:
                self.model = TextCNN(args)
            elif model is ModelE.BiLSTM_Attention:
                self.model = BiLSTM_Attention(embedding_dim=300, num_hiddens=64, num_layers=1)
            train_steps = int((len(train_data) + self.config.batch_size - 1) / self.config.batch_size)
            self.max_lattice_num = np.max([len(each["lattice_token"]) for each in train_data])
            train_dataloader = DataLoader(train_data, batch_size=self.config.batch_size, collate_fn=self.collate_fn,
                                          shuffle=True)  # 不自己定义collate_fn则会报错每个batch不一样长
            optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-4)
            loss_func = nn.CrossEntropyLoss()

            best_dev_loss = 99999
            last_better_epoch = 0
            for epoch in range(self.config.epochs):
                for step, (inputs, targets) in enumerate(train_dataloader):  # error
                    optimizer.zero_grad()
                    self.model.train()
                    output = self.model(inputs)  # inputs需要是float类型
                    loss = loss_func(output, targets)  # targets需要是一维的long类型
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        _, pred_label = torch.max(output, 1)  # 获取预测标签
                        test_accuracy = metrics.accuracy_score(targets, pred_label)
                        P = metrics.precision_score(targets, pred_label, average='micro')
                        R = metrics.recall_score(targets, pred_label, average='micro')
                        F1 = metrics.f1_score(targets, pred_label, average='micro')
                        print('epoch: {}, step: {}/{}, loss: {:.6f}, acc: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(
                            epoch, step + 1, train_steps, loss, test_accuracy, P, R, F1
                        ))
                # dev
                with torch.no_grad():
                    eval_result = self.eval(dev_data)
                    if eval_result['loss'] < best_dev_loss:
                        best_dev_loss = eval_result['loss']
                        last_better_epoch = epoch
                        # self.save_model(self.model)
                    elif self.early_stop:
                        if epoch - last_better_epoch >= self.patience:
                            print('===============================early stopping...===============================')
                            print('best dev loss: ', best_dev_loss)
                            break
            print("train finished!")

            # test
            with torch.no_grad():
                print('============================testing...=====================================')
                test_result = self.eval(test_data)
            print("test finished!")

    def eval(self, dev_data):
        dev_dataloader = DataLoader(dev_data, batch_size=64, collate_fn=self.collate_fn, shuffle=True)
        result = {}
        metrics_data = {'loss': 0, 'p': 0, 'r': 0, 'f1': 0, 'batch_num': 0}
        loss_func = nn.CrossEntropyLoss()

        with torch.no_grad():
            print('==================================evaluating dev data...==================================')
            for step, (inputs, targets) in enumerate(dev_dataloader):
                self.model.eval()
                output = self.model(inputs)
                loss = loss_func(output, targets)  # 不能直接用CrossEntropyLoss，这是一个类，并且output和target位置不能反
                _, pred_label = torch.max(output, 1)
                test_accuracy = metrics.accuracy_score(targets, pred_label)
                P = metrics.precision_score(targets, pred_label, average='micro')
                R = metrics.recall_score(targets, pred_label, average='micro')
                F1 = metrics.f1_score(targets, pred_label, average='micro')
                metrics_data['loss'] += loss.cpu().float()
                metrics_data['p'] += P
                metrics_data['r'] += R
                metrics_data['f1'] += F1
                metrics_data['batch_num'] += 1
            result['loss'] = metrics_data['loss'] / metrics_data['batch_num']
            result['p'] = metrics_data['p'] / metrics_data['batch_num']
            result['r'] = metrics_data['r'] / metrics_data['batch_num']
            result['f1'] = metrics_data['f1'] / metrics_data['batch_num']
            print('loss: {:.6f}, acc: {:.4f}, p: {:.4f}, r: {:.4f}, f1: {:.4f}'.format(result['loss'], test_accuracy, result['p'], result['r'],
                                                                          result['f1']))
            print('confusion_metrics:\n', metrics.confusion_matrix(targets, pred_label))
        return result

    # Todo
    def test(self, test_data):
        print('==================================evaluating test data...==================================')
        model = TextCNN
        model.load_state_dict(torch.load(self.config.result_model_path))
        # if self.use_cuda:
        #     model = model.cuda()
        self.model = model
        result = self.eval(test_data)
        return result

    def collate_fn(self, data):
        batch_size = np.array(data).shape[0]
        X = np.zeros(shape=(batch_size, self.max_lattice_num, 300), dtype=np.float64)#.astype(float)
        for i in range(batch_size):
            for j, token in enumerate(data[i]['lattice_token']):
                X[i][j] += np.array(self.w2v_array[token])

        data = pd.DataFrame(np.array(data).tolist())
        y = np.array([self.label2idx[label] for label in data['sentence_type'].values])
        return torch.FloatTensor(X), torch.Tensor(y).long()

    def save_model(self, model):
        print('==================================saving best model...==================================')
        torch.save(model.state_dict(), self.config.result_model_path)
