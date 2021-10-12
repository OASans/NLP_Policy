import os
import pandas as pd
import numpy as np
from sklearn import metrics

from NLP_Policy.data_process.word2vec import get_w2v_vector
from models import ModelE, XGBoost


class FittingConfig:
    def __init__(self, unique):
        self.result_data_path = os.path.join(os.getcwd(), 'result/')
        if not os.path.exists(self.result_data_path):
            os.makedirs(self.result_data_path)
        self.result_model_path = os.path.join(self.result_data_path, 'best_model_{}.pt'.format(unique))
        self.result_data_path = os.path.join(self.result_data_path, 'acc_result_{}.json'.format(unique))
        self.result_pic_path = os.path.join(self.result_data_path, 'acc_pic_{}.png'.format(unique))

        self.w2v = True


class ModelFitting:
    def __init__(self, config, model_config):
        self.config = config
        self.model_config = model_config
        self.w2v_array = get_w2v_vector() if self.config.w2v else None
        self.label2idx, self.idx2label = None, None
        self.model = None

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

    def train(self, train_inputs):
        model = train_inputs['model']
        train_data = train_inputs['train_data']
        dev_data = train_inputs['dev_data']
        test_data = train_inputs['test_data']
        self.label2idx, self.idx2label = train_inputs['label2idx'], train_inputs['idx2label']

        train_X, train_y = self._get_X_y(train_data)
        dev_X, dev_y = self._get_X_y(dev_data)
        test_X, test_y = self._get_X_y(test_data)

        if model is ModelE.xgboost:
            train_X = np.vstack((train_X, dev_X))
            train_y = np.hstack((train_y, dev_y))
            self.model = XGBoost(model_config=self.model_config)

        xgboost_clf = self.model(train_X, train_y)
        test_pred = xgboost_clf.predict(test_X)

        print('accuracy_score: ', metrics.accuracy_score(test_y, test_pred))
        print('f1_score: ', metrics.f1_score(test_y, test_pred, average='micro'))
        print('recall_score:', metrics.recall_score(test_y, test_pred, average='micro'))
        print('precision_score:', metrics.precision_score(test_y, test_pred, average='micro'))
        print('confusion_metrics:', metrics.confusion_matrix(test_y, test_pred))
