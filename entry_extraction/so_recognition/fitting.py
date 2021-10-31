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
        self.result_acc_path = os.path.join(self.result_data_path, 'acc_result_{}.json'.format(unique))
        self.result_pic_path = os.path.join(self.result_data_path, 'acc_pic_{}.png'.format(unique))
        self.entity_result_path = os.path.join(os.getcwd(), '../entity_result/')
        if not os.path.exists(self.entity_result_path):
            os.makedirs(self.entity_result_path)


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

        def _get_bio(sentence_len, entity_list):
            bioes_label = ['O'] * sentence_len
            for entity in entity_list:
                bioes_label[entity[0]] = 'B-{}'.format(entity[2])
                for i in range(entity[0] + 1, entity[1] + 1):
                    bioes_label[i] = 'I-{}'.format(entity[2])
            bioes_label = [self.label2idx[label] for label in bioes_label]
            return bioes_label

        inputs = self.collate_fn_test(batch)
        y_true = []
        max_len = inputs['sentence_tokens'].shape[1]
        for sample in batch:
            text_length = len(sample['sentence_tokens'])
            bioes_label = _get_bio(text_length, sample['entry_list'])  # TODO: 改名
            bioes_label = bioes_label + [0] * (max_len - text_length)
            y_true.append(bioes_label)
        y_true = torch.tensor(y_true).long()
        if self.use_cuda:
            y_true = y_true.cuda()
        return inputs, y_true

    def collate_fn_entity(self, batch):
        inputs, targets = self.collate_fn_train(batch)
        sids = []
        sentence_tokens = []
        entities = []
        for sample in batch:
            sids.append(sample['sid'])
            sentence_tokens.append(sample['sentence_tokens'])
            entities.append(sample['entry_list'])  # TODO: 改名成entity
        return inputs, {'sids': sids, 'sentence_tokens': sentence_tokens, 'entities': entities,
                        'targets': targets.numpy().tolist()}

    def get_collate_fn(self, mode='train'):
        if mode == 'train' or mode == 'dev':
            return self.collate_fn_train
        elif mode == 'test':
            return self.collate_fn_test
        elif mode == 'entity':
            return self.collate_fn_entity

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

    def get_chunks(self, seq):
        # TODO: 感觉这个怪怪的，但是网上好多人都用这个，我感觉不太对
        """Given a sequence of tags, group entities and their position

        Args:
            seq: [4, 4, 0, 0, ...] sequence of labels
            tags: dict["O"] = 4

        Returns:
            list of (chunk_type, chunk_start, chunk_end)

        Example:
            seq = [4, 5, 0, 3]
            tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
            result = [("PER", 0, 2), ("LOC", 3, 4)]

        """

        def get_chunk_type(tok):
            tag_name = self.idx2label[str(tok)]
            tag_class = tag_name.split('-')[0]
            tag_type = tag_name.split('-')[-1]
            return tag_class, tag_type

        # We assume by default the tags lie outside a named entity
        default = self.label2idx['O']
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            if tok == -1:
                break
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
                if chunk_type is None:
                    # Initialize chunk for each entity
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    # If chunk class is B, i.e., its a beginning of a new named entity
                    # or, if the chunk type is different from the previous one, then we
                    # start labelling it as a new entity
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
                else:
                    pass
        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)

        return chunks

    def evaluate(self, pred, y_true):
        batch_size = len(pred)
        y_true = y_true.numpy()
        correct_preds, total_correct, total_preds = 0., 0., 0.

        for i in range(batch_size):
            ground_truth_id = y_true[i]
            predicted_id = pred[i]
            lab_chunks = set(self.get_chunks(ground_truth_id))
            lab_pred_chunks = set(self.get_chunks(predicted_id))

            # Updating the count variables
            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)

        # Calculating the F1-Score
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        new_F1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return p, r, new_F1

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
                with torch.no_grad():
                    P, R, F1 = self.evaluate(preds['pred'], targets)
                    print('epoch: {}, step: {}/{}, loss: {:.6f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(
                        epoch, step, train_steps, loss, P, R, F1))
                print('lr: {:.6f}'.format(optimizer.param_groups[0]['lr']))
            # dev
            with torch.no_grad():
                dev_inputs = {'data': dev_data}
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
            self._plotting_data(eval_result['p'], eval_result['r'], eval_result['f1'], eval_result['loss'].item())
            scheduler.step()

    def eval(self, dev_inputs):
        dev_data = dev_inputs['data']
        dev_dataloader = DataLoader(dev_data, batch_size=self.batch_size, collate_fn=self.get_collate_fn('dev'))
        result = {}
        metrics_data = {'loss': 0, 'p': 0, 'r': 0, 'f1': 0, 'batch_num': 0}
        with torch.no_grad():
            print('==================================evaluating dev data...==================================')
            for step, (inputs, targets) in enumerate(dev_dataloader):
                self.model.eval()
                preds = self.model(inputs)
                loss = self.model.cal_loss(preds, targets, inputs['sentence_masks'])
                P, R, F1 = self.evaluate(preds['pred'], targets)
                metrics_data['loss'] += loss.cpu().float()
                metrics_data['p'] += P
                metrics_data['r'] += R
                metrics_data['f1'] += F1
                metrics_data['batch_num'] += 1
            result['loss'] = metrics_data['loss'] / metrics_data['batch_num']
            result['p'] = metrics_data['p'] / metrics_data['batch_num']
            result['r'] = metrics_data['r'] / metrics_data['batch_num']
            result['f1'] = metrics_data['f1'] / metrics_data['batch_num']
            print('loss: {:.6f}, p: {:.4f}, r: {:.4f}, f1: {:.4f}'.format(result['loss'], result['p'], result['r'], result['f1']))
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

    def get_pred_entity(self, inputs, data_type='test'):
        model = inputs['model']
        self.label2idx, self.idx2label = inputs['label2idx'], inputs['idx2label']
        data = inputs['data']
        model.load_state_dict(torch.load(self.config.result_model_path))
        if self.use_cuda:
            model = model.cuda()
        self.model = model

        dataloader = DataLoader(data, batch_size=self.batch_size, collate_fn=self.get_collate_fn('entity'))
        results = []
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(dataloader):
                self.model.eval()
                preds = self.model(inputs)
                for i, pred in enumerate(preds['pred']):
                    pred_entity = self.get_chunks(pred)
                    true_entity = self.get_chunks(targets['targets'][i])
                    res = {'sid': targets['sids'][i], 'true_entity': true_entity, 'pred_entity': pred_entity}
                    results.append(res)
        result_path = os.path.join(self.config.entity_result_path, '{}_entity.json'.format(data_type))
        with open(result_path, 'w') as f:
            json.dump(results, f)
        return results