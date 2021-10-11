import os


class ModelFitting():
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.class_num = config.label_num
        self.label2idx = config.label2idx
        self.idx2label = config.idx2label
        self.lr = config.lr
        self.use_cuda = config.use_cuda
        self.early_stop = config.early_stop
        self.patience = config.patience

        self.vocab = config.vocab2idx
        self.result_path = config.result_path
        self.result_model_path = os.path.join(self.result_path, 'best_model_{}.pt'.format(config.unique))
        self.result_data_path = os.path.join(self.result_path, 'acc_result_{}.json'.format(config.unique))
        self.result_pic_path = os.path.join(self.result_path, 'acc_pic_{}.png'.format(config.unique))
        self.result_submit_path = os.path.join(self.result_path, 'submission.csv')

        # acc_data
        self.acc_result = {'train_acc': [], 'train_loss': [], 'dev_acc': [], 'dev_loss': []}

        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)