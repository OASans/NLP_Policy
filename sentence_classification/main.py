from models import MethodE


class Config:
    def __init__(self):
        # TODO: check these 5 lines every time
        self.method = MethodE.ml
        self.debug_mode = True
        self.en_train = True
        self.en_test = True
        self.need_preprocess = True
        self.unique = 'testing'

        # cuda setting
        self.use_cuda = False

        # DataSet
        self.raw_data_path = '../data_process/datasets/sentence_classification.json'
        self.label_num = 0
        self.label2idx = {}
        self.idx2label = {}
        self.vocab_size = 0
        self.vocab2idx = {}
        self.idx2vocab = {}
        self.dev_rate = 0.2
        self.test_rate = 0.2

        # fitting
        self.batch_size = 10
        self.epochs = 2 if self.debug_mode else 100
        self.lr = 0.001
        self.shuffle = True
        self.early_stop = True
        self.patience = 8
        self.decay = 0.95
        # fitting result
        self.result_path = './result'

        # models
        self.dropout = 0.5
        self.word_embedding_dim = 300


if __name__ == '__main__':
    config = Config()
