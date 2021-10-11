from data_process import DataProcessConfig, DataProcess


class Config:
    def __init__(self):
        # TODO: check these lines every running time
        self.use_cuda = False
        self.need_preprocess = True
        self.debug_mode = True
        self.en_train = True
        self.en_test = True
        self.unique = 'testing'

        # data process
        self.data_process_config = DataProcessConfig()
        self.data_process_config.preprocess = True if self.need_preprocess else False

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
    data_process = DataProcess(config.data_process_config)

    if config.need_preprocess:
        data_process.preprocess()

    if config.en_train:
        train_data = data_process.get_data('train')
        dev_data = data_process.get_data('dev')



    if config.en_test:
        test_data = data_process.get_data('test')
