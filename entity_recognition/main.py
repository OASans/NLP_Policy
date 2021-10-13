from data_process import DataProcessConfig, DataProcess


class Config:
    def __init__(self):
        # TODO: check these lines every running time
        self.use_cuda = False
        self.need_preprocess = False
        self.debug_mode = True
        self.en_train = True
        self.en_test = False
        self.unique = 'testing'

        # data process
        self.data_process_config = DataProcessConfig()
        self.data_process_config.preprocess = True if self.need_preprocess else False


if __name__ == '__main__':
    config = Config()
    data_process = DataProcess(config.data_process_config)

    if config.need_preprocess:
        print('preprocessing...')
        data_process.preprocess()

    label2idx, idx2label = data_process.get_data('label')

    if config.en_train:
        print('training...')
        train_data = data_process.get_data('train')
        dev_data = data_process.get_data('dev')
        test_data = data_process.get_data('test')

