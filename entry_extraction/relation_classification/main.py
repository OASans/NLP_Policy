from entry_extraction.relation_classification.data_process import DataProcessConfig, DataProcess


class Config:
    def __init__(self):
        # TODO: check these lines every running time
        self.use_cuda = False
        self.need_preprocess = True
        self.debug_mode = False
        self.en_train = False
        self.en_test = False
        self.en_pred_so_entity = True
        self.unique = 'testing'

        # data process
        self.data_process_config = DataProcessConfig()
        self.data_process_config.preprocess = True if self.need_preprocess else False
        self.data_process_config.debug_mode = True if self.debug_mode else False


if __name__ == '__main__':
    config = Config()
    data_process = DataProcess(config.data_process_config)

    if config.need_preprocess:
        print('preprocessing...')
        data_process.preprocess()

    label2idx, idx2label = data_process.get_data('label')
