from module.data_process import DataProcessConfig, DataProcess, RandomDataProcess
from module.fitting import FittingConfig, ModelFitting, RandomModelFitting
from module.models import ModelConfig, Bert_Crf, Bert_Flat_Crf, Random_Crf, Random_Flat_Crf, Bert_Transformer_Crf


class Config:
    def __init__(self):
        # TODO: check these lines every running time
        self.use_cuda = False
        self.need_preprocess = False
        self.debug_mode = False
        self.en_train = True
        self.en_test = False
        self.unique = 'testing'
        self.w2v = False

        # data process
        self.data_process_config = DataProcessConfig()
        self.data_process_config.preprocess = True if self.need_preprocess else False
        self.data_process_config.debug_mode = True if self.debug_mode else False
        self.data_process_config.w2v = self.w2v

        # fitting
        self.fitting_config = FittingConfig(self.unique)
        self.fitting_config.use_cuda = True if self.use_cuda else False
        self.fitting_config.num_tags = self.data_process_config.num_tags
        self.fitting_config.epochs = 1 if self.debug_mode else self.fitting_config.epochs
        self.fitting_config.w2v = self.w2v

        # models
        self.model_config = ModelConfig()
        self.model_config.num_tags = self.data_process_config.num_tags
        self.model_config.w2v = self.w2v
        self.model_config.max_len = self.data_process_config.max_len


if __name__ == '__main__':
    config = Config()

    data_process = DataProcess(config.data_process_config)
    # data_process = RandomDataProcess(config.data_process_config)
    if config.need_preprocess:
        print('preprocessing...')
        data_process.preprocess()

    label2idx, idx2label = data_process.get_data('label')

    # model_fitting = RandomModelFitting(config.fitting_config)
    model_fitting = ModelFitting(config.fitting_config)
    if type(model_fitting) == RandomModelFitting:
        config.model_config.add_vocab()

    # model selection
    # model = Random_Crf(config.model_config)
    # model = Random_Flat_Crf(config.model_config)
    # model = Bert_Crf(config.model_config)
    # model = Bert_Flat_Crf(config.model_config)
    model = Bert_Transformer_Crf(config.model_config)

    if config.en_train:
        print('training...')
        train_data = data_process.get_data('train')
        dev_data = data_process.get_data('dev')
        test_data = data_process.get_data('test')
        train_inputs = {'model': model, 'train_data': train_data, 'dev_data': dev_data, 'label2idx': label2idx,
                        'idx2label': idx2label}
        model_fitting.train(train_inputs)

        # plot train and dev acc
        model_fitting.plot_acc()

    if config.en_test:
        print('testing...')
        test_data = data_process.get_data('test')
        test_inputs = {'model': model, 'data': test_data, 'label2idx': label2idx, 'idx2label': idx2label}
        model_fitting.test(test_inputs)
