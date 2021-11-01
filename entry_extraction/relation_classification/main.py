import sys
sys.path.append('/remote-home/aqshi/NLP_Policy/NLP_Policy')
sys.path.append('/remote-home/aqshi/NLP_Policy')

from entry_extraction.relation_classification.data_process import DataProcessConfig, DataProcess
from entry_extraction.relation_classification.fitting import FittingConfig, ModelFitting
from entry_extraction.relation_classification.models import BertESIM, ModelConfig


class Config:
    def __init__(self):
        # TODO: check these lines every running time
        self.use_cuda = True
        self.need_preprocess = False
        self.debug_mode = False
        self.en_train = True
        self.en_test = True
        self.unique = 'testing'

        # data process
        self.data_process_config = DataProcessConfig()
        self.data_process_config.preprocess = True if self.need_preprocess else False
        self.data_process_config.debug_mode = True if self.debug_mode else False

        # fitting
        self.fitting_config = FittingConfig(self.unique)
        self.fitting_config.use_cuda = True if self.use_cuda else False
        self.fitting_config.num_tags = self.data_process_config.num_tags
        self.fitting_config.epochs = 2 if self.debug_mode else self.fitting_config.epochs

        # models
        self.model_config = ModelConfig()
        self.model_config.num_tags = self.data_process_config.num_tags
        self.model_config.use_cuda = True if self.use_cuda else False
        self.model_config.device = 'cpu' if not self.use_cuda else self.model_config.device


if __name__ == '__main__':
    config = Config()
    data_process = DataProcess(config.data_process_config)

    if config.need_preprocess:
        print('preprocessing...')
        data_process.preprocess()

    label2idx, idx2label = data_process.get_data('label')

    model_fitting = ModelFitting(config.fitting_config)
    model = BertESIM(config.model_config)

    if config.en_train:
        print('training...')
        train_data = data_process.get_data('train')
        dev_data = data_process.get_data('dev')
        train_inputs = {'model': model, 'train_data': train_data, 'dev_data': dev_data, 'label2idx': label2idx,
                        'idx2label': idx2label}
        model_fitting.train(train_inputs)

        # plot train and dev acc
        model_fitting.plot_acc()

    if config.en_test:
        print('testing...')
        test_data = data_process.get_data('test')
        test_inputs = {'model': model, 'test_data': test_data, 'label2idx': label2idx, 'idx2label': idx2label}
        model_fitting.test(test_inputs)
