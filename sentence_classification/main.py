from data_process import DataProcessConfig, DataProcess
from fitting import FittingConfig, ModelFitting
from models import ModelE


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
        self.fitting_config = FittingConfig(self.unique)

        # models
        self.model = ModelE.xgboost


if __name__ == '__main__':
    config = Config()
    data_process = DataProcess(config.data_process_config)

    if config.need_preprocess:
        data_process.preprocess()

    if config.en_train:
        train_data = data_process.get_data('train')
        dev_data = data_process.get_data('dev')
        test_data = data_process.get_data('test')
        train_inputs = {'model': config.model, 'train_data': train_data, 'dev_data': dev_data}

    if config.en_test:
        test_data = data_process.get_data('test')
        test_inputs = {'model': config.model, 'test_data': test_data}