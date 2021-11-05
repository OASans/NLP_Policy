from data_process import DataProcessConfig, DataProcess
from fitting import FittingConfig, ModelFitting
from models import ModelE, ModelConfig


class Config:
    def __init__(self):
        # ==========================================开关=======================================================
        # TODO: check these lines every running time
        self.use_cuda = False
        self.need_preprocess = False
        self.debug_mode = True
        self.en_train = True
        self.en_test = False
        self.unique = 'testing'
        # ===========================================各部分配置======================================================

        # data process
        self.data_process_config = DataProcessConfig()
        self.data_process_config.preprocess = True if self.need_preprocess else False

        # fitting
        self.fitting_config = FittingConfig(self.unique)

        # models
        self.model = ModelE.xgboost
        self.model_config = ModelConfig()


if __name__ == '__main__':
    config = Config()
    data_process = DataProcess(config.data_process_config)
    model_fitting = ModelFitting(config.fitting_config, config.model_config)

    if config.need_preprocess:
        print('preprocessing...')
        data_process.preprocess()
        print('preprocess finished!')

    label2idx, idx2label = data_process.get_data('label')
    if config.en_train:
        print('training...')
        train_data = data_process.get_data('train')
        dev_data = data_process.get_data('dev')
        test_data = data_process.get_data('test')
        train_inputs = {'model': config.model, 'train_data': train_data, 'dev_data': dev_data, 'test_data': test_data,
                        'label2idx': label2idx, 'idx2label': idx2label}
        model_fitting.train(train_inputs)

    if config.en_test:
        print('testing...')
        test_data = data_process.get_data('test')
        test_inputs = {'model': config.model, 'test_data': test_data}