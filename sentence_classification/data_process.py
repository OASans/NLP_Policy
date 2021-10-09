import os
import re
import json
import hanlp


class DataProcess:
    def __init__(self, config):
        self.data_path = config.raw_data_path
        self.pretrained_path = config.pretrained_path

        self.processed_data_path = os.path.join(os.getcwd(), 'data')

        self.train_path = os.path.join(self.processed_data_path, 'train.json')
        self.dev_path = os.path.join(self.processed_data_path, 'dev.json')
        self.test_path = os.path.join(self.processed_data_path, 'test.json')


