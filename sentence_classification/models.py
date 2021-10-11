from enum import Enum
from xgboost import XGBClassifier

ModelE = Enum('ModelE', ('xgboost', 'lgbm', 'adaboost'))


class ModelConfig:
    def __init__(self):
        # xgboost
        self.lr = 0.01
        self.n_estimators = 100
        self.objective = 'multi:softmax'
        self.max_depth = 8


class XGBoost:
    def __init__(self, model_config):
        self.model = XGBClassifier(learning_rate=model_config.lr, n_estimators=model_config.n_estimators,
                                   objective=model_config.objective, max_depth=model_config.max_depth)
