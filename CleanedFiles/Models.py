from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from ModelParameters import *
from Model import Model
import xgboost as xgb

config = {
    "modelType": "RF",
    "params": RandomForestParams()
}


def getModel():
    values = list(config.values())
    match values[0]:
        case "NN":
            return NeuralNetwork(values[1])
        case "LR":
            return LogisticRegression(values[1]) 
        case "NB":
            return NaiveBayes(values[1])
        case "GB":
            return GradientBoostingMachine(values[1])
        case "XG":
            return XGBoost(values[1])
        case "CB":
            return CatBoost(values[1])
        case "RF":
            return RandomForest(values[1])
        case "SGD":
            return SGDClassifier(values[1])


class NeuralNetwork(Model):
    def __init__(self, params: NeuralNetworkParams):
        self.params = params
        super().__init__(params.threshold)

    def getModel(self):
        return MLPClassifier(**self.params.getParams())

    def getModelType(self):
        return "NN"


class LogisticRegression(Model):
    def __init__(self, params: LogisticRegressionParams):
        self.params = params
        super().__init__(params.threshold)

    def getModel(self):
        return LogisticRegression(**self.params.getParams())

    def getModelType(self):
        return "LR"


class NaiveBayes(Model):
    def __init__(self, params: NaiveBayesParams):
        self.params = params
        super().__init__(params.threshold)

    def getModel(self):
        return GaussianNB(**self.params.getParams())

    def getModelType(self):
        return "NB"


class GradientBoostingMachine(Model):
    def __init__(self, params: GradientBoostingMachineParams):
        self.params = params
        super().__init__(params.threshold)

    def getModel(self):
        return GradientBoostingClassifier(**self.params.getParams())

    def getModelType(self):
        return "GB"


class XGBoost(Model):
    def __init__(self, params: XGBoostParams):
        self.params = params
        super().__init__(params.threshold)

    def getModel(self):
        return xgb.XGBClassifier(**self.params.getParams())

    def getModelType(self):
        return "XGB"


class CatBoost(Model):
    def __init__(self, params: CatBoostParams):
        self.params = params
        super().__init__(params.threshold)

    def getModel(self):
        return CatBoostClassifier(**self.params.getParams())

    def getModelType(self):
        return "CB"


class RandomForest(Model):
    def __init__(self, params: RandomForestParams):
        self.params = params
        super().__init__(params.threshold)

    def getModel(self):
        return RandomForestClassifier(**self.params.getParams())

    def getModelType(self):
        return "RF"


class SGDClassifier(Model):
    def __init__(self, params: SGDClassifierParams):
        self.params = params
        super().__init__(params.threshold)

    def getModel(self):
        return SGDClassifier(**self.params.getParams())

    def getModelType(self):
        return "SGD"
