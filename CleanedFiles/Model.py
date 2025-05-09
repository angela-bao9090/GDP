from sklearn.metrics import f1_score, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
import numpy as np
import joblib
import time
import os


def undersample(data):
    X = data[:, :-1]
    Y = data[:, -1]
    X_res, Y_res = RandomUnderSampler().fit_resample(X, Y)
    return np.hstack((X_res, Y_res[:, None]))


def checkInputFormat(data: np.array, target: bool):
    if data.ndim != 2:
        raise Exception("Requires a two dimensional ndarray")
    elif (data.shape[1] != 8 and not target) or (data.shape[1] != 9 and target) or not np.issubdtype(data.dtype,
                                                                                                     np.floating):
        raise Exception("Expected input of transaction consisting of 8 floats (or 9 including target)")


class Model(ABC):
    def __init__(self, threshold):
        self.predictedResults = []
        self.scalar = StandardScaler()
        self.threshold = threshold
        self.model = self.getModel()
        self.modelType = self.getModelType()

    @abstractmethod
    def getModel(self):
        pass

    @abstractmethod
    def getModelType(self):
        pass

    def commenceTraining(self, trainingData: np.ndarray, testData: np.ndarray):
        checkInputFormat(trainingData, True)
        checkInputFormat(testData, True)
        scaledData = self.scalar.fit_transform(trainingData[:, :-1])
        trueResults = trainingData[:, -1]
        self.model.fit(scaledData, trueResults)
        self.predict(testData, True)
        trueResults = testData[:, -1]
        print(f"Accuracy: {(accuracy_score(trueResults, self.predictedResults)) * 100:.2f}%")
        print(f"F1 Score: {(f1_score(trueResults, self.predictedResults)):.4f}")
        print(f"weighted F1 score: {(f1_score(trueResults, self.predictedResults, average='weighted')):.4f}")

    def predict(self, data: np.ndarray, hasTarget: bool):
        checkInputFormat(data, hasTarget)
        scaledData = self.scalar.transform(data[:, :-1]) if hasTarget else self.scalar.transform(data)
        predictedProbs = self.model.predict_proba(scaledData)[:, 1]
        self.predictedResults = [1 if x >= self.threshold else 0 for x in predictedProbs]
        return self.predictedResults

    def saveModel(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"{self.modelType}Model-{timestamp}.joblib"
        folder_path = os.path.join(os.getcwd(), "Saved Models", file_name)
        joblib.dump(self, folder_path)
