from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from Plot import plotCM, plotP
import numpy as np
import joblib
import time
import os


def undersample(data):
    data = np.array(data)
    X = data[:, :-1]
    Y = data[:, -1]
    X_res, Y_res = RandomUnderSampler().fit_resample(X, Y)
    return np.hstack((X_res, Y_res[:, None])).tolist()


class Model(ABC):
    def __init__(self, threshold):
        self.yPred = []
        self.yTrue = []
        self.scalar = StandardScaler()
        self.threshold = threshold
        self.testing = False
        self.targetless = True
        self.target = None
        self.features = None
        self.model = self.getModel()
        self.modelType = self.getModelType()
        self.cm: np.ndarray = np.array([[]])
        self.padded_cm: np.ndarray = np.array([])

    @abstractmethod
    def getModel(self):
        pass

    @abstractmethod
    def getModelType(self):
        pass

    def train(self):
        self.model.fit(self.features, self.target)

    def commenceTraining(self, trainingData, testData):
        self.loadTargeted(trainingData, False)
        self.train()
        self.resetStored()
        self.loadTargeted(testData)
        self.test()
        self.resetStored()

    def saveModel(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"{self.modelType}Model-{timestamp}.joblib"
        folder_path = os.path.join(os.getcwd(), "Saved Models", file_name)
        joblib.dump(self, folder_path)

    def resetStored(self):
        self.yPred = []
        self.yTrue = []

    # Ensure that input is a 2D array (each row is a separate data point)
    def loadTargetless(self, data_points):
        self.features = np.array(data_points)
        self.features = self.scalar.transform(self.features)
        self.targetless = False

    # Ensure that input is a 2D array (each row is a separate data point)
    def loadTargeted(self, data_points, existScalar=True):
        data = np.array(data_points)
        self.features = data[:, :-1]
        self.features = self.scalar.transform(self.features) if existScalar \
            else self.scalar.fit_transform(self.features)
        self.target = data[:, -1]
        self.targetless = False

    def predict(self):
        if self.features is None:
            pass
        elif self.testing:
            y_prob = self.model.predict_proba(self.features)[:, 1]
            self.yPred.extend((y_prob >= self.threshold).astype(int))
        else:
            self.resetStored()
            y_prob = self.model.predict_proba(self.features)[:, 1]
            self.yPred.extend((y_prob >= self.threshold).astype(int))
            self.cm = confusion_matrix([0] * len(self.yPred), self.yPred)
            if self.cm.shape[1] < 2:
                self.padded_cm = np.zeros((1, 2), dtype=int)
                self.padded_cm[0, int(self.yPred[0])] = self.cm[0, 0]
                self.cm = self.padded_cm
            plotP(self.cm)
            self.resetStored()

    def test(self):
        if self.targetless:
            pass
        else:
            self.testing = True
            self.predict()
            self.yTrue.extend(self.target)
            self.cm = confusion_matrix(self.yTrue, self.yPred)
            if self.cm.shape[0] < 2 or self.cm.shape[1] < 2:
                self.padded_cm = np.zeros((2, 2), dtype=int)
                if self.cm.shape == (1, 1):
                    self.padded_cm[int(self.yTrue[0]), int(self.yPred[0])] = self.cm[0, 0]
                elif self.cm.shape == (1, 2):
                    self.padded_cm[int(self.yTrue[0]), 0] = self.cm[0, 0]
                    self.padded_cm[int(self.yTrue[0]), 1] = self.cm[0, 1]
                elif self.cm.shape == (2, 1):
                    self.padded_cm[0, int(self.yPred[0])] = self.cm[0, 0]
                    self.padded_cm[1, int(self.yPred[0])] = self.cm[1, 0]

                self.cm = self.padded_cm
            self.testing = False

            print(f"Accuracy: {((self.cm[0, 0] + self.cm[1, 1]) / self.cm.sum()) * 100:.2f}%")
            print(f"F1 Score: {(f1_score(self.yTrue, self.yPred)):.4f}")
            print(f"weighted F1 score: {(f1_score(self.yTrue, self.yPred, average='weighted')):.4f}")

    def getCM(self):
        plotCM([self.cm], ["Model Test Results"])
