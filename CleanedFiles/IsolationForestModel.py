from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from Model import Model
import pandas as pd
import numpy as np
import datetime
import joblib
import time
import os


def checkInputFormat(data: np.array):
    if data.ndim != 2:
        raise Exception("Requires a two dimensional ndarray")
    elif data.shape[1] != 10:
        raise Exception("Expected input of transaction consisting of a string and 9 floats")
    for row in data:
        if not isinstance(row[0], str) or not all(isinstance(x, float) for x in row[1:]):
            raise Exception("Expected input of transaction consisting of a string and 9 floats")


def getStartOfDayUnixTime(unixTime: float):
    dateTime = datetime.datetime.fromtimestamp(unixTime)
    startOfDay = datetime.datetime(dateTime.year, dateTime.month, dateTime.day)
    return int(startOfDay.timestamp())


class IsolationForestModel:
    def __init__(self, fraudModel: Model, contamination=0.01):
        self.fraudModel = fraudModel
        self.isoForest = IsolationForest(contamination=contamination, n_jobs=-1)
        self.scaler = StandardScaler()
        self.fraudProbs = {}

    def saveForest(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fName = os.path.join(os.getcwd(), "Saved Models", f"IsoForest_{timestamp}.joblib")
        joblib.dump(self, fName)

    async def getMerchReport(self, merchant: str, day: int) -> pd.DataFrame:
        if self.fraudProbs == {}:
            raise Exception("Model not trained")
        try:
            return self.fraudProbs[merchant, day]
        except Exception:
            raise Exception("Merchant had zero transactions that day")

    def buildForest(self, data: np.array):
        checkInputFormat(data)
        merchDayStats = self.getMerchDayStats(data)
        print("hello")
        statsWithoutMerchantDay = pd.DataFrame([
            {
                k: v for k, v in d.items() if k not in ('merchant', 'day', 'fraud')
            }
            for d in merchDayStats
        ])

        self.fitForest(statsWithoutMerchantDay)
        DailyFraudProbs = self.getIsolationForestFraudProbs(statsWithoutMerchantDay)

        for idx, info in enumerate(merchDayStats):
            self.fraudProbs[(info['merchant'], info['day'])] = max(DailyFraudProbs[idx], info['fraud'])

    def getMerchDayStats(self, data: np.array):
        sameMerchDayTransactions = []
        dataStats = []
        oddHourCount = 0
        prevDay = None
        prevMerch = None

        for row in data:
            hour = datetime.datetime.fromtimestamp(row[6]).hour
            day = getStartOfDayUnixTime(row[6])

            if hour >= 23 or hour <= 6:
                oddHourCount += 1

            if prevMerch == row[0] and day == prevDay:
                sameMerchDayTransactions.append(row)
            else:
                if sameMerchDayTransactions:
                    sameMerchDayTransactions = np.array(sameMerchDayTransactions, dtype=object)
                    transactionFraudProbs = self.getModelFraudProbs(sameMerchDayTransactions[:, 1:9])
                    sameMerchDayAmounts = sameMerchDayTransactions[:, 1]

                    cnt = len(transactionFraudProbs)
                    mean_f, max_f = transactionFraudProbs.mean(), transactionFraudProbs.max()
                    std_f, med_f = transactionFraudProbs.std(), np.median(transactionFraudProbs)
                    meanSpend = sameMerchDayAmounts.mean()
                    maxSpend = sameMerchDayAmounts.max()
                    stdSpend = sameMerchDayAmounts.std()

                    dataStats.append(
                        {
                            'merchant': row[0],
                            'day': day,
                            'num_transactions': cnt,
                            'mean_fraud': mean_f,
                            'max_fraud': max_f,
                            'std_fraud': std_f,
                            'median_fraud': med_f,
                            'mean_spend': meanSpend,
                            'max_spend': maxSpend,
                            'std_spend': stdSpend,
                            'odd_hour_transactions': oddHourCount,
                            'fraud': np.sum(transactionFraudProbs >= self.fraudModel.threshold)
                        }
                    )

                sameMerchDayTransactions = [row]
                oddHourCount = 0
            prevDay = day
            prevMerch = sameMerchDayTransactions[0][0]

        return dataStats

    def fitForest(self, statsWithoutMerchantDay: pd.DataFrame):
        scaledStats = self.scaler.fit_transform(statsWithoutMerchantDay)
        self.isoForest.fit(scaledStats)

    def getIsolationForestFraudProbs(self, statsWithoutMerchantDay: pd.DataFrame):
        scaledProbStats = self.scaler.transform(statsWithoutMerchantDay)
        scores = self.isoForest.decision_function(scaledProbStats)
        probs = 1.0 / (1.0 + np.exp(scores))
        return probs

    def getModelFraudProbs(self, statsWithoutMerchantDay: np.array):
        scaledStats = self.fraudModel.scalar.transform(statsWithoutMerchantDay)
        return self.fraudModel.model.predict_proba(scaledStats)[:, 1]

