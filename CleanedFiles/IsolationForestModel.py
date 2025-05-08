from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from Models import mlModelTypes
from Model import Model
import pandas as pd
import numpy as np
import datetime
import joblib
import time
import os


class IsolationForestModel:
    def __init__(self, mlModel: mlModelTypes, fraudModel: Model, contamination=0.01):
        self.fraudModel = fraudModel
        self.isoForest = IsolationForest(contamination=contamination, n_jobs=-1)
        self.scaler = StandardScaler()
        self.fraudProbs: dict = {}

    def saveForest(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fName = os.path.join(os.getcwd(), "Saved Models", f"IsoForest_{timestamp}.joblib")
        joblib.dump(self, fName)

    async def getMerchReport(self, merchant: str, day: int) -> pd.DataFrame:
        if self.fraudProbs is not None:
            raise Exception("Model not trained")
        else:
            try:
                return self.fraudProbs[merchant, day]
            except Exception:
                raise Exception("Merchant had zero transactions that day")

    def buildForest(self, data: [list[str | float]]):
        self.fraudProbs = {}
        merchDayStats = self.getMerchDayStats(data)

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

    def getMerchDayStats(self, data: [list[str | float]]):
        dailyTransaction = []
        dailyTransactionStats = []
        dailyTransactionAmounts = []
        dataStats = []
        oddHourCount = 0
        prevDay = None
        for row in data:
            hour = datetime.datetime.fromtimestamp(row[6]).hour
            dateTime = datetime.datetime.fromtimestamp(row[6])
            startOfDay = datetime.datetime(dateTime.year, dateTime.month, dateTime.day)
            day = int(startOfDay.timestamp())

            if hour >= 23 or hour <= 6:
                oddHourCount += 1

            if not dailyTransaction:
                dailyTransaction.append(row)
                dailyTransactionStats.append(row[1:6] + row[7:])
                dailyTransactionAmounts.append(row[1])

            elif dailyTransaction[0][0] == row[0] and day == prevDay:
                dailyTransaction.append(row)
                dailyTransactionStats.append(row[1:6] + row[7:])
                dailyTransactionAmounts.append(row[1])

                dailyTransactionAmounts = np.array(dailyTransactionAmounts)

                transactionFraudProbs = self.getModelFraudProbs([x[1:9] for x in dailyTransaction])

                cnt = len(transactionFraudProbs)
                mean_f, max_f = transactionFraudProbs.mean(), transactionFraudProbs.max()
                std_f, med_f = transactionFraudProbs.std(), np.median(transactionFraudProbs)
                meanSpend = dailyTransactionAmounts.mean()
                maxSpend = dailyTransactionAmounts.max()
                stdSpend = dailyTransactionAmounts.std()

                dataStats.append(
                    {
                        'merchant': dailyTransaction[0][0],
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
                        'fraud': sum([1 if x >= self.fraudModel.threshold else 0 for x in transactionFraudProbs])
                    }
                )

                dailyTransaction = []
                dailyTransactionStats = []
                dailyTransactionAmounts = []
                oddHourCount = 0
            prevDay = day

        return dataStats

    def fitForest(self, statsWithoutMerchantDay: pd.DataFrame):
        scaledStats = self.scaler.fit_transform(statsWithoutMerchantDay)
        self.isoForest.fit(scaledStats)

    def getIsolationForestFraudProbs(self, statsWithoutMerchantDay: pd.DataFrame):
        scaledProbStats = self.scaler.transform(statsWithoutMerchantDay)
        scores = self.isoForest.decision_function(scaledProbStats)
        probs = 1.0 / (1.0 + np.exp(scores))
        return probs

    def getModelFraudProbs(self, statsWithoutMerchantDay: [[float]]):
        scaledStats = self.fraudModel.scalar.transform(np.array(statsWithoutMerchantDay))
        return self.fraudModel.model.predict_proba(scaledStats)[:, 1]

