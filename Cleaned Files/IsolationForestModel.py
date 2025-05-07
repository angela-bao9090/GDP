from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import datetime
import joblib
import time
import os


class IsolationForestModel:
    def __init__(self, fraud_model, contamination=0.01):
        # THIS HAS TO BE THE MODEL, NOT THE CLASS
        self.fraud_model = fraud_model
        self.isoForest = IsolationForest(contamination=contamination, n_jobs=-1)
        self.scaler = StandardScaler()
        self.fraudProbs = None

    def save(self, path: str = "."):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fName = os.path.join(path, f"IsoForest_{timestamp}.joblib")
        joblib.dump(self, fName)

    def getMerchReport(self, merchant, day) -> pd.DataFrame:
        if self.fraudProbs is not None:
            raise Exception("Model not trained")

        try:
            return self.fraudProbs[merchant, day]
        except Exception:
            raise Exception("Merchant had zero transactions that day")

    def buildForest(self, data: [list[str | float]]):
        merchDayStats = self.getMerchDayStats(data)

        statsWithoutMerchantDay = pd.DataFrame([
            {
                k: v for k, v in d.items() if k not in ('merchant', 'day')
            }
            for d in merchDayStats
        ])

        self.fitForest(statsWithoutMerchantDay)

        DailyFraudProbs = self.getDailyFraudProbs(statsWithoutMerchantDay)

        MerchDayFraudProbs = {}
        for idx, info in enumerate(merchDayStats):
            MerchDayFraudProbs[(info['merchant'], info['day'])] = DailyFraudProbs[idx]

        self.fraudProbs = MerchDayFraudProbs

        del statsWithoutMerchantDay

    def getMerchDayStats(self, data: [list[str | float]]):
        dailyTransaction = []
        dailyTransactionStats = []
        dailyTransactionHours = []
        dailyTransactionAmounts = []
        dataStats = []
        for row in data:
            row[6] = datetime.datetime.fromtimestamp(row[6])
            if not dailyTransaction:
                dailyTransaction.append(row)
                dailyTransactionStats.append(row[1:6, 7:])
                dailyTransactionHours.append(row[6])
                dailyTransactionAmounts.append(row[1])

            elif dailyTransaction[0][0] == row[0] and dailyTransaction[0][6].day == row[6].day:
                dailyTransaction.append(row)
                dailyTransactionStats.append(row[1:6, 7:])
                dailyTransactionHours.append(row[6].hour)
                dailyTransactionAmounts.append(row[1])

            else:
                if hasattr(self.fraud_model, 'scalar') and self.fraud_model.scalar is not None:
                    scaledStats = self.fraud_model.scalar.transform(dailyTransactionStats)
                else:
                    scaledStats = dailyTransactionStats

                transactionFraudProb = self.fraud_model.predict_prob(scaledStats)[:, 1]
                cnt = len(transactionFraudProb)
                mean_f, max_f = transactionFraudProb.mean(), transactionFraudProb.max()
                std_f, med_f = transactionFraudProb.std(), np.median(transactionFraudProb)
                odd_cnt = sum(1 for x in dailyTransactionHours if x >= 23 or x <= 6)
                meanSpend = dailyTransactionAmounts.mean()
                maxSpend = dailyTransactionAmounts.max()
                stdSpend = dailyTransactionAmounts.std()

                dataStats.append(
                    {
                        'merchant': dailyTransaction[0][0],
                        'day': dailyTransaction[0][6].day,
                        'num_transactions': cnt,
                        'mean_fraud': mean_f,
                        'max_fraud': max_f,
                        'std_fraud': std_f,
                        'median_fraud': med_f,
                        'mean_spend': meanSpend,
                        'max_spend': maxSpend,
                        'std_spend': stdSpend,
                        'odd_hour_transactions': odd_cnt
                    }
                )

        del dailyTransaction
        del dailyTransactionAmounts
        del dailyTransactionStats
        del dailyTransactionHours

        return dataStats

    def fitForest(self, statsWithoutMerchantDay: pd.DataFrame):
        scaledStats = self.scaler.fit_transform(statsWithoutMerchantDay)
        self.isoForest.fit(scaledStats)

    def getDailyFraudProbs(self, statsWithoutMerchantDay: pd.DataFrame):
        scaledProbStats = self.scaler.transform(statsWithoutMerchantDay)
        scores = self.isoForest.decision_function(scaledProbStats)
        probs = 1.0 / (1.0 + np.exp(scores))
        return probs
