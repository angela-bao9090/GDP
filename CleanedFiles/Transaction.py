from pydantic import BaseModel
import numpy as np


class Transaction(BaseModel):
    merchId: str
    amount: float
    zip: float
    lat: float
    long: float
    cityPop: float
    unixTime: float
    merchLat: float
    merchLong: float

    def toArray(self):
        return np.array([
            self.merchId, self.amount, self.zip, self.lat, self.long, self.cityPop, self.unixTime, self.merchLat,
            self.merchLong
        ], dtype=object)


class TargetedTransaction(Transaction):
    isFraud: float

    def toArray(self):
        return np.array([
            self.merchId, self.amount, self.zip, self.lat, self.long, self.cityPop, self.unixTime, self.merchLat,
            self.merchLong, self.isFraud
        ], dtype=object)

