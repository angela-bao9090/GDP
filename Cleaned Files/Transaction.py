from pydantic import BaseModel


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
    isFraud: float

    def toArray(self):
        return [
            self.merchId, self.amount, self.zip, self.lat, self.long, self.cityPop, self.unixTime, self.merchLat,
            self.merchLong, self.isFraud
        ]