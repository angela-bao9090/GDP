from Transaction import TargetedTransaction
from databases import Database
import numpy as np
import ssl

ssl_context = ssl.create_default_context(
        cafile="/Users/jackm/Documents/DigiCertGlobalRootCA.crt.pem"
    )
ssl_context.check_hostname = True
ssl_context.verify_mode = ssl.CERT_REQUIRED

databaseURL = (
    "mysql+aiomysql://zofia:Password123!@gdp-dojo-2025.mysql.database.azure.com:3306/"
    "fraud_engine_database?ssl_ca=/Users/jackm/Documents/DigiCertGlobalRootG2.crt.pem"
    "&ssl=true"
)

database = Database(databaseURL, ssl=ssl_context)


class DbConnection:
    def __init__(self, db):
        self.db = db

    async def getTestData(self):
        query = "SELECT * FROM test"
        rows = await self.db.fetch_all(query=query)
        return np.array([
            TargetedTransaction(
                merchId=row[0], amount=row[1], zip=row[2], lat=row[3], long=row[4],
                cityPop=row[5], unixTime=row[6], merchLat=row[7], merchLong=row[8], isFraud=row[9]
            ).toArray()
            for row in rows
        ], dtype=object)

    async def getOrderedTrainingData(self):
        query = "SELECT * FROM train ORDER BY merchant, unix_time"
        rows = await self.db.fetch_all(query=query)
        return np.array([
            TargetedTransaction(
                merchId=row[0], amount=row[1], zip=row[2], lat=row[3], long=row[4],
                cityPop=row[5], unixTime=row[6], merchLat=row[7], merchLong=row[8], isFraud=row[9]
            ).toArray()
            for row in rows
        ], dtype=object)

    async def storeTransaction(self, transaction: TargetedTransaction):
        query = "INSERT INTO transactions (merchId, amount, zip, lat, long, cityPop, unixTime, merchLat, merchLong, " \
                "isFraud) VALUES (:merchId, :amount, :zip, :lat, :long, :cityPop, :unixTime, :merchLat, :merchLong, " \
                ":isFraud) "

        values = {
            "merchId": transaction.merchId,
            "amount": transaction.amount,
            "zip": transaction.zip,
            "lat": transaction.lat,
            "long": transaction.long,
            "cityPop": transaction.cityPop,
            "unixTime": transaction.unixTime,
            "merchLat": transaction.merchLat,
            "merchLong": transaction.merchLong,
            "isFraud": transaction.isFraud
        }

        await self.db.execute(query, values)
