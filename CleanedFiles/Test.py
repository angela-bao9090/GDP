from Transaction import TargetedTransaction, Transaction
from DbConnection import DbConnection, database
import asyncio
import aiohttp
import time

db = DbConnection(database)


async def sendTransaction(session, url, transaction: Transaction):
    async with session.post(url, json=transaction.model_dump()) as response:
        if response.status != 200:
            print(f"Request failed with status: {response.status}")
            print(await response.text())
        return await response.json()


async def sendAllTransactions(transactions, url):
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [sendTransaction(session, url, transaction) for transaction in transactions]
        result = await asyncio.gather(*tasks)
        end_time = time.time()
        duration = end_time - start_time
        print(f"All requests took {duration:.4f} seconds to complete.")
        return result


async def main():
    try:
        await database.connect()
        rows = await db.runQuery(0, "SELECT * FROM test ORDER BY RAND() LIMIT 500;")
        transactions = [
            TargetedTransaction(
                merchId=row[0], amount=row[1], zip=row[2], lat=row[3], long=row[4],
                cityPop=row[5], unixTime=row[6], merchLat=row[7], merchLong=row[8], isFraud=row[9]
            )
            for row in rows
        ]

        responses = await sendAllTransactions(transactions, "http://127.0.0.1:8000/check-fraudulent-status")

        # for i, row in enumerate(transactions):
        #     print("Input", row.toArray(), " : Output ", responses[i])

        correctCount = 0
        for i, row in enumerate(transactions):
            if (row.toArray()[-1] == 1 and responses[i] == {'fraudStatus': 'Fraud'}) or \
                    (row.toArray()[-1] == 0 and responses[i] == {'fraudStatus': 'Not Fraud'}):
                correctCount += 1
        print(correctCount, " correct out of ", len(transactions))

    except Exception as err:
        print(err)

    finally:
        await database.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
