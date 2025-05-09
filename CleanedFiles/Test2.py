from DbConnection import DbConnection, database
import asyncio
import aiohttp
from datetime import datetime

db = DbConnection(database)


async def sendMerchantReport(session, url, merchant: str, unix_time: int):
    date_str = datetime.utcfromtimestamp(unix_time).strftime('%d/%m/%Y')
    params = {"merchant": merchant, "date": date_str}

    async with session.get(url, params=params) as response:
        return await response.json()


async def sendAllMerchantReports(data, url):
    async with aiohttp.ClientSession() as session:
        tasks = [
            sendMerchantReport(session, url, merchant=row[0], unix_time=row[1])
            for row in data
        ]
        responses = await asyncio.gather(*tasks)
        return responses


async def main():
    try:
        await database.connect()

        # Modify query and indices as needed
        rows = await db.runQuery(0, "SELECT merchant, unix_time FROM train ORDER BY RAND() LIMIT 500;")

        responses = await sendAllMerchantReports(rows, "http://127.0.0.1:8000/get-merchant-report")

        for i, row in enumerate(rows):
            print(f"Input: merchant={row[0]}, unix={row[1]} => Response: {responses[i]}")

    except Exception as err:
        print("Error:", err)

    finally:
        await database.disconnect()


if __name__ == "__main__":
    asyncio.run(main())