# To run this file, use the command:
# fastapi dev endpoint.py
from DbConnection import DbConnection, database
from fastapi.responses import JSONResponse
from Transaction import Transaction
from pydantic import BaseModel
from Model import undersample
from fastapi import FastAPI
from Models import getModel


# Create Endpoint on 127.0.0.1:8000
# Create app instance
app = FastAPI()

db = DbConnection(database)
model = None


class Report(BaseModel):
    message: str


@app.on_event("startup")
async def startup():
    global model
    try:
        await database.connect()
        print("Database connection successful")

        print("Waiting to receive data from database")
        trainingData = undersample([x[1:] for x in await db.getTrainingData()])
        testData = [x[1:] for x in await db.getTestData()]
        print("Data Recieved")

        model = getModel()
        model.commenceTraining(trainingData, testData)
        model.saveModel()

    except Exception as err:
        print(f"Error during startup: {err}")
        raise err


@app.on_event("shutdown")
async def shutdown():
    global db
    try:
        await database.disconnect()
        print("Database disconnected successfully.")
    except Exception as err:
        print(f"Shutdown error: {err}")


@app.get("/get-daily-report")
async def get_report() -> Report:
    return Report(message="all good")


@app.post("/check-fraudulent-status")
async def check_transaction(transaction: Transaction):
    model.loadTargeted([transaction.toArray()[1:]])
    model.test()
    status = "Fraud" if model.yPred[0] == 1 else "Not Fraud"
    model.resetStored()

    # try:
    #     db.storeTransaction(transaction)
    #
    # except Exception as err:
    #     print(f"Error during startup: {err}")
    #     raise err

    return JSONResponse(content={"fraudStatus": status})
