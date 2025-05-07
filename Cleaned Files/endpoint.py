# To run this file, use the command:
# fastapi dev endpoint.py
from DbConnection import DbConnection, database
from fastapi.responses import JSONResponse
from ModelParameters import ModelParams
from Transaction import Transaction
from pydantic import BaseModel
from Model import undersample
from fastapi import FastAPI
from Models import getModel
from Model import Model
import anyio.to_thread
import asyncio
import joblib

# Create Endpoint on 127.0.0.1:8000
# Create app instance
app = FastAPI()
exclusiveLock = asyncio.Lock()

db = DbConnection(database)
model: Model = None


class Report(BaseModel):
    message: str

def loadModel(filepath):
    global model
    model: Model = joblib.load(filepath)

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

        del trainingData
        del testData

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


@app.get("/get-merchant-report")
async def get_report() -> Report:
    async with exclusiveLock:
        return Report(message="all good")


@app.post("/check-fraudulent-status")
async def check_transaction(transaction: Transaction):
    async with exclusiveLock:
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


@app.get("/get-confusion-matrix")
async def getCM():
    async with exclusiveLock:
        await anyio.to_thread.run_sync(model.getCM())
        return {"message": "Confusion matrix retrieved"}


@app.post("/load-model")
async def load(filepath: str):
    async with exclusiveLock:
        await anyio.to_thread.run_sync(loadModel(filepath))
        return {"message": "Model loaded"}


@app.post("/save-model")
async def saveModel():
    async with exclusiveLock:
        model.saveModel()
