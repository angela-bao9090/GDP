# To run this file, use the command:
# fastapi dev endpoint.py
from IsolationForestModel import IsolationForestModel
from DbConnection import DbConnection, database
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from Transaction import Transaction
from Model import undersample
from Models import getModel
from Model import Model
from Lock import Lock
import datetime
import joblib
import anyio
import copy
import sys

# Create Endpoint on 127.0.0.1:8000
# Create app instance
app = FastAPI()

modelLock = Lock()
forestLock = Lock()

db = DbConnection(database)
model: Model = None
isolationForest: IsolationForestModel = None


@app.on_event("startup")
async def startup():
    global model, isolationForest
    try:
        await database.connect()
        print("Database connection successful")
        print("Waiting to receive data from database")
        forestTrainingData = await db.getOrderedTrainingData()
        modelTrainingData = undersample([x[1:] for x in forestTrainingData])
        print("Data Received1")
        testData = [x[1:] for x in await db.getTestData()]
        print("Data Received")

        model = getModel()
        model.commenceTraining(modelTrainingData, testData)

        del modelTrainingData

        isolationForest = IsolationForestModel(model.getModel(), model)
        isolationForest.buildForest(forestTrainingData)

    except Exception as err:
        print(f"Startup error: {err}")
        shutdown()
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown():
    global db
    try:
        await database.disconnect()
        print("Database disconnected successfully.")
    except Exception as err:
        print(f"Shutdown error: {err}")


@app.get("/get-merchant-report")
async def get_report(merchant: str, date: str):
    await forestLock.acquirePassiveLock()
    try:
        dateTime = datetime.datetime.strptime(date, "%d/%m/%Y")
        status = isolationForest.getMerchReport(merchant, int(dateTime.timestamp()))
        return JSONResponse(content={"fraudStatus": status})
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use DD/MM/YYYY.")
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {err}")
    finally:
        await forestLock.releasePassiveLock()


@app.post("/check-fraudulent-status")
async def check_transaction(transaction: Transaction):
    await modelLock.acquirePassiveLock()
    try:
        # modelCopy = copy.deepcopy(model)
        # status = modelCopy.testTransaction(transaction.toArray()[1:])
        status = model.testTransaction(transaction.toArray()[1:])
        return JSONResponse(content={"fraudStatus": status})
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error checking transaction: {err}")
    finally:
        await modelLock.releasePassiveLock()


@app.post("/load-model")
async def loadModel(filepath: str):
    await modelLock.acquireActiveLock()
    try:
        global model
        model = await anyio.to_thread.run_sync(joblib.load, filepath)
        return {"message": "Model loaded"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found.")
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {err}")
    finally:
        await modelLock.releaseActiveLock()


@app.post("/load-forest")
async def loadForest(filepath: str):
    await forestLock.acquireActiveLock()
    try:
        global isolationForest
        isolationForest = await anyio.to_thread.run_sync(joblib.load, filepath)
        return {"message": "Forest loaded"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Forest file not found.")
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to load forest: {err}")
    finally:
        await forestLock.releaseActiveLock()


@app.post("/save-model")
async def saveModel():
    await modelLock.acquirePassiveLock()
    try:
        model.saveModel()
        return {"message": "Model Saved"}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to save model: {err}")
    finally:
        await modelLock.releasePassiveLock()


@app.post("/save-forest")
async def saveForest():
    await forestLock.acquirePassiveLock()
    try:
        isolationForest.saveForest()
        return {"message": "Forest Saved"}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to save forest: {err}")
    finally:
        await forestLock.releasePassiveLock()
