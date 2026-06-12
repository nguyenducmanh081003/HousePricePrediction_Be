from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(
    title="House Price Prediction API",
    description="Machine Learning API",
    version="1.0.0"
)

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")


class HouseRequest(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def home():
    return {
        "project": "House Price Prediction API",
        "status": "running"
    }


@app.post("/predict")
def predict(data: HouseRequest):

    features = np.array([[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    return {
        "success": True,
        "predicted_price_usd": round(prediction * 100000, 2)
    }