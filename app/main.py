import time

from fastapi import FastAPI

from app.schemas import (
ForecastRequest,
ForecastResponse
)

from app.model_loader import load_model

from app.utils import prepare_features

from src.utils.logger import get_logger

logger = get_logger("fastapi_app")

app = FastAPI(
title="Electricity Forecasting API",
version="1.0"
)

model = load_model()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/forecast",response_model=ForecastResponse)

def forecast(request: ForecastRequest):

    start_time = time.time()

    features = prepare_features(request)

    prediction = model.predict(features)[0]

    latency = (
        time.time() - start_time
    ) * 1000

    logger.info(
        f"Prediction generated "
        f"in {latency:.2f} ms"
    )

    return ForecastResponse(
        forecast_price=float(prediction)
    )

@app.get("/")
def root():

    return {
        "message":
        "AI Electricity Forecasting API Running"
    }
