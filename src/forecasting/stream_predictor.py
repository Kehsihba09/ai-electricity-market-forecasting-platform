import joblib
import pandas as pd

from src.feature_engineering.feature_pipeline import (build_features)

from src.utils.logger import get_logger

logger = get_logger("stream_predictor")

model = joblib.load("models/xgboost_model.pkl")

def predict_stream_event(state):

    df = pd.DataFrame(state)

    df = build_features(df)

    latest_row = df.iloc[-1:]

    prediction = model.predict(latest_row)[0]

    logger.info(
        f"Streaming prediction: {prediction}"
    )

    return prediction