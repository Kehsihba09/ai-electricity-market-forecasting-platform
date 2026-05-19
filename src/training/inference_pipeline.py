import joblib
import pandas as pd

MODEL_PATH = (
    "artifacts/models/xgboost.pkl"
)

model = joblib.load(MODEL_PATH)

def predict(input_df):

    prediction = model.predict(
        input_df
    )

    return prediction