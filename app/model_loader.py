import joblib

from src.utils.logger import get_logger

logger = get_logger("model_loader")

MODEL_PATH = "models/xgboost_model.pkl"

def load_model():

    logger.info(
        f"Loading model from {MODEL_PATH}"
    )

    model = joblib.load(MODEL_PATH)

    logger.info("Model loaded successfully.")

    return model
