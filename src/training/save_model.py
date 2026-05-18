import joblib

from src.utils.logger import get_logger

logger = get_logger("model_saver")

def save_model(model,save_path):
    joblib.dump(model, save_path)

    logger.info(
        f"Model saved successfully at {save_path}"
    )
