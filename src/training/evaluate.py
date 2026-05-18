import numpy as np

from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

from src.utils.logger import get_logger

logger = get_logger("model_evaluation")

def calculate_mape(y_true, y_pred):

    return np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100

def evaluate_regression_model(y_true,y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    mape = calculate_mape(y_true, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }

    logger.info(f"Evaluation metrics: {metrics}")

    return metrics
