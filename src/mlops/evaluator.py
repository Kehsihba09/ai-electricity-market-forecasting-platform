from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score
)

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("evaluator")

def evaluate_forecasting_model(

    y_true,

    predictions
):

    logger.info(
        "Evaluating forecasting model."
    )

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    metrics = {

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2
    }

    logger.info(
        f"Evaluation metrics: {metrics}"
    )

    return metrics