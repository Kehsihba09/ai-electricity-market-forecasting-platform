import mlflow
import xgboost as xgb

from src.training.evaluate import (
evaluate_regression_model
)

from src.training.mlflow_tracker import (
initialize_experiment,
log_parameters,
log_metrics,
log_model
)

from src.training.save_model import save_model

from src.utils.logger import get_logger

logger = get_logger("xgboost_training")

def train_xgboost_model(X_train,y_train,X_val,y_val,X_test,y_test,params):

    initialize_experiment(
        "electricity_forecasting"
    )

    with mlflow.start_run():

        logger.info("Training XGBoost model.")

        model = xgb.XGBRegressor(
            **params
        )

        model.fit(
            X_train.values,
            y_train.values,
            eval_set=[
                (X_train.values, y_train.values),
                (X_val.values, y_val.values)
            ],
            verbose=False
        )

        predictions = model.predict(X_test.values)

        metrics = evaluate_regression_model(
            y_test,
            predictions
        )

        log_parameters(params)

        log_metrics(metrics)

        log_model(
            model,
            "xgboost_model"
        )

        save_model(
            model,
            "models/xgboost_model.pkl"
        )

        logger.info(
            "XGBoost training completed."
        )

        return model, metrics
