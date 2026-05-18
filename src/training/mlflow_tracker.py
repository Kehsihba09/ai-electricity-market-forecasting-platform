import mlflow
from mlflow import metrics
from src.utils.logger import get_logger

logger = get_logger("mlflow_tracker")

def initialize_experiment(experiment_name):

    mlflow.set_experiment(experiment_name)

    logger.info(
        f"MLflow experiment initialized: "
        f"{experiment_name}"
    )

def log_parameters(params: dict):

    mlflow.log_params(params)

def log_metrics(metrics: dict):

    mlflow.log_metrics(metrics)

def log_model(model, artifact_path):

    mlflow.sklearn.log_model(
        model,
        artifact_path
    )

    logger.info("Model logged successfully.")
