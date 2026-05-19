import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from src.models.model_factory import (
    get_model
)

from src.features.feature_pipeline import (
    build_advanced_features
)

from src.mlops.evaluator import (
    evaluate_forecasting_model
)

from src.mlops.leaderboard import (
    update_leaderboard
)

from src.mlops.model_registry import (
    register_model
)

from src.mlops.metadata_tracker import (
    save_run_metadata
)

from src.mlops.artifact_manager import (
    create_artifact_folders
)

from src.utils.logger import get_logger

logger = get_logger(
    "central_training_engine"
)

TARGET_COLUMN = "mcv"

DATA_PATH = (
    "data/processed/final_processed_data.csv"
)

def train_model(model_name):

    logger.info(
        f"Starting training for {model_name}"
    )

    create_artifact_folders()

    # LOAD DATA

    df = pd.read_csv(DATA_PATH)

    # FEATURE ENGINEERING

    df = build_advanced_features(
        df,
        TARGET_COLUMN
    )

    # SPLIT FEATURES

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,

            y,

            test_size=0.2,

            shuffle=False
        )
    )

    # LOAD MODEL

    model = get_model(model_name)

    # TRAIN

    model.train(
        X_train,
        y_train
    )

    # PREDICT

    predictions = model.predict(
        X_test
    )

    # EVALUATE

    metrics = evaluate_forecasting_model(

        y_test,

        predictions
    )

    logger.info(
        f"Metrics: {metrics}"
    )

    # SAVE MODEL

    model_path = (
        f"artifacts/models/"
        f"{model_name}.pkl"
    )

    model.save_model(model_path)

    # UPDATE LEADERBOARD

    leaderboard_entry = {

        "model_name": model_name,

        "MAE": metrics["MAE"],

        "RMSE": metrics["RMSE"],

        "R2": metrics["R2"]
    }

    update_leaderboard(
        leaderboard_entry
    )

    # REGISTER MODEL

    register_model(

        model_name=model_name,

        version="v1",

        metrics=metrics,

        features=list(X.columns),

        artifact_path=model_path
    )

    # SAVE METADATA

    save_run_metadata({

        "model_name": model_name,

        "features_used":
            list(X.columns),

        "training_rows":
            len(X_train),

        "testing_rows":
            len(X_test)
    })

    logger.info(
        f"{model_name} training complete."
    )

if __name__ == "__main__":

    MODELS = [

        "xgboost",

        "lightgbm",

        "catboost",

        "random_forest"
    ]

    for model_name in MODELS:

        train_model(model_name)