import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from src.models.model_factory import (
    get_model
)

from src.models.ensemble_model import (
    EnsembleForecastModel
)

from src.features.feature_pipeline import (
    build_advanced_features
)

from src.mlops.evaluator import (
    evaluate_forecasting_model
)

from src.utils.logger import get_logger

logger = get_logger(
    "ensemble_training"
)

TARGET_COLUMN = "mcv"

DATA_PATH = (
    "data/processed/final_processed_data.csv"
)

def train_ensemble():

    logger.info(
        "Training ensemble model."
    )

    df = pd.read_csv(DATA_PATH)

    df = build_advanced_features(
        df,
        TARGET_COLUMN
    )

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

    model_names = [

        "xgboost",

        "lightgbm",

        "random_forest"
    ]

    trained_models = []

    for model_name in model_names:

        model = get_model(model_name)

        model.train(
            X_train,
            y_train
        )

        trained_models.append(model)

    ensemble_model = (
        EnsembleForecastModel(
            trained_models
        )
    )

    predictions = ensemble_model.predict(
        X_test
    )

    metrics = evaluate_forecasting_model(

        y_test,

        predictions
    )

    logger.info(
        f"Ensemble metrics: {metrics}"
    )

if __name__ == "__main__":

    train_ensemble()