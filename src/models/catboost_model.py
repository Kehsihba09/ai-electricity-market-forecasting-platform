import joblib

from catboost import CatBoostRegressor

from src.models.base_model import (
    BaseForecastModel
)

class CatBoostForecastModel(
    BaseForecastModel
):

    def __init__(self):

        self.model = CatBoostRegressor(

            iterations=300,

            learning_rate=0.05,

            depth=6,

            verbose=False,

            random_seed=42
        )

    def train(
        self,
        X_train,
        y_train
    ):

        self.model.fit(
            X_train,
            y_train
        )

    def predict(
        self,
        X_test
    ):

        return self.model.predict(
            X_test
        )

    def save_model(
        self,
        path
    ):

        joblib.dump(
            self.model,
            path
        )