import joblib

from sklearn.ensemble import (
    RandomForestRegressor
)

from src.models.base_model import (
    BaseForecastModel
)

class RandomForestForecastModel(
    BaseForecastModel
):

    def __init__(self):

        self.model = RandomForestRegressor(

            n_estimators=200,

            max_depth=10,

            random_state=42,

            n_jobs=-1
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