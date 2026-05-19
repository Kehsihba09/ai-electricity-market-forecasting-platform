import joblib

from xgboost import XGBRegressor

from src.models.base_model import (
    BaseForecastModel
)

class XGBoostForecastModel(
    BaseForecastModel
):

    def __init__(self):

        self.model = XGBRegressor(

            n_estimators=300,

            learning_rate=0.05,

            max_depth=6,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=42
        )

    def train(
        self,
        X_train,
        y_train
    ):

        self.model.fit(
            X_train.values,
            y_train.values
        )

    def predict(
        self,
        X_test
    ):

        return self.model.predict(
            X_test.values
        )

    def save_model(
        self,
        path
    ):

        joblib.dump(
            self.model,
            path
        )