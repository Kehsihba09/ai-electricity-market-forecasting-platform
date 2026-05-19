from src.models.xgboost_model import (
    XGBoostForecastModel
)

from src.models.lightgbm_model import (
    LightGBMForecastModel
)

from src.models.catboost_model import (
    CatBoostForecastModel
)

from src.models.random_forest_model import (
    RandomForestForecastModel
)

def get_model(model_name):

    models = {

        "xgboost":
            XGBoostForecastModel(),

        "lightgbm":
            LightGBMForecastModel(),

        "catboost":
            CatBoostForecastModel(),

        "random_forest":
            RandomForestForecastModel()
    }

    return models[model_name]