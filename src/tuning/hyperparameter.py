import optuna
import xgboost as xgb

from sklearn.metrics import mean_absolute_error

from src.utils.logger import get_logger

logger = get_logger("optuna_tuning")

def optimize_xgboost(X_train,y_train,X_val,y_val,n_trials=20):

    def objective(trial):

        params = {

            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                1000
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.001,
                0.3,
                log=True
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.5,
                1.0
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.5,
                1.0
            )
        }

        model = xgb.XGBRegressor(
            **params
        )

        model.fit(
            X_train,
            y_train,
            verbose=False
        )

        predictions = model.predict(X_val)

        mae = mean_absolute_error(
            y_val,
            predictions
        )

        return mae

    study = optuna.create_study(
        direction="minimize"
    )

    study.optimize(
        objective,
        n_trials=n_trials
    )

    logger.info(
        f"Best parameters: "
        f"{study.best_params}"
    )

    return study.best_params
