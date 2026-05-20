import optuna
import pandas as pd

from src.features.feature_pipeline import (
    build_advanced_features
)

from src.deep_learning.gru_model import (
    GRUForecastModel
)

from src.deep_learning.trainer import (
    ForecastTrainer
)

from src.deep_learning.optimization_utils import (
    prepare_dataloaders
)

TARGET_COLUMN = "mcp_rs_mwh"

DATA_PATH = (
    "data/processed/final_processed_data.csv"
)

def objective(trial):

    # LOAD DATA

    df = pd.read_csv(DATA_PATH)

    df = build_advanced_features(

        df,

        TARGET_COLUMN
    )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    X = X.select_dtypes(
        include=["number"]
    )

    y = df[TARGET_COLUMN]

    # HYPERPARAMETERS

    hidden_size = trial.suggest_categorical(

        "hidden_size",

        [32, 64, 128]
    )

    num_layers = trial.suggest_int(

        "num_layers",

        1,

        3
    )

    learning_rate = trial.suggest_float(

        "learning_rate",

        1e-4,

        1e-2,

        log=True
    )

    sequence_length = (
        trial.suggest_categorical(

            "sequence_length",

            [12, 24, 48]
        )
    )

    # DATALOADERS

    train_loader, validation_loader = (
        prepare_dataloaders(

            X,

            y,

            sequence_length
        )
    )

    # MODEL

    model = GRUForecastModel(

        input_size=X.shape[1],

        hidden_size=hidden_size,

        num_layers=num_layers
    )

    # TRAINER

    trainer = ForecastTrainer(

        model=model,

        learning_rate=learning_rate,

        epochs=10
    )

    training_losses, validation_losses = (
        trainer.train(

            train_loader,

            validation_loader
        )
    )

    return validation_losses[-1]

def run_optimization():

    study = optuna.create_study(

        direction="minimize"
    )

    study.optimize(

        objective,

        n_trials=10
    )

    print("\nBEST PARAMETERS:\n")

    print(study.best_params)

    print("\nBEST VALIDATION LOSS:\n")

    print(study.best_value)

if __name__ == "__main__":

    run_optimization()