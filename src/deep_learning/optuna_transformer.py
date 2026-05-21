import optuna
import pandas as pd

from src.features.feature_pipeline import (
    build_advanced_features
)

from src.deep_learning.transformer_model import (
    TransformerForecastModel
)

from src.deep_learning.trainer import (
    ForecastTrainer
)

from src.deep_learning.optimization_utils import (
    prepare_dataloaders
)

from src.mlops.model_registry import (
    register_model
)

from src.mlops.config_tracker import (
    save_model_config
)

from src.mlops.checkpoint_manager import (
    save_checkpoint
)

TARGET_COLUMN = "mcp_rs_mwh"

DATA_PATH = (
    "data/processed/final_processed_data.csv"
)

def objective(trial):

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

    # TRANSFORMER PARAMETERS

    d_model = trial.suggest_categorical(

        "d_model",

        [32, 64, 128]
    )

    nhead = trial.suggest_categorical(

        "nhead",

        [2, 4]
    )

    num_layers = trial.suggest_int(

        "num_layers",

        1,

        3
    )

    dropout = trial.suggest_float(

        "dropout",

        0.1,

        0.4
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

            [12, 24]
        )
    )

    train_loader, validation_loader = (
        prepare_dataloaders(

            X,

            y,

            sequence_length
        )
    )

    model = TransformerForecastModel(

        input_size=X.shape[1],

        d_model=d_model,

        nhead=nhead,

        num_layers=num_layers,

        dropout=dropout
    )

    trainer = ForecastTrainer(

        model=model,

        learning_rate=learning_rate,

        epochs=5
    )

    _, validation_losses = (
        trainer.train(

            train_loader,

            validation_loader
        )
    )

    return validation_losses[-1]

def run_transformer_optimization():

    study = optuna.create_study(

        direction="minimize"
    )

    study.optimize(

        objective,

        n_trials=5
    )

    print("\nBEST TRANSFORMER PARAMETERS:\n")

    print(study.best_params)

    best_config = {

    "d_model":
        study.best_params["d_model"],

    "nhead":
        study.best_params["nhead"],

    "num_layers":
        study.best_params["num_layers"],

    "dropout":
        study.best_params["dropout"],

    "learning_rate":
        study.best_params["learning_rate"],

    "sequence_length":
        study.best_params["sequence_length"]
    }

    save_model_config(

    best_config,

    "best_transformer"
    )

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

    best_model = TransformerForecastModel(

        input_size=X.shape[1],

        d_model=study.best_params["d_model"],

        nhead=study.best_params["nhead"],

        num_layers=study.best_params["num_layers"],

        dropout=study.best_params["dropout"]
    )

    checkpoint_path = save_checkpoint(

    best_model,

    "best_transformer"
    )
    register_model(

    model_name="best_transformer",

    model_type="Transformer",

    validation_loss=study.best_value,

    hyperparameters=best_config,

    checkpoint_path=checkpoint_path
    )
    print("\nBEST VALIDATION LOSS:\n")

    print(study.best_value)

if __name__ == "__main__":

    run_transformer_optimization()

    