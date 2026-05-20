import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from src.features.feature_pipeline import (
    build_advanced_features
)

from src.deep_learning.dataloader import (
    create_dataloader
)

from src.deep_learning.lstm_model import (
    LSTMForecastModel
)

from src.deep_learning.trainer import (
    ForecastTrainer
)

from src.deep_learning.experiment_tracker import (
    log_experiment
)

TARGET_COLUMN = "mcv"

DATA_PATH = (
    "data/processed/final_processed_data.csv"
)

def run_lstm_experiment():

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

    train_loader = create_dataloader(

        X_train,

        y_train
    )

    validation_loader = create_dataloader(

        X_test,

        y_test
    )

    model = LSTMForecastModel(

        input_size=X.shape[1]
    )

    trainer = ForecastTrainer(
        model=model
    )

    training_losses, validation_losses = (
        trainer.train(

            train_loader,

            validation_loader
        )
    )

    results = {

        "model": "LSTM",

        "final_train_loss":
            training_losses[-1],

        "final_validation_loss":
            validation_losses[-1]
    }

    log_experiment(results)

if __name__ == "__main__":

    run_lstm_experiment()