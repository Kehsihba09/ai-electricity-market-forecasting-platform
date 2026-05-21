import torch
import torch.nn as nn

from src.deep_learning.metrics import (
    calculate_metrics
)

from src.deep_learning.early_stopping import (
    EarlyStopping
)

from src.deep_learning.checkpoint_manager import (
    save_checkpoint
)

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class ForecastTrainer:

    def __init__(

        self,

        model,

        learning_rate=0.001,

        epochs=20
    ):

        self.model = model.to(device)

        self.epochs = epochs

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(

            self.model.parameters(),

            lr=learning_rate
        )

        self.early_stopping = (
            EarlyStopping(
                patience=5
            )
        )

    def train(

        self,

        train_loader,

        validation_loader
    ):

        training_losses = []

        validation_losses = []

        for epoch in range(self.epochs):

            self.model.train()

            train_loss = 0

            for X_batch, y_batch in train_loader:

                X_batch = X_batch.to(
                    self.device
                )

                y_batch = y_batch.to(
                    self.device
                )

                predictions = self.model(
                    X_batch
                ).squeeze()

                loss = self.criterion(

                    predictions,

                    y_batch
                )

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = (

                train_loss
                / len(train_loader)
            )

            training_losses.append(
                avg_train_loss
            )

            # VALIDATION

            self.model.eval()

            validation_loss = 0

            all_predictions = []

            all_targets = []

            with torch.no_grad():

                for X_batch, y_batch in validation_loader:

                    X_batch = X_batch.to(
                        self.device
                    )

                    y_batch = y_batch.to(
                        self.device
                    )

                    predictions = self.model(
                        X_batch
                    ).squeeze()

                    loss = self.criterion(

                        predictions,

                        y_batch
                    )

                    validation_loss += (
                        loss.item()
                    )

                    all_predictions.extend(

                        predictions.cpu().numpy()
                    )

                    all_targets.extend(

                        y_batch.cpu().numpy()
                    )

            avg_validation_loss = (

                validation_loss
                / len(validation_loader)
            )

            validation_losses.append(
                avg_validation_loss
            )

            metrics = calculate_metrics(

                all_targets,

                all_predictions
            )

            print(

                f"Epoch {epoch+1}/{self.epochs} | "

                f"Train Loss: {avg_train_loss:.4f} | "

                f"Validation Loss: {avg_validation_loss:.4f} | "

                f"MAE: {metrics['MAE']:.4f}"
            )

            # EARLY STOPPING

            self.early_stopping.update(
                avg_validation_loss
            )

            if self.early_stopping.should_stop:

                print(
                    "Early stopping triggered."
                )

                break

        save_checkpoint(

            self.model,

            "artifacts/checkpoints/"
            "forecast_model.pt"
        )

        return (

            training_losses,

            validation_losses
        )