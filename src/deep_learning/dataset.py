import torch

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        sequence_length=24
    ):

        self.X = X.values
        self.y = y.values

        self.sequence_length = (
            sequence_length
        )

    def __len__(self):

        return (
            len(self.X)
            - self.sequence_length
        )

    def __getitem__(self, idx):

        X_sequence = self.X[
            idx:
            idx + self.sequence_length
        ]

        y_target = self.y[
            idx + self.sequence_length
        ]

        return (

            torch.tensor(
                X_sequence,
                dtype=torch.float32
            ),

            torch.tensor(
                y_target,
                dtype=torch.float32
            )
        )