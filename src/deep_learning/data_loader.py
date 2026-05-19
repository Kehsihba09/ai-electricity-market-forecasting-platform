from torch.utils.data import DataLoader

from src.deep_learning.dataset import (
    TimeSeriesDataset
)

def create_dataloader(

    X,

    y,

    batch_size=32,

    sequence_length=24
):

    dataset = TimeSeriesDataset(

        X,

        y,

        sequence_length
    )

    dataloader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False
    )

    return dataloader