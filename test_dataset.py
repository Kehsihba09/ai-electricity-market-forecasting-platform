import pandas as pd
import numpy as np

from src.deep_learning.dataset import (
    TimeSeriesDataset
)

X = pd.DataFrame(
    np.random.rand(100, 10)
)

y = pd.Series(
    np.random.rand(100)
)

dataset = TimeSeriesDataset(
    X,
    y,
    sequence_length=24
)

print(len(dataset))

sample_X, sample_y = dataset[0]

print(sample_X.shape)

print(sample_y.shape)