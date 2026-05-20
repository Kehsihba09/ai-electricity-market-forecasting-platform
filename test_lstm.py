import torch

from src.deep_learning.lstm_model import (
    LSTMForecastModel
)

model = LSTMForecastModel(
    input_size=10
)

sample = torch.randn(
    32,
    24,
    10
)

output = model(sample)

print(output.shape)