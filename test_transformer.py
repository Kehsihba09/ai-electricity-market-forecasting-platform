import torch

from src.deep_learning.transformer_model import (
    TransformerForecastModel
)

model = TransformerForecastModel(
    input_size=10
)

sample = torch.randn(
    32,
    24,
    10
)

output = model(sample)

print(output.shape)
