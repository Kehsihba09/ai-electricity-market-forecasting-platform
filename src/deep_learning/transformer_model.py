import torch
import torch.nn as nn

from src.deep_learning.positional_encoding import (
    PositionalEncoding
)

class TransformerForecastModel(nn.Module):

    def __init__(

        self,

        input_size,

        d_model=64,

        nhead=4,

        num_layers=2
    ):

        super().__init__()

        self.input_projection = nn.Linear(

            input_size,

            d_model
        )

        self.positional_encoding = (
            PositionalEncoding(
                d_model
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(

                d_model=d_model,

                nhead=nhead,

                batch_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(

                encoder_layer,

                num_layers=num_layers
            )
        )

        self.fc = nn.Linear(
            d_model,
            1
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.positional_encoding(x)

        output = self.transformer(x)

        output = output[:, -1, :]

        output = self.fc(output)

        return output