import torch
import torch.nn as nn

class TransformerForecastModel(nn.Module):

    def __init__(

        self,

        input_size,

        d_model=64,

        nhead=4,

        num_layers=2,

        dropout=0.1
    ):

        super().__init__()

        self.input_projection = nn.Linear(

            input_size,

            d_model
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(

                d_model=d_model,

                nhead=nhead,

                dropout=dropout,

                batch_first=True
            )
        )

        self.transformer_encoder = (
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

        attention_output = (
            self.transformer_encoder(x)
        )

        output = self.fc(

            attention_output[:, -1, :]
        )

        return output