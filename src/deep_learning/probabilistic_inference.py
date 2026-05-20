import torch
import pandas as pd

from src.deep_learning.transformer_model import (
    TransformerForecastModel
)

from src.deep_learning.monte_carlo_forecasting import (
    monte_carlo_predict
)

from src.deep_learning.risk_metrics import (
    calculate_prediction_risk

)

MODEL_PATH = (
    "artifacts/checkpoints/"
    "forecast_model.pt"
)

def run_probabilistic_inference():

    model = TransformerForecastModel(

    input_size=37,

    d_model=128,

    nhead=4,

    num_layers=3,

    dropout=0.24562126161268596
    )

    model.load_state_dict(

        torch.load(MODEL_PATH)
    )

    model.eval()

    dummy_input = torch.randn(

        1,

        12,

        37
    )

    results = monte_carlo_predict(

        model,

        dummy_input
    )

    risk_level = (
        calculate_prediction_risk(

            results["uncertainty"]
        )
    )

    print("\nPROBABILISTIC FORECAST\n")

    print(
        "Mean Prediction:\n",

        results["mean_prediction"]
    )

    print(
        "\nUncertainty:\n",

        results["uncertainty"]
    )

    print(
        "\nRisk Level:\n",

        risk_level
    )

if __name__ == "__main__":

    run_probabilistic_inference()