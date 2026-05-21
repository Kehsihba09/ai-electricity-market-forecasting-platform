import numpy as np

from src.ensemble.ensemble_predictor import (
    EnsemblePredictor
)

from src.ensemble.uncertainty_aggregation import (
    aggregate_uncertainty,

    disagreement_score
)

from src.ensemble.ensemble_signal import (
    ensemble_decision
)

def run_ensemble_system():

    predictions = {

        "gru": 5.4,

        "lstm": 5.5,

        "transformer": 5.7
    }

    uncertainties = [

        0.08,

        0.06,

        0.10
    ]

    ensemble_model = (
        EnsemblePredictor()
    )

    ensemble_prediction = (
        ensemble_model.weighted_average(

            predictions
        )
    )

    total_uncertainty = (
        aggregate_uncertainty(

            uncertainties
        )
    )

    disagreement = (
        disagreement_score(

            list(predictions.values())
        )
    )

    signal = ensemble_decision(

        list(predictions.values())
    )

    print("\nENSEMBLE FORECASTING\n")

    print(
        "Ensemble Prediction:",
        ensemble_prediction
    )

    print(
        "Aggregated Uncertainty:",
        total_uncertainty
    )

    print(
        "Model Disagreement:",
        disagreement
    )

    print(
        "Trading Signal:",
        signal
    )

if __name__ == "__main__":

    run_ensemble_system()