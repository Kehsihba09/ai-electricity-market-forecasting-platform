import numpy as np

class EnsemblePredictor:

    def __init__(

        self,

        model_weights=None
    ):

        if model_weights is None:

            model_weights = {

                "gru": 0.33,

                "lstm": 0.33,

                "transformer": 0.34
            }

        self.model_weights = (
            model_weights
        )

    def weighted_average(

        self,

        predictions
    ):

        ensemble_prediction = 0

        for model_name, prediction in (
            predictions.items()
        ):

            ensemble_prediction += (

                prediction
                *
                self.model_weights[
                    model_name
                ]
            )

        return ensemble_prediction