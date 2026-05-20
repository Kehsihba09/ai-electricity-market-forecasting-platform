import numpy as np

def calculate_confidence_interval(

    predictions,

    confidence=0.95
):

    predictions = np.array(
        predictions
    )

    mean = np.mean(
        predictions
    )

    std = np.std(
        predictions
    )

    z = 1.96

    lower = mean - z * std

    upper = mean + z * std

    return {

        "mean": mean,

        "lower_bound": lower,

        "upper_bound": upper
    }