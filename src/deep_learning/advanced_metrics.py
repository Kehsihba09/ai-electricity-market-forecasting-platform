import numpy as np

def directional_accuracy(

    actual,

    predictions
):

    actual_direction = np.sign(
        np.diff(actual)
    )

    prediction_direction = np.sign(
        np.diff(predictions)
    )

    return np.mean(

        actual_direction
        ==
        prediction_direction

    ) * 100

def forecast_bias(

    actual,

    predictions
):

    return np.mean(
        predictions - actual
    )