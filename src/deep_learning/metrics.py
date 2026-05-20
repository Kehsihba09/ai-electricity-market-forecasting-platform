import numpy as np

def calculate_metrics(

    y_true,

    predictions
):

    y_true = np.array(y_true)

    predictions = np.array(predictions)

    mae = np.mean(
        np.abs(y_true - predictions)
    )

    rmse = np.sqrt(
        np.mean(
            (y_true - predictions) ** 2
        )
    )

    mape = np.mean(

        np.abs(
            (y_true - predictions)
            / (y_true + 1e-6)
        )

    ) * 100

    return {

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape
    }