import numpy as np

def calculate_prediction_risk(

    uncertainties
):

    mean_uncertainty = np.mean(
        uncertainties
    )

    if mean_uncertainty < 0.05:

        return "Low Risk"

    elif mean_uncertainty < 0.15:

        return "Moderate Risk"

    else:

        return "High Risk"

def volatility_score(

    predictions
):

    return np.std(predictions)