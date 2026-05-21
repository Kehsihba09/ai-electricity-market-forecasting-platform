import numpy as np

def aggregate_uncertainty(

    uncertainties
):

    return np.mean(
        uncertainties
    )

def disagreement_score(

    predictions
):

    return np.std(
        predictions
    )