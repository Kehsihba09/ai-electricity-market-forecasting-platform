import torch
import numpy as np

def monte_carlo_predict(

    model,

    input_tensor,

    num_samples=50
):

    model.train()

    predictions = []

    with torch.no_grad():

        for _ in range(num_samples):

            prediction = model(
                input_tensor
            )

            predictions.append(

                prediction
                .cpu()
                .numpy()
            )

    predictions = np.array(
        predictions
    )

    mean_prediction = np.mean(

        predictions,

        axis=0
    )

    std_prediction = np.std(

        predictions,

        axis=0
    )

    lower_bound = (

        mean_prediction
        -
        1.96 * std_prediction
    )

    upper_bound = (

        mean_prediction
        +
        1.96 * std_prediction
    )

    return {

        "mean_prediction":
            mean_prediction,

        "uncertainty":
            std_prediction,

        "lower_bound":
            lower_bound,

        "upper_bound":
            upper_bound
    }