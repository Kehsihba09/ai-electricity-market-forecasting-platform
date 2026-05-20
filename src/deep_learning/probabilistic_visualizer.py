import matplotlib.pyplot as plt
import numpy as np

def plot_probabilistic_forecast(

    actual,

    mean_prediction,

    lower_bound,

    upper_bound
):

    x_axis = np.arange(len(actual))

    plt.figure(figsize=(12, 6))

    plt.plot(

        x_axis,

        actual,

        label="Actual"
    )

    plt.plot(

        x_axis,

        mean_prediction,

        label="Prediction"
    )

    plt.fill_between(

        x_axis,

        lower_bound.flatten(),

        upper_bound.flatten(),

        alpha=0.3,

        label="Confidence Interval"
    )

    plt.title(
        "Probabilistic Forecast"
    )

    plt.xlabel("Time")

    plt.ylabel("Price")

    plt.legend()

    plt.grid(True)

    plt.show()