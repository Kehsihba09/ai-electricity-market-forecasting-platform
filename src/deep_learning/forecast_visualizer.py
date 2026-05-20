import matplotlib.pyplot as plt

def plot_forecasts(

    actual,

    predictions,

    model_name="Forecast Model"
):

    plt.figure(figsize=(12, 6))

    plt.plot(
        actual,
        label="Actual"
    )

    plt.plot(
        predictions,
        label="Predicted"
    )

    plt.title(
        f"{model_name} Forecast"
    )

    plt.xlabel("Time")

    plt.ylabel("Price")

    plt.legend()

    plt.grid(True)

    plt.show()