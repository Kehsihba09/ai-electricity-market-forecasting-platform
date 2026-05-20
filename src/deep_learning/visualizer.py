import matplotlib.pyplot as plt

def plot_training_curves(

    training_losses,

    validation_losses
):

    plt.figure(figsize=(10, 5))

    plt.plot(
        training_losses,
        label="Train Loss"
    )

    plt.plot(
        validation_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "Training Curves"
    )

    plt.legend()

    plt.grid(True)

    plt.show()