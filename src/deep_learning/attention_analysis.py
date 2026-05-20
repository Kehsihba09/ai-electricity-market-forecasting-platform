import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(

    attention_matrix,

    title="Attention Map"
):

    plt.figure(figsize=(10, 8))

    plt.imshow(

        attention_matrix,

        aspect="auto"
    )

    plt.colorbar()

    plt.title(title)

    plt.xlabel("Sequence Position")

    plt.ylabel("Attention Weight")

    plt.show()