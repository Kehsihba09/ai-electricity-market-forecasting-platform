import torch
import os

def save_checkpoint(

    model,

    model_name
):

    os.makedirs(

        "artifacts/checkpoints",

        exist_ok=True
    )

    save_path = (

        f"artifacts/checkpoints/"
        f"{model_name}.pt"
    )

    torch.save(

        model.state_dict(),

        save_path
    )

    print(
        f"Checkpoint saved: {save_path}"
    )

    return save_path