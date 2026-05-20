import os
import torch

def save_checkpoint(

    model,

    path
):

    os.makedirs(
        "artifacts/checkpoints",
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        path
    )