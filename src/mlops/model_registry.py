import json
import os
from datetime import datetime

REGISTRY_PATH = (
    "artifacts/model_registry.json"
)

def register_model(

    model_name,

    model_type,

    validation_loss,

    hyperparameters,

    checkpoint_path
):

    model_entry = {

        "model_name":
            model_name,

        "model_type":
            model_type,

        "validation_loss":
            validation_loss,

        "hyperparameters":
            hyperparameters,

        "checkpoint_path":
            checkpoint_path,

        "registered_at":
            str(datetime.now())
    }

    if os.path.exists(
        REGISTRY_PATH
    ):

        with open(

            REGISTRY_PATH,

            "r"
        ) as f:

            registry = json.load(f)

    else:

        registry = []

    registry.append(model_entry)

    with open(

        REGISTRY_PATH,

        "w"
    ) as f:

        json.dump(

            registry,

            f,

            indent=4
        )

    print(
        "Model registered successfully."
    )