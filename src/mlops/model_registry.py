import os
import json
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger("model_registry")

REGISTRY_PATH = "artifacts/model_registry.json"

def register_model(

    model_name,

    version,

    metrics,

    features,

    artifact_path
):

    logger.info(
        f"Registering model: {model_name}"
    )

    model_entry = {

        "model_name": model_name,

        "version": version,

        "timestamp": str(datetime.now()),

        "metrics": metrics,

        "features": features,

        "artifact_path": artifact_path
    }

    if os.path.exists(REGISTRY_PATH):

        with open(
            REGISTRY_PATH,
            "r"
        ) as file:

            registry = json.load(file)

    else:

        registry = []

    registry.append(model_entry)

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    with open(
        REGISTRY_PATH,
        "w"
    ) as file:

        json.dump(
            registry,
            file,
            indent=4
        )

    logger.info(
        "Model registered successfully."
    )