import json
import os
from datetime import datetime
from src.deep_learning.dl_leaderboard import (
    update_dl_leaderboard
)

EXPERIMENT_PATH = (
    "artifacts/dl_experiments.json"
)

def log_experiment(results):

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    results["timestamp"] = (
        str(datetime.now())
    )

    if os.path.exists(
        EXPERIMENT_PATH
    ):

        with open(
            EXPERIMENT_PATH,
            "r"
        ) as file:

            experiments = json.load(file)

    else:

        experiments = []

    experiments.append(results)

    with open(
        EXPERIMENT_PATH,
        "w"
    ) as file:

        json.dump(
            experiments,
            file,
            indent=4
        )

    update_dl_leaderboard({

    "model":
        results["model"],

    "validation_loss":
        results["final_validation_loss"],

    "train_loss":
        results["final_train_loss"],

    "timestamp":
        results["timestamp"]
    })

