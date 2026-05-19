import json
import os
from datetime import datetime

METADATA_PATH = (
    "artifacts/run_metadata.json"
)

def save_run_metadata(metadata):

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    metadata["timestamp"] = (
        str(datetime.now())
    )

    with open(
        METADATA_PATH,
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )