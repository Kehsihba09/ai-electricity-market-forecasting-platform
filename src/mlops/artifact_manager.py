import os

def create_artifact_folders():

    folders = [

        "artifacts",

        "artifacts/models",

        "artifacts/plots",

        "artifacts/reports"
    ]

    for folder in folders:

        os.makedirs(
            folder,
            exist_ok=True
        )