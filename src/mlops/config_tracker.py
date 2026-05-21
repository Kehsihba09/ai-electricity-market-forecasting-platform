import json
import os

def save_model_config(

    config,

    model_name
):

    os.makedirs(

        "artifacts/configs",

        exist_ok=True
    )

    save_path = (

        f"artifacts/configs/"
        f"{model_name}.json"
    )

    with open(

        save_path,

        "w"
    ) as f:

        json.dump(

            config,

            f,

            indent=4
        )

    print(
        f"Config saved: {save_path}"
    )