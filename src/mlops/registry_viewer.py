import json

REGISTRY_PATH = (
    "artifacts/model_registry.json"
)

def view_registered_models():

    with open(

        REGISTRY_PATH,

        "r"
    ) as f:

        registry = json.load(f)

    print("\nMODEL REGISTRY\n")

    for model in registry:

        print("=" * 50)

        print(
            "Model:",
            model["model_name"]
        )

        print(
            "Type:",
            model["model_type"]
        )

        print(
            "Validation Loss:",
            model["validation_loss"]
        )

        print(
            "Registered At:",
            model["registered_at"]
        )

if __name__ == "__main__":

    view_registered_models()