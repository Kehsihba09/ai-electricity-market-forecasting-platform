import yaml
from pathlib import Path

CONFIG_PATH = Path("configs/config.yaml")

def load_config(config_path=CONFIG_PATH):

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config
