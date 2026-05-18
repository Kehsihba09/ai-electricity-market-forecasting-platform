from dotenv import load_dotenv
import os

load_dotenv()

def get_env_variable(name: str):

    value = os.getenv(name)

    if value is None:
        raise ValueError(f"Environment variable '{name}' not found.")

    return value