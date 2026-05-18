import pandas as pd

from src.utils.logger import get_logger
from src.utils.helpers import validate_datetime_column

logger = get_logger("data_loader")

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")

    df = pd.read_csv(file_path)

    df = validate_datetime_column(df)

    logger.info(f"Dataset loaded successfully with shape: {df.shape}")

    return df