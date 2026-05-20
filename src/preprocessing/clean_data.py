import re
from src.utils.logger import get_logger

logger = get_logger("data_cleaning")

def clean_data(df):
    logger.info("Starting data cleaning process.")

    df = df.drop_duplicates()

    df = df.sort_values("Datetime")

    df = df.ffill()

    # STANDARDIZE COLUMN NAMES

    df.columns = [

    re.sub(
        r"_+",
        "_",

        col.strip()
           .lower()
           .replace(" ", "_")
           .replace("(", "")
           .replace(")", "")
           .replace("/", "_")
           .replace("*", "")
           .replace("-", "_")
    ).strip("_")

    for col in df.columns
    ]
    print(df.columns.tolist())

    logger.info(f"Data cleaned successfully. Shape: {df.shape}")

    return df
