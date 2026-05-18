from src.utils.logger import get_logger

logger = get_logger("data_cleaning")

def clean_data(df):
    logger.info("Starting data cleaning process.")

    df = df.drop_duplicates()

    df = df.sort_values("Datetime")

    df = df.ffill()

    logger.info(f"Data cleaned successfully. Shape: {df.shape}")

    return df
