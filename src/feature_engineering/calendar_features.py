from src.utils.logger import get_logger

logger = get_logger("calendar_features")

def create_calendar_features(df):

    df["hour"] = df["Datetime"].dt.hour
    df["day_of_week"] = df["Datetime"].dt.dayofweek
    df["month"] = df["Datetime"].dt.month
    df["quarter"] = df["Datetime"].dt.quarter

    df["is_weekend"] = (
        df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    logger.info("Calendar features created successfully.")

    return df
