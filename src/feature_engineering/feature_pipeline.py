from src.utils.logger import get_logger

logger = get_logger("feature_pipeline")

DROP_COLUMNS = [
"Datetime",
"Unnamed: 0",
"Session ID"
]

def build_features(df):

    logger.info(
        "Building production-grade features."
    )

    df["hour"] = df["Datetime"].dt.hour

    df["day_of_week"] = (
        df["Datetime"].dt.dayofweek
    )

    df["month"] = (
        df["Datetime"].dt.month
    )

    df["is_weekend"] = (
        df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    columns_to_drop = [
        col for col in DROP_COLUMNS
        if col in df.columns
    ]

    df = df.drop(columns=columns_to_drop)

    logger.info(
        f"Feature engineering completed. "
        f"Shape: {df.shape}"
    )

    return df
