from src.feature_engineering.lag_features import (
create_lag_features
)

from src.feature_engineering.rolling_features import (
create_rolling_features
)

from src.feature_engineering.calendar_features import (
create_calendar_features
)

from src.utils.logger import get_logger

logger = get_logger("feature_pipeline")

TARGET_COLUMN = "MCP (Rs/MWh) *"

def build_features(df):
    logger.info("Starting feature engineering.")

    df = create_calendar_features(df)

    df = create_lag_features(
        df,
        TARGET_COLUMN
    )

    df = create_rolling_features(
        df,
        TARGET_COLUMN
    )

    df = df.dropna()

    logger.info(
        f"Feature engineering completed. Shape: {df.shape}"
    )

    return df
