from src.features.lag_features import (
    create_lag_features
)

from src.features.rolling_features import (
    create_rolling_features
)

from src.features.cyclical_features import (
    create_cyclical_features
)

from src.features.volatility_features import (
    create_volatility_features
)

from src.features.market_features import (
    create_market_features
)

from src.utils.logger import get_logger

logger = get_logger("advanced_feature_pipeline")

def build_advanced_features(
    df,
    target_column
):

    logger.info(
        "Starting advanced feature engineering."
    )

    df = create_lag_features(
        df,
        target_column
    )

    df = create_rolling_features(
        df,
        target_column
    )

    df = create_cyclical_features(df)

    df = create_volatility_features(
        df,
        target_column
    )

    df = create_market_features(df)

    df = df.dropna()

    logger.info(
        f"Advanced features created. "
        f"Final shape: {df.shape}"
    )

    return df