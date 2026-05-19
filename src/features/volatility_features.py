from src.utils.logger import get_logger

logger = get_logger("volatility_features")

def create_volatility_features(
    df,
    target_column
):

    logger.info(
        "Creating volatility features."
    )

    df[
        f"{target_column}_volatility"
    ] = (
        df[target_column]
        .rolling(24)
        .std()
    )

    return df