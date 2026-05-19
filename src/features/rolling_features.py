from src.utils.logger import get_logger

logger = get_logger("rolling_features")

def create_rolling_features(
    df,
    target_column,
    windows=[3, 6, 12, 24]
):

    logger.info(
        "Creating rolling statistics."
    )

    for window in windows:

        df[
            f"{target_column}_rolling_mean_{window}"
        ] = (
            df[target_column]
            .rolling(window)
            .mean()
        )

        df[
            f"{target_column}_rolling_std_{window}"
        ] = (
            df[target_column]
            .rolling(window)
            .std()
        )

        df[
            f"{target_column}_rolling_max_{window}"
        ] = (
            df[target_column]
            .rolling(window)
            .max()
        )

        df[
            f"{target_column}_rolling_min_{window}"
        ] = (
            df[target_column]
            .rolling(window)
            .min()
        )

    return df