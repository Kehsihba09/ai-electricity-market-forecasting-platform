from src.utils.logger import get_logger

logger = get_logger("rolling_features")

def create_rolling_features(df,target_column,windows=[24, 168]):
    for window in windows:

        df[f"{target_column}_rolling_mean_{window}"] = (
            df[target_column]
            .rolling(window)
            .mean()
        )

        df[f"{target_column}_rolling_std_{window}"] = (
            df[target_column]
            .rolling(window)
            .std()
        )

    logger.info("Rolling features created successfully.")

    return df
