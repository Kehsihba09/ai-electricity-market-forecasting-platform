from src.utils.logger import get_logger

logger = get_logger("lag_features")

def create_lag_features(df,target_column,lags=[1, 24, 48, 168]):

    for lag in lags:

        df[f"{target_column}_lag_{lag}"] = (
            df[target_column].shift(lag)
        )

    logger.info("Lag features created successfully.")

    return df
