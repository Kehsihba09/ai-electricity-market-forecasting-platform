from src.utils.logger import get_logger

logger = get_logger("market_features")

def create_market_features(df):

    logger.info(
        "Creating market interaction features."
    )

    df["bid_spread"] = (
        df["purchase_bid"]
        - df["sell_bid"]
    )

    df["volume_pressure"] = (
        df["final_scheduled_volume"]
        / (
            df["purchase_bid"] + 1
        )
    )

    return df