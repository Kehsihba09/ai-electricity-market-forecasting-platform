import numpy as np

from src.utils.logger import (
    get_logger
)

logger = get_logger(
    "cyclical_features"
)

def create_cyclical_features(df):

    logger.info(
        "Creating cyclical time features."
    )

    # DETECT DATETIME COLUMN

    datetime_column = None

    possible_datetime_columns = [

        "datetime",

        "Datetime",

        "date",

        "Date",

        "timestamp",

        "Timestamp"
    ]

    for col in possible_datetime_columns:

        if col in df.columns:

            datetime_column = col

            break

    # IF DATETIME COLUMN EXISTS

    if datetime_column:

        df[datetime_column] = (
            df[datetime_column]
            .astype("datetime64[ns]")
        )

        df["hour"] = (
            df[datetime_column]
            .dt.hour
        )

        df["day_of_week"] = (
            df[datetime_column]
            .dt.dayofweek
        )

    else:

        logger.warning(

            "No datetime column found. "

            "Using synthetic temporal features."
        )

        df["hour"] = (
            np.arange(len(df)) % 24
        )

        df["day_of_week"] = (
            np.arange(len(df)) % 7
        )

    # CYCLICAL ENCODING

    df["hour_sin"] = np.sin(

        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(

        2 * np.pi * df["hour"] / 24
    )

    df["day_sin"] = np.sin(

        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(

        2 * np.pi * df["day_of_week"] / 7
    )

    return df