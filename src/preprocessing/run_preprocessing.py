import os

from src.preprocessing.load_data import (
    load_data
)

from src.preprocessing.clean_data import (
    clean_data
)

from src.preprocessing.validate_data import (
    validate_data
)

from src.preprocessing.preprocessing_pipeline import (

    fit_transform_pipeline,

    save_pipeline
)

from src.features.feature_pipeline import (
    build_advanced_features
)

from src.utils.logger import get_logger

logger = get_logger(
    "run_preprocessing"
)

RAW_DATA_PATH = (
    "data/raw/power.csv"
)

TARGET_COLUMN = "mcp_rs_mwh"

def run_preprocessing_pipeline():

    logger.info(
        "Loading raw dataset."
    )

    df = load_data(
        RAW_DATA_PATH
    )

    logger.info(
        "Cleaning dataset."
    )

    df = clean_data(df)

    logger.info(
        "Validating dataset."
    )

    validate_data(df)

    logger.info(
        "Applying preprocessing pipeline."
    )

    df, pipeline = (
        fit_transform_pipeline(df)
    )

    logger.info(
        "Building advanced features."
    )

    df = build_advanced_features(

        df,

        TARGET_COLUMN
    )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    output_path = (
        "data/processed/"
        "final_processed_data.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    logger.info(
        f"Processed dataset saved to "
        f"{output_path}"
    )

    save_pipeline(pipeline)

    logger.info(
        "Preprocessing pipeline complete."
    )

if __name__ == "__main__":

    run_preprocessing_pipeline()