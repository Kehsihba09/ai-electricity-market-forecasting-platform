import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("drift_monitor")

def detect_feature_drift(

    reference_df,

    current_df,

    threshold=0.2
):

    logger.info(
        "Checking feature drift."
    )

    drift_results = {}

    for column in reference_df.columns:

        reference_mean = (
            reference_df[column]
            .mean()
        )

        current_mean = (
            current_df[column]
            .mean()
        )

        drift_score = abs(

            current_mean
            - reference_mean

        ) / (
            abs(reference_mean) + 1e-6
        )

        drift_results[column] = {

            "drift_score": drift_score,

            "drift_detected":
                drift_score > threshold
        }

    return drift_results