import pandas as pd
import os

from src.utils.logger import get_logger

logger = get_logger("leaderboard")

LEADERBOARD_PATH = (
    "artifacts/model_leaderboard.csv"
)

def update_leaderboard(results):

    logger.info(
        "Updating model leaderboard."
    )

    leaderboard_df = pd.DataFrame([results])

    if os.path.exists(
        LEADERBOARD_PATH
    ):

        existing_df = pd.read_csv(
            LEADERBOARD_PATH
        )

        leaderboard_df = pd.concat(

            [
                existing_df,
                leaderboard_df
            ],

            ignore_index=True
        )

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    leaderboard_df.to_csv(
        LEADERBOARD_PATH,
        index=False
    )

    logger.info(
        "Leaderboard updated."
    )