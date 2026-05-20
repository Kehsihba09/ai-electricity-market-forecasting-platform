import pandas as pd
import os

LEADERBOARD_PATH = (
    "artifacts/dl_leaderboard.csv"
)

def update_dl_leaderboard(results):

    df_new = pd.DataFrame([results])

    if os.path.exists(
        LEADERBOARD_PATH
    ):

        df_existing = pd.read_csv(
            LEADERBOARD_PATH
        )

        df_final = pd.concat(

            [df_existing, df_new],

            ignore_index=True
        )

    else:

        df_final = df_new

    df_final = df_final.sort_values(

        by="validation_loss",

        ascending=True
    )

    df_final.to_csv(

        LEADERBOARD_PATH,

        index=False
    )

    print(
        "DL leaderboard updated."
    )