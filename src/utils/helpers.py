import pandas as pd

def validate_datetime_column(df: pd.DataFrame,column: str = "Datetime"):

    if column not in df.columns:
        raise ValueError(f"{column} not found.")

    df[column] = pd.to_datetime(df[column])

    return df
