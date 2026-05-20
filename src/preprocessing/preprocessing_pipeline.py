import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from feature_engine.outliers import Winsorizer

from src.utils.logger import get_logger

logger = get_logger("preprocessing_pipeline")

NUMERICAL_COLUMNS = [

    "purchase_bid_mw",

    "sell_bid_mw",

    "final_scheduled_volume_mw",

    "mcp_rs_mwh"
]

def build_preprocessing_pipeline():

    pipeline = Pipeline([
        (
            "scaler",
            MinMaxScaler()
        ),
        (
            "winsorizer",
            Winsorizer(
                capping_method="iqr",
                tail="both",
                fold=1.5
            )
        )
    ])

    return pipeline

def fit_transform_pipeline(df):

    pipeline = build_preprocessing_pipeline()

    df[NUMERICAL_COLUMNS] = pipeline.fit_transform(
        df[NUMERICAL_COLUMNS]
    )

    logger.info("Preprocessing pipeline fitted successfully.")

    return df, pipeline

def save_pipeline(pipeline, save_path="models/preprocessing_pipeline.pkl"):

    joblib.dump(pipeline, save_path)

    logger.info(f"Pipeline saved at {save_path}")
