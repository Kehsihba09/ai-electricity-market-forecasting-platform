from src.preprocessing.load_data import load_data
from src.preprocessing.validate_data import validate_data
from src.preprocessing.clean_data import clean_data

from src.preprocessing.preprocessing_pipeline import (
fit_transform_pipeline
)

from src.feature_engineering.feature_pipeline import (
build_features
)

from src.utils.config_loader import load_config

config = load_config()

print(config)

df = load_data(
config["data"]["raw_data_path"]
)

validate_data(df)

df = clean_data(df)

df, pipeline = fit_transform_pipeline(df)

df = build_features(df)

print(df.head())

print(df.shape)
