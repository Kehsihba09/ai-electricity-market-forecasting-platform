from src.preprocessing.load_data import load_data
from src.preprocessing.validate_data import validate_data
from src.preprocessing.clean_data import clean_data

from src.preprocessing.preprocessing_pipeline import (
fit_transform_pipeline
)

from src.feature_engineering.feature_pipeline import (
build_features
)

from src.training.dataset import (
create_train_validation_test_split
)

from src.training.train_xgboost import (
train_xgboost_model
)

from src.tuning.hyperparameter import (
optimize_xgboost
)

from src.utils.config_loader import load_config

TARGET_COLUMN = "MCP (Rs/MWh) *"

config = load_config()

df = load_data(
config["data"]["raw_data_path"]
)

validate_data(df)

df = clean_data(df)

df, pipeline = fit_transform_pipeline(df)

df = build_features(df)

(
X_train,
X_val,
X_test,
y_train,
y_val,
y_test
) = create_train_validation_test_split(
df,
TARGET_COLUMN
)

best_params = optimize_xgboost(
X_train,
y_train,
X_val,
y_val
)

model, metrics = train_xgboost_model(
X_train,
y_train,
X_val,
y_val,
X_test,
y_test,
best_params
)

print(metrics)
