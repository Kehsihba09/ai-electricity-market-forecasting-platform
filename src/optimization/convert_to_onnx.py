import joblib

from xgboost import XGBRegressor

from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import (
calculate_linear_regressor_output_shapes
)

from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
convert_xgboost
)

from skl2onnx.common.data_types import (
FloatTensorType
)

from skl2onnx import convert_sklearn

from src.utils.logger import get_logger

logger = get_logger("onnx_conversion")

MODEL_PATH = "models/xgboost_model.pkl"

ONNX_PATH = "models/xgboost_model.onnx"

def convert_model_to_onnx():

    logger.info(
        "Loading trained model."
    )

    model = joblib.load(MODEL_PATH)

    logger.info(
        "Registering XGBoost converter."
    )

    update_registered_converter(

        XGBRegressor,

        "XGBoostXGBRegressor",

        calculate_linear_regressor_output_shapes,

        convert_xgboost
    )

    feature_count = 8

    initial_type = [

        (
            "float_input",

            FloatTensorType(
                [None, feature_count]
            )
        )
    ]

    logger.info(
        "Converting model to ONNX."
    )

    onnx_model = convert_sklearn(

        model,
        initial_types=initial_type,
        target_opset={ "": 12, "ai.onnx.ml": 3 }
    )

    with open(ONNX_PATH, "wb") as f:

        f.write(
            onnx_model.SerializeToString()
        )

    logger.info(
        f"ONNX model saved at "
        f"{ONNX_PATH}"
    )

if __name__ == "__main__":

    convert_model_to_onnx()

