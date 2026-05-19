import numpy as np
import onnxruntime as ort

from src.utils.logger import get_logger

logger = get_logger("onnx_predictor")

ONNX_PATH = "models/xgboost_model.onnx"

session = ort.InferenceSession(
ONNX_PATH
)

input_name = session.get_inputs()[0].name

def predict_onnx(features):

    features = np.array(
        features,
        dtype=np.float32
    )

    prediction = session.run(

        None,

        {
            input_name: features
        }
    )

    logger.info(
        f"ONNX prediction: {prediction}"
    )

    return prediction[0]
