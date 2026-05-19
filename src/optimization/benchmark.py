import time
import joblib
import numpy as np

from src.optimization.onnx_predictor import (
predict_onnx
)

from src.utils.logger import get_logger

logger = get_logger("benchmark")

MODEL_PATH = "models/xgboost_model.pkl"

model = joblib.load(MODEL_PATH)

sample_input = np.array([[
1200,
1000,
200,
950,
14,
2,
5,
0
]], dtype=np.float32)

NUM_ITERATIONS = 1000

def benchmark_pickle():

    start = time.time()

    for _ in range(NUM_ITERATIONS):

        prediction = model.predict(
            sample_input
        )

    latency = (
        (time.time() - start)
        * 1000
        / NUM_ITERATIONS
    )

    logger.info(
        f"Pickle average latency: "
        f"{latency:.4f} ms"
    )

    return latency

def benchmark_onnx():

    start = time.time()

    for _ in range(NUM_ITERATIONS):

        prediction = predict_onnx(
            sample_input
        )

    latency = (
        (time.time() - start)
        * 1000
        / NUM_ITERATIONS
    )

    logger.info(
        f"ONNX average latency: "
        f"{latency:.4f} ms"
    )

    return latency

if __name__ == "__main__":

    pickle_latency = benchmark_pickle()

    onnx_latency = benchmark_onnx()

    print(
        f"Pickle Avg Latency: "
        f"{pickle_latency:.4f} ms"
    )

    print(
        f"ONNX Avg Latency: "
        f"{onnx_latency:.4f} ms"
    )
