import json
import time

from kafka import KafkaProducer

from src.preprocessing.load_data import load_data

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("kafka_producer")

config = load_config()

TOPIC_NAME = config["streaming"]["kafka_topic"]

producer = KafkaProducer(
bootstrap_servers="localhost:9092",
value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def stream_historical_data():

    df = load_data(
        config["data"]["raw_data_path"]
    )

    logger.info(
        "Starting simulated market stream."
    )

    for _, row in df.iterrows():

        message = row.to_dict()

        if "Datetime" in message:
            message["Datetime"] = str(
                message["Datetime"]
            )

        producer.send(
            TOPIC_NAME,
            value=message
        )

        logger.info(
            f"Streamed event: {message}"
        )

        time.sleep(1)

if __name__ == "__main__":
    stream_historical_data()
