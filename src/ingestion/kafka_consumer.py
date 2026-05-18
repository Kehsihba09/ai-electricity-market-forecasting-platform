import json

from kafka import KafkaConsumer

from src.forecasting.state_manager import (
StreamingState
)

from src.forecasting.stream_predictor import (
predict_stream_event
)

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("kafka_consumer")

config = load_config()

TOPIC_NAME = config["streaming"]["kafka_topic"]

consumer = KafkaConsumer(
TOPIC_NAME,
bootstrap_servers="localhost:9092",
value_deserializer=lambda m:
json.loads(m.decode("utf-8"))
)

state = StreamingState()

def consume_stream():

    logger.info(
        "Starting stream consumer."
    )

    for message in consumer:

        event = message.value

        logger.info(
            f"Received event: {event}"
        )

        state.update(event)

        if state.size() > 168:

            prediction = predict_stream_event(
                state.get_state()
            )

            logger.info(
                f"Forecasted price: "
                f"{prediction}"
            )

if __name__ == "__main__":
    consume_stream()
