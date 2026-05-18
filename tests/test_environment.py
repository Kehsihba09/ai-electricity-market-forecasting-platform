from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("test")

config = load_config()

logger.info(config)