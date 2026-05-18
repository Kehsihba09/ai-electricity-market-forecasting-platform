from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import ROOT_DIR

logger = get_logger("environment_test")

config = load_config()

logger.info("Environment initialized successfully.")
logger.info(f"Project root: {ROOT_DIR}")
logger.info(f"Loaded config: {config}")
