from pathlib import Path

ROOT_DIR = Path("C:/Users/alpha/Documents/ML projects/ai-electricity-market-forecasting-platform").resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models"

REPORT_DIR = ROOT_DIR / "reports"

CONFIG_DIR = ROOT_DIR / "configs"

LOG_DIR = ROOT_DIR / "logs"
