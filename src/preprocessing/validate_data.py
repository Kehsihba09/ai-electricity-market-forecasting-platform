from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationError

logger = get_logger("data_validation")

REQUIRED_COLUMNS = [
"Datetime",
"Purchase Bid (MW)",
"Sell Bid (MW)",
"Final Scheduled Volume (MW)",
"MCP (Rs/MWh) *"
]

def validate_data(df):
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise DataValidationError(
            f"Missing columns: {missing_columns}"
        )

    duplicate_count = df.duplicated().sum()

    logger.info(f"Duplicate rows found: {duplicate_count}")

    logger.info("Data validation successful.")

    return True
