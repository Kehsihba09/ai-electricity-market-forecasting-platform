from src.utils.logger import get_logger

logger = get_logger("dataset_builder")

def create_train_validation_test_split(df,target_column,test_size=0.2,validation_size=0.1):

    logger.info("Creating train-validation-test split.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    total_size = len(df)

    train_end = int(
        total_size * (1 - test_size - validation_size)
    )

    validation_end = int(
        total_size * (1 - test_size)
    )

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:validation_end]
    y_val = y.iloc[train_end:validation_end]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    logger.info(
        f"Train shape: {X_train.shape}, "
        f"Validation shape: {X_val.shape}, "
        f"Test shape: {X_test.shape}"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )
