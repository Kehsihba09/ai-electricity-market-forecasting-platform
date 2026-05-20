from sklearn.model_selection import (
    train_test_split
)

from src.deep_learning.dataloader import (
    create_dataloader
)

def prepare_dataloaders(

    X,

    y,

    sequence_length,

    batch_size=32
):

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,

            y,

            test_size=0.2,

            shuffle=False
        )
    )

    train_loader = create_dataloader(

        X_train,

        y_train,

        batch_size=batch_size,

        sequence_length=sequence_length
    )

    validation_loader = create_dataloader(

        X_test,

        y_test,

        batch_size=batch_size,

        sequence_length=sequence_length
    )

    return (

        train_loader,

        validation_loader
    )