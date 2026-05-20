import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split
)

from src.deep_learning.dataloader import (
    create_dataloader
)

from src.deep_learning.gru_model import (
    GRUForecastModel
)

from src.deep_learning.trainer import (
    ForecastTrainer
)

# DUMMY DATA

X = pd.DataFrame(
    np.random.rand(500, 10)
)

y = pd.Series(
    np.random.rand(500)
)

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

    y_train
)

validation_loader = create_dataloader(

    X_test,

    y_test
)

model = GRUForecastModel(
    input_size=10
)

trainer = ForecastTrainer(
    model=model
)

trainer.train(

    train_loader,

    validation_loader
)