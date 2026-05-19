from abc import ABC, abstractmethod

class BaseForecastModel(ABC):

    @abstractmethod
    def train(
        self,
        X_train,
        y_train
    ):
        pass

    @abstractmethod
    def predict(
        self,
        X_test
    ):
        pass

    @abstractmethod
    def save_model(
        self,
        path
    ):
        pass