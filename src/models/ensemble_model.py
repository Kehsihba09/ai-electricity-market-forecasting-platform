import numpy as np

class EnsembleForecastModel:

    def __init__(self, models):

        self.models = models

    def predict(self, X_test):

        predictions = []

        for model in self.models:

            predictions.append(
                model.predict(X_test)
            )

        predictions = np.array(
            predictions
        )

        return predictions.mean(axis=0)