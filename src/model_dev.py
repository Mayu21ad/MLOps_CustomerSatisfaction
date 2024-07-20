import logging
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin

class Model(ABC):

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> RegressorMixin:
        pass

class LinearRegressionModel(Model):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> RegressorMixin:
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model:", exc_info=True)
            raise e
