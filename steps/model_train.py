import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,  # y_train should be pd.Series for target values
    model_name: str,     # Updated to directly receive the model name as a string
) -> RegressorMixin:
    """
    Trains a machine learning model based on the provided configuration.

    Parameters:
        X_train (pd.DataFrame): The training feature data.
        y_train (pd.Series): The training target data.
        model_name (str): The name of the model to be trained.

    Returns:
        RegressorMixin: The trained machine learning model.
    """
    try:
        model = None
        logging.info(f"Starting training for model: {model_name}")

        if model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            logging.info(f"Model {model_name} training completed successfully")
            return trained_model
        else:
            raise ValueError(f"Model {model_name} not supported")
    except Exception as e:
        logging.error("Error in training model:", exc_info=True)
        raise e
