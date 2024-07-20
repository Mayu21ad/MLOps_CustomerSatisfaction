import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import MSE, R2, RMSE

@step
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
    Annotated[float, "mse"],  # Include mse in the return type hint
]:
    """
    Evaluate the model using the test data and calculate MSE, R2, and RMSE scores.

    Parameters:
        model (RegressorMixin): The trained model to be evaluated.
        X_test (pd.DataFrame): The test set features.
        y_test (pd.DataFrame): The test set target variable.

    Returns:
        Tuple: A tuple containing the R2 score, RMSE, and MSE.
    """
    try:
        logging.info("Starting model evaluation")
        prediction = model.predict(X_test)

        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)

        logging.info(f"Evaluation completed: R2 Score = {r2}, RMSE = {rmse}, MSE = {mse}")
        return r2, rmse, mse
    
    except Exception as e:
        logging.error("Error in evaluating model:", exc_info=True)
        raise e
