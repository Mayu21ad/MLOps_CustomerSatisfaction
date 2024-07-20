import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"],
]:
    """
    Cleans the input DataFrame and splits it into training and testing sets.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame to be cleaned and split.
    
    Returns:
        Tuple: A tuple containing the training and testing sets for features and target variable.
    """
    try:
        logging.info("Starting data preprocessing")
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        logging.info("Data preprocessing completed")

        logging.info("Starting data splitting")
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data splitting completed")

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data:", exc_info=True)
        raise e
