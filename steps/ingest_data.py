import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Class to ingest data from a specified path.

    Attributes:
        data_path (str): The path to the data file.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        """
        Reads data from the specified CSV file.

        Returns:
            pd.DataFrame: The ingested data as a DataFrame.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)  # Use the provided data_path

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Step to ingest data using the IngestData class.

    Parameters:
        data_path (str): The path to the data file.

    Returns:
        pd.DataFrame: The ingested data as a DataFrame.
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        logging.info("Data ingestion completed successfully")
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}", exc_info=True)
        raise e
