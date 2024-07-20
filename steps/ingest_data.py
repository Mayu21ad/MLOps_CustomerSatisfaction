import logging
import pandas as pd 
from zenml import step

class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info("Ingesting data from {}".format(self.data_path))
        return pd.read_csv("data\olist_customers_dataset.csv")
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}", exc_info=True)
        raise e
