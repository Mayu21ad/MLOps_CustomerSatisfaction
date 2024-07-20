from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline
def train_pipeline(data_path: str, model_name: str):
    """
    Define the training pipeline.
    
    Args:
        data_path (str): The path to the dataset.
        model_name (str): The name of the model to be used for training.
    """
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train=X_train, y_train=y_train, model_name=model_name)
    evaluate_model(model=model, X_test=X_test, y_test=y_test)

if __name__ == "__main__":
    # Execute the pipeline
    train_pipeline(data_path="data/olist_customers_dataset.csv", model_name="LinearRegression")
