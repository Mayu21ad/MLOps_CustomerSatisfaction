from zenml import pipeline
from steps.config import ModelNameConfig
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    config = ModelNameConfig(model_name="LinearRegression")
    model = train_model(X_train=X_train, y_train=y_train, config=config)
    evaluate_model(model=model, X_test=X_test, y_test=y_test)

if __name__ == "__main__":
    train_pipeline(data_path="data/olist_customers_dataset.csv")
