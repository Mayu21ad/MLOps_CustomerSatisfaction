from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    # Execute the pipeline with the corrected file path
    train_pipeline(data_path="data/olist_customers_dataset.csv")
