from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Configuration class for specifying the model name.
    
    Attributes:
        model_name (str): The name of the model to be used. Default is "LinearRegression".
    """
    model_name: str = "LinearRegression"
