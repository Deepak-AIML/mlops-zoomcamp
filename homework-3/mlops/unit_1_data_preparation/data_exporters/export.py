import mlflow
import mlflow.sklearn
import os
import joblib

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def log_model_and_artifacts(data, *args, **kwargs):
    model, dict_vectorizer = data  # Unpack the model and dict vectorizer from the data tuple

    with mlflow.start_run() as run:
        # Log the model
        mlflow.sklearn.log_model(model, "linear_regression_model")
        
        # Save and log the dict vectorizer as an artifact
        dict_vectorizer_path = "dict_vectorizer.pkl"
        joblib.dump(dict_vectorizer, dict_vectorizer_path)
        mlflow.log_artifact(dict_vectorizer_path)

        # Get the logged model path
        model_uri = mlflow.get_artifact_uri("linear_regression_model")
        print(f"Model URI: {model_uri}")

        # Find the logged model and get its size
        model_path = mlflow.artifacts.download_artifacts(model_uri)
        model_size_bytes = os.path.getsize(model_path)
        print(f"Model size (bytes): {model_size_bytes}")