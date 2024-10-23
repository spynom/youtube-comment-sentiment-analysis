import json
import pandas as pd
from lightgbm import LGBMClassifier
import os
import yaml
import pickle
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from dotenv import load_dotenv
from mlflow.models import infer_signature
import dagshub

# Load environment variables from .env file
load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token



# Set up logging for the model building process
logger = logging.getLogger("Model evaluation")
logger.setLevel(logging.DEBUG)

# Create a console handler to output logs to the console
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Define the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_data(path: str)->pd.DataFrame:
    """Load training dataset from a CSV file."""
    try:
        train_dataset = pd.read_csv(path)
        logger.info("Train dataset read successfully")
        return train_dataset

    except FileNotFoundError as e:
        logger.error("Train dataset file not found")
        raise
    except Exception as e:
        logger.error(e)
        raise

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def load_model(path:str)->LGBMClassifier:
    try:
        model = pickle.load(open(path, 'rb'))
        logger.info("Model loaded from disk")
        return model

    except FileNotFoundError as e:
        logger.error("Model pickle file not found")
        raise
    except Exception as e:
        logger.error(e)
        raise


def evaluate_model(model, X, y,dataset_name):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)


        for label, metrics in report.items():  # Skip overall metrics
            if hasattr(metrics, 'items'):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{dataset_name}_{label}_{metric}", value)
            else:
                mlflow.log_metric(f"{dataset_name}_{label}", metrics)


        log_confusion_matrix(cm,dataset_name)

        logger.debug('Model evaluation completed %s',dataset_name)
        return y_pred

    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = f'reports/figures/confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def save_run_info(run_id,artifact_path):
    with open("mlflow_experiment_info.json","w") as f:
        json.dump(
            {"run_id": run_id,
             "artifact_path": artifact_path}
        ,f)

def main():
    mlflow.set_tracking_uri("https://dagshub.com/spynom/youtube-comment-sentiment-analysis.mlflow")
    mlflow.set_experiment('youtube_sentiment_analysis_dvc_pipeline')

    with mlflow.start_run() as run:

        params = load_params('params.yaml')

        for key, value in params.items():
            mlflow.log_param(key, value)

        train_dataset = load_data(os.path.join('data', 'final','train.csv'))
        test_dataset = load_data(os.path.join('data', 'final','test.csv'))

        model = load_model(os.path.join('models','final_model.pkl'))



        evaluate_model(model, train_dataset.iloc[:,:-1], train_dataset.iloc[:,-1], 'train')
        y_test_pred = evaluate_model(model, test_dataset.iloc[:,:-1], test_dataset.iloc[:,-1], 'test')

        signature = infer_signature(test_dataset.iloc[:,:-1], y_test_pred)

        # Log model and vectorizer
        mlflow.sklearn.log_model(model, artifact_path="LGBMClassifier_model",signature=signature)
        mlflow.log_artifact(os.path.join("models", 'transformer.pkl'))
        # Add important tags
        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag("transformer_type", "tfidf_vectorizer")
        mlflow.set_tag("task", "Sentiment Analysis")
        mlflow.set_tag("dataset", "YouTube Comments")

        run_id = run.info.run_id
        artifact_uri = mlflow.get_artifact_uri()
        save_run_info(run_id,artifact_uri)



if __name__ == '__main__':
    main()





