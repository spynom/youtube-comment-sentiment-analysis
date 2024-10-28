import numpy as np
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
import mlflow
import pickle
from dotenv import load_dotenv
import os
import logging

# Load environment variables from .env file
load_dotenv()

# Set up logging for the model building process
logger = logging.getLogger("server")
logger.setLevel(logging.DEBUG)

# Create a console handler to output logs to the console
handler = logging.FileHandler('server.log')
handler.setLevel(logging.DEBUG)

# Define the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    logger.error("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri("https://dagshub.com/spynom/youtube-comment-sentiment-analysis.mlflow")

# Initialize the MLflow client
client = MlflowClient()

# Specify the model name and the tag you want to search for
model_name = "youtube_sentimental_analysis_model"  #  model name
specific_tag_key = "approved"  #  tag key
specific_tag_value = "True"  #  tag value


class comment(BaseModel):
    comments: list[str]

def get_run_id():
    # Fetch the specified model
    try:
        # Load the new model from MLflow model registry
        model_name = "youtube_sentimental_analysis_model"

        client = mlflow.tracking.MlflowClient()

        registered_model = client.get_registered_model(model_name)

        production_model_version = registered_model.aliases["production"]

        # Get the production stage model details
        production_model_run_id = client.get_model_version(name=model_name, version=production_model_version).run_id
        logger.info("fetched run_id %s", production_model_run_id)

        return production_model_run_id

    except Exception as e:
        logger.error("Error occurred while fetching run_id %s",e)
        return None

def download_artifacts(run_id):
    try:
        # Load model as a PyFuncModel.
       # model = mlflow.pyfunc.load_model(f"runs:/{run_id}/LGBMClassifier_model")
        model_artifact = "LGBMClassifier_model/model.pkl"

        transformer = "transformer.pkl"
        artifacts = client.list_artifacts(run_id)

        client.download_artifacts(run_id, model_artifact, dst_path="artifacts")
        client.download_artifacts(run_id, transformer, dst_path="artifacts")
        logger.info("Downloaded artifacts successfully")

    except Exception as e:
        logger.error("Error occurred while downloading artifacts %s",e)


def update_model():
    run_id = get_run_id()
    if run_id is None:
        print("No run_id")
        pass
    else:
        download_artifacts(run_id)
