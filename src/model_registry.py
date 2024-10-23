import os

import mlflow
import json
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import dagshub
dagshub.init(repo_owner='spynom', repo_name='youtube-comment-sentiment-analysis', mlflow=True)

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


def load_run_info(file:str):
    try:
        with open(file, 'r') as run_info:
            file = json.load(run_info)
            logger.debug("mlflow_experiment_info.json loaded from %s", file)
            return file
    except FileNotFoundError:
        logger.error("mlflow_experiment_info.json not found")
        raise
    except json.decoder.JSONDecodeError:
        logger.error("mlflow_experiment_info.json decoding error")
        raise
    except Exception as e:
        logger.error(e)
        raise

def model_registry(model_info,model_name):
    model_version = mlflow.register_model(f"runs:/{model_info['run_id']}/model", model_name,tags={"Stage":"Staging"})
    client = mlflow.tracking.MlflowClient()
    #latest_mv = client.get_latest_versions(model_name)[-1]
    #client.set_registered_model_alias(model_name, "staging", latest_mv.version)


def main():
    model_info = load_run_info("mlflow_experiment_info.json")
    model_name = "youtube_sentimental_analysis_model"
    model_registry(model_info,model_name)

main()
