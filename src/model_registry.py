import os
import mlflow
import json
from DagshubConnector import connector
from setup_logger import logger
# connect to Dagshub server
connector()
# Set up logging for the model building process
logger = logger("Model Registry")


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
    mlflow.set_tracking_uri("https://dagshub.com/spynom/youtube-comment-sentiment-analysis.mlflow")
    model_info = load_run_info("mlflow_experiment_info.json")
    model_name = "youtube_sentimental_analysis_model"
    model_registry(model_info,model_name)

main()
