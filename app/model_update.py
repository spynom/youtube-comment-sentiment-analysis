
from mlflow.tracking import MlflowClient
import mlflow
from dotenv import load_dotenv
from get_logger import Logger
# Load environment variables from .env file
load_dotenv()
import os
from pathlib import Path
#setup logger
logger = Logger.get_logger("model_update")

# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri("https://dagshub.com/spynom/youtube-comment-sentiment-analysis.mlflow")

# Initialize the MLflow client
client = MlflowClient()





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
        if not os.path.exists(path=Path(f"artifacts/{model_artifact}")) and not os.path.exists(Path(f"artifacts/{transformer}")):
            artifacts = client.list_artifacts(run_id)

            client.download_artifacts(run_id, model_artifact, dst_path="artifacts")
            client.download_artifacts(run_id, transformer, dst_path="artifacts")
            logger.info("Downloaded artifacts successfully")

        else:
            logger.info("Artifacts already exist")

    except Exception as e:
        logger.error("Error occurred while downloading artifacts %s",e)


def update_model():
    run_id = get_run_id()
    if run_id is None:
        pass
    else:
        download_artifacts(run_id)

if __name__ == "__main__":
    update_model()