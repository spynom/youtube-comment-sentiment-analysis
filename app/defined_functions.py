import numpy as np
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
import mlflow
import pickle
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

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
        model = client.get_registered_model(model_name)
        for version in model.latest_versions:
            # Directly check the tags of the model version
            if version.tags.get(specific_tag_key) == specific_tag_value:
                run_id = version.run_id
                return run_id

            else:
                return None

    except:
        return None

def download_artifacts(run_id):
    try:
        # Load model as a PyFuncModel.
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/lgbm_model")
        model_artifact = "lgbm_model/model.pkl"

        transformer = "tfidf_vectorizer.pkl"

        client.download_artifacts(run_id, model_artifact, dst_path="artifacts")
        client.download_artifacts(run_id, transformer, dst_path="artifacts")
        return True

    except:
        return False

def get_pridiction(comments:list[str]):
    comments = np.array(comments)
    # Load the pickle file
    with open("artifacts/transformer.pkl", 'rb') as file:
        transformer = pickle.load(file)

    print(transformer.transform(comments))

def main():
    run_id = get_run_id()
    if run_id is None:
        print("No run_id")
        pass
    else:
        download_artifacts(run_id)

    get_pridiction(["my name is saurav","what is ur name"])

main()