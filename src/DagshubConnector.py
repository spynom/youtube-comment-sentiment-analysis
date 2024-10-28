from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()


def connector():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token