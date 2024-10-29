import os
from dotenv import load_dotenv
from get_logger import Logger
# Load environment variables from .env file
load_dotenv()

#setup logger
logger = Logger.get_logger("connect_dagshub")

class ConnectDagsHub:
    @staticmethod
    def connect():
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            logger.error("DAGSHUB_TOKEN environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token