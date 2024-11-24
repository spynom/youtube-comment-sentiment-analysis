import mlflow
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Set up MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/spynom/youtube-comment-sentiment-analysis.mlflow")

# Load the new model from MLflow model registry
model_name = "youtube_sentimental_analysis_model"

client = mlflow.tracking.MlflowClient()
version = client.get_registered_model(model_name).latest_versions[-1].version
client.set_registered_model_alias(model_name, "production", "{}".format(version))
