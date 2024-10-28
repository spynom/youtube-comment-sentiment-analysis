import mlflow
import json
from src.DagshubConnector import connector
import unittest


class TestModelLoading(unittest.TestCase):

    def setUp(self):
        # connect to Dagshub server
        connector()

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri("https://dagshub.com/spynom/youtube-comment-sentiment-analysis.mlflow")

        # Load the new model from MLflow model registry
        model_name = "youtube_sentimental_analysis_model"

        client = mlflow.tracking.MlflowClient()

        registered_model = client.get_registered_model(model_name)

        production_model_version = registered_model.aliases["production"]

        # Get the production stage model details
        production_model_run_id = client.get_model_version(name=model_name, version=production_model_version).run_id

        production_model_run = mlflow.get_run(production_model_run_id)

        self.production_model_metrics = production_model_run.data.metrics

        # Get the production stage model details
        latest_model_run_id = registered_model.latest_versions[-1].run_id

        self.latest_model_metrics = mlflow.get_run(latest_model_run_id).data.metrics








    def test_model_performance(self):

        # Define expected thresholds from production stage model
        expected_accuracy = self.production_model_metrics["test_accuracy"]
        expected_class_0_precision = self.production_model_metrics["test_0_precision"]
        expected_class_0_recall = self.production_model_metrics["test_1_precision"]

        # Define new model
        actual_accuracy = self.latest_model_metrics["test_accuracy"]
        actual_class_0_precision = self.latest_model_metrics["test_0_precision"]
        actual_class_0_recall = self.latest_model_metrics["test_1_precision"]

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(actual_accuracy, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(actual_class_0_precision,expected_class_0_precision, f'Precision should be at least {expected_class_0_precision} for class 0')
        self.assertGreaterEqual(actual_class_0_recall, expected_class_0_recall, f'Recall should be at least {expected_class_0_recall}' f'for class 0')