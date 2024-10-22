import pandas as pd
from lightgbm import LGBMClassifier
import os
import yaml
import pickle
import logging

# Set up logging for the model building process
logger = logging.getLogger("Model building")
logger.setLevel(logging.DEBUG)

# Create a console handler to output logs to the console
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Define the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def params_file_load(filepath: str):
    """Load parameters from a YAML file."""
    try:
        with open(filepath) as f:
            # Load model building parameters from YAML
            params = yaml.safe_load(f)["model_building"]
        logger.info("Params YAML file is read successfully")
        return params

    except FileNotFoundError:
        logger.error("Params YAML file not found")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"Error in params.yaml file format: {exc}")
        raise
    except Exception as e:
        logger.error(e)
        raise

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



def train_model(X_train, y_train, **kwargs)->LGBMClassifier:
    """Train the LightGBM model with the provided training data."""
    try:
        model = LGBMClassifier(**kwargs)  # Initialize the model with parameters
        model.fit(X_train, y_train)  # Fit the model on the training data
        logger.info("Model trained successfully")

        return model

    except Exception as e:
        logger.error(e)
        raise

def model_save(model, path: str)->None:
    """Save the trained model to a file."""
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)  # Serialize the model to a file
        logger.info("Model saved successfully")

    except FileNotFoundError as e:
        logger.error(e)
        raise
    except Exception as e:
        logger.error(e)

def main()->None:
    """Main function to orchestrate model building."""
    params = params_file_load("params.yaml")  # Load model parameters

    train_dataset = load_data(os.path.join("data", "final", "train.csv"))  # Load training data

    # Split dataset into features and target variable
    X_train, y_train = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]

    # Train the model with the features and target variable
    model = train_model(X_train, y_train, **params["model_parameters"])

    # Save the trained model to a specified path
    model_save(model, os.path.join("models", "final_model.pkl"))

# Entry point of the script
if __name__ == "__main__":
    main()
