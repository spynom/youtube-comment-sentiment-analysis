import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import logging
import yaml
import pickle

# Set up logging for the feature engineering process
logger = logging.getLogger("Feature Engineering")
logger.setLevel(logging.DEBUG)

# Create a console handler to output logs to the console
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Define the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def read_yaml(path: str) -> dict:
    """Read parameters from a YAML file."""
    try:
        with open(path, 'r') as ymlfile:
            # Load feature engineering parameters from YAML
            params = yaml.safe_load(ymlfile)["feature_engineering"]
            logger.info("Params YAML file read: %s", path)
            return params
    except FileNotFoundError as e:
        logger.error(f"Params.yaml file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading params.yaml: {e}")
        raise

def apply_feature_engineering(train_dataset, test_dataset, ngram_range: list, max_features: int) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply TF-IDF vectorization to the training and testing datasets."""
    try:
        # Extract comments and categories from datasets
        x_train = train_dataset["comment"]
        x_test = test_dataset["comment"]
        y_train = train_dataset["category"]
        y_test = test_dataset["category"]

        # Initialize the TF-IDF vectorizer with specified n-gram range and max features
        vectorizer = TfidfVectorizer(ngram_range=tuple(ngram_range), max_features=max_features)

        # Fit and transform the training data, and transform the testing data
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)

        # Convert sparse matrix to DataFrame
        x_train = pd.DataFrame(x_train.toarray(), columns=vectorizer.get_feature_names_out())
        x_test = pd.DataFrame(x_test.toarray(), columns=vectorizer.get_feature_names_out())

        # Reset index for target variables
        y_train = y_train.to_frame().reset_index(drop=True)
        y_test = y_test.to_frame().reset_index(drop=True)

        logger.info(f"Feature transformation applied: with ngram_range: {ngram_range} max_features: {max_features} ")
        return x_train, x_test, y_train, y_test, vectorizer

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise

def main():
    """Main function to orchestrate feature engineering."""
    try:
        # Load parameters for feature engineering
        params = read_yaml("params.yaml")

        # Define file paths for training and testing datasets
        train_file_path = os.path.join("data", "processed", "train.csv")
        test_file_path = os.path.join("data", "processed", "test.csv")

        # Load training and testing datasets
        train = pd.read_csv(train_file_path)
        logger.info(f"Train data read: {train_file_path} ")
        test = pd.read_csv(test_file_path)
        logger.info(f"Test data read: {test_file_path}")

        # Apply feature engineering
        X_train, X_test, y_train, y_test, tfidf_vectorizer = apply_feature_engineering(
            train,
            test,
            ngram_range=params["ngram_range"],
            max_features=params["max_features"]
        )

        # Save the processed datasets
        train_path = os.path.join("data", "final", "train.csv")
        test_path = os.path.join("data", "final", "test.csv")

        pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
        logger.info("Train data saved successfully")

        pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)
        logger.info("Test data saved successfully")

        # Save the vectorizer model to a file
        with open(os.path.join("models", "transformer.pkl"), "wb") as f:
            pickle.dump(tfidf_vectorizer, f)
        logger.info("Transformer model saved successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")

# Entry point of the script
if __name__ == '__main__':
    main()
