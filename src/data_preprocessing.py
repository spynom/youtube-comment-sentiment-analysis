from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import pandas as pd
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from setup_logger import logger

logger = logger("DataPreprocessing")

# Function to preprocess individual comments
def preprocess_comment(comment):
    try:
        # Convert the comment to lowercase for uniformity
        comment = comment.lower()

        # Remove leading and trailing whitespaces
        comment = comment.strip()

        # Replace newline characters with spaces
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, keeping punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Define stopwords and exclude certain words to retain their importance
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize words to their base forms
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        # Return the processed comment or NaN if empty
        return comment if comment.strip() else np.nan
    except Exception as e:
        logger.error(e)

def read_yaml():
    try:
        # Load parameters from YAML file for preprocessing
        with open('params.yaml', 'r') as ymlfile:
            params = yaml.safe_load(ymlfile)["data_preprocessing"]
            logger.info(f"Params extracted")
            return params

    except FileNotFoundError as e:
        logger.error(f"params.yaml file not found {e}")
        raise
    except Exception as e:
        logger.error(e)


# Load the raw data, rename the comment column, and drop any rows with missing values
def read_data(path:str):
    raw_data = pd.read_csv(path).rename(columns={'clean_comment': 'comment'}).dropna(how='any')
    logger.info(f"Raw data extracted")
    return raw_data


# Function to split the dataset into training and testing sets
def data_split(dataset: pd.DataFrame, test_size, random_state) -> tuple:
    # Split the dataset while preserving the comment and category columns
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["comment"], dataset["category"], test_size=test_size, random_state=random_state
    )
    return (pd.DataFrame({"comment": X_train.values, "category": y_train.values}),
            pd.DataFrame({"comment": X_test.values, "category": y_test.values}))

# Function to save the processed dataset to CSV
def processed_data_saved(dataset: pd.DataFrame, dir: str, type: str = "train") -> None:
    dataset.to_csv(os.path.join(dir, f"{type}.csv"), index=False)

# Function to preprocess category labels
def preprocess_category(category: pd.Series) -> pd.Series:
    # Increment category labels by 1 (e.g., from 0 to 1)
    return category + 1

def main():
    params = read_yaml()
    # Define the path to the raw data CSV file
    raw_data = read_data(os.path.join("data", "raw", "reddit.csv"))

    # Define stopwords, excluding certain words to keep their significance
    stop_words = set(stopwords.words('english')) - set(params["keep_stop_words"])
    logger.info(f"Stop words defined")

    # Clean the raw data by applying preprocessing to comments and categories
    raw_data = raw_data.assign(
        comment=raw_data.comment.apply(preprocess_comment),
        category=raw_data.category.pipe(preprocess_category)
    ).dropna(how='any').drop_duplicates()

    logger.info(f"Raw data cleaned")

    # Define directory for saving processed data
    save_dir = os.path.join("data", "processed")

    # Split the cleaned data into training and testing sets
    train, test = data_split(raw_data, params["test_size"], params["random_state"])
    logger.info(f"Train and test data split")

    # Save the training and testing datasets
    processed_data_saved(train, save_dir, "train")
    logger.info(f"Train data saved")

    processed_data_saved(test, save_dir, "test")
    logger.info(f"Test data saved")

if __name__ == '__main__':
    main()

