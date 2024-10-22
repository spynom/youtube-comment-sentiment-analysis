import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import logging, yaml




def apply_feature_engineering(train_dataset, test_dataset,ngram_range:list,max_features)->[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    # Vectorization using TF-IDF with varying max_features
    x_train=train_dataset["comment"]
    x_test=test_dataset["comment"]
    y_train=train_dataset["category"]
    y_test=test_dataset["category"]
    vectorizer = TfidfVectorizer(ngram_range=tuple(ngram_range), max_features=max_features)


    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_train = pd.DataFrame(x_train.toarray(), columns=vectorizer.get_feature_names_out())
    x_test = pd.DataFrame(x_test.toarray(), columns=vectorizer.get_feature_names_out())

    y_train = y_train.to_frame().reset_index(drop=True)  # if y_train is a Series
    y_test = y_test.to_frame().reset_index(drop=True)  # if y_test is a Series

    return x_train, x_test, y_train, y_test



if __name__ == '__main__':
    logger = logging.getLogger("Feature Engineering")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        with open('params.yaml', 'r') as ymlfile:
            params = yaml.safe_load(ymlfile)["feature_engineering"]
    except FileNotFoundError as e:
        logger.error(f"params.yaml file not found {e}")
        raise
    except Exception as e:
        logger.error(e)
        raise

    train_file_path = os.path.join("data", "processed", "train.csv")
    test_file_path = os.path.join("data", "processed", "test.csv")

    train = pd.read_csv(train_file_path)
    logger.info("train data read successfully")
    test = pd.read_csv(test_file_path)
    logger.info("test data read successfully")

    X_train, X_test, y_train, y_test =apply_feature_engineering(train, test,ngram_range=params["ngram_range"],max_features=params["max_features"])
    logger.info("feature transformation applied")

    train_path = os.path.join("data", "final", "train.csv")
    test_path = os.path.join("data", "final", "test.csv")

    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    logger.info("train data saved successfully")
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)
    logger.info("test data saved successfully")









