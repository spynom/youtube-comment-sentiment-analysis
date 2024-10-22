from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import pandas as pd
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
import logging




def preprocess_comment(comment):
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment if comment.strip()  else np.nan
    except Exception as e:
        logger.error(e)

def data_split(dataset:pd.DataFrame,test_size,random_state)->tuple:

    X_train,X_test,y_train,y_test=train_test_split(dataset["comment"],dataset["category"],test_size=test_size,random_state=random_state)

    return pd.DataFrame( {"comment":X_train.values,"category":y_train.values}),pd.DataFrame( {"comment":X_test.values,"category":y_test.values})

def processed_data_saved(dataset:pd.DataFrame,dir:str,type:str="train")->None:
    dataset.to_csv(os.path.join(dir,f"{type}.csv"),index=False)




def preprocess_category(category:pd.Series)->pd.Series:
    return category+1

if __name__ == '__main__':
    logger = logging.getLogger("DataIngestion")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        with open('params.yaml', 'r') as ymlfile:
            params = yaml.safe_load(ymlfile) ["data_preprocessing"]
            logger.info(f"Params extracted")

    except FileNotFoundError as e:
        logger.error(f"params.yaml file not found {e}")
        raise

    except Exception as e:
        logger.error(e)

    path = os.path.join("data","raw","reddit.csv")

    raw_data = pd.read_csv(path).rename(columns={'clean_comment':'comment'}).dropna(how='any')
    logger.info(f"Raw data extracted")

    stop_words = set(stopwords.words('english')) - set(params ["keep_stop_words"])
    logger.info(f"Stop words defined")

    raw_data=raw_data.assign(
        comment=raw_data.comment.apply(preprocess_comment),
        category = raw_data.category.pipe(preprocess_category)
                            ).dropna(how='any').drop_duplicates()

    logger.info(f"Raw data cleaned")

    save_dir = os.path.join("data","processed")

    train,test=data_split(raw_data,params["test_size"],params["random_state"])
    logger.info(f"Train and test data split")

    processed_data_saved(train,save_dir,"train")
    logger.info(f"Test train saved")

    processed_data_saved(test,save_dir,"test")
    logger.info(f"Processed test saved")
