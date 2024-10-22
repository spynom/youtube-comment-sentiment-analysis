from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import pandas as pd
import os
import numpy as np

stop_words = set(stopwords.words('english')) - {"not", "however", "although", "but"}


def read_file(*args, file_type="csv") -> pd.DataFrame:
    """:parameter input directories in sequence
        :return DataFrame"""

    path:str = os.path.join(*args)

    return pd.read_csv(path).rename(columns={'clean_comment':'comment'})


def preprocess_comment(comment):
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

if __name__ == '__main__':
    raw_data = read_file("data","raw","reddit.csv").dropna(how='any')

    raw_data=raw_data.assign(
        comment=raw_data.comment.apply(preprocess_comment)
                            ).dropna(how='any').drop_duplicates()

    raw_data.to_csv(os.path.join("data", "processed", "cleaned.csv"), index=False)
