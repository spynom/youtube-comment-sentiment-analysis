import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def read_file(*args)->pd.DataFrame:
    """:parameter input directories in sequence
        :return DataFrame"""

    path =os.path.join(*args)

    return pd.read_csv(path)

def load_data()->[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    df = read_file("data", "processed", "cleaned.csv").dropna(how="any").drop_duplicates()

    X = df.comment
    y = df.category + 1

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def apply_feature_engineering(X_train, X_test, y_train, y_test)->[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    # Vectorization using TF-IDF with varying max_features
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)


    x_train = vectorizer.fit_transform(X_train)
    x_test = vectorizer.transform(X_test)
    x_train = pd.DataFrame(x_train.toarray(), columns=vectorizer.get_feature_names_out())
    x_test = pd.DataFrame(x_test.toarray(), columns=vectorizer.get_feature_names_out())

    y_train = y_train.to_frame().reset_index(drop=True)  # if y_train is a Series
    y_test = y_test.to_frame().reset_index(drop=True)  # if y_test is a Series

    return x_train, x_test, y_train, y_test

def main():
    x_train, x_test, y_train, y_test = load_data()

    x_train, x_test, y_train, y_test = apply_feature_engineering(x_train, x_test, y_train, y_test)

    train_path = os.path.join("data", "final", "train.csv")
    test_path = os.path.join("data", "final", "test.csv")

    pd.concat([x_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([x_test, y_test], axis=1).to_csv(test_path, index=False)

if __name__ == '__main__':
    main()





