from nltk.corpus import stopwords
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from nltk.corpus import stopwords



class DataPreprocessing:
    @staticmethod
    def text_preprocessing(comment: str):
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
            return None