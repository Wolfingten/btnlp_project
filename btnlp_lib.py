import re
import string

import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report
from textblob import TextBlob
from tqdm import tqdm
from transformers import pipeline

# set global options
pd.options.mode.copy_on_write = True
tqdm.pandas()


class SentimentAnalyzer:
    """
    A class that performs sentiment analysis on text data using various methods.

    Args:
        train_df (pandas.DataFrame): DataFrame containing training data.
        valid_df (pandas.DataFrame): DataFrame containing validation data.
    """

    def __init__(self, train_df, valid_df):
        # find id columns
        id_cols = []
        pattern = r"(^id$|^id_|\bid$)"
        for column in train_df.columns:
            if re.match(pattern, column, flags=re.IGNORECASE):
                id_cols.append(column)
        # removing duplicate items
        # dramatically reduces dataset size and improves performance
        self.train_df = train_df.drop_duplicates(subset=id_cols)
        self.valid_df = valid_df.drop_duplicates(subset=id_cols)
        self.true_labels = None

    def summarize(self, label_column):
        """
        Provides a short overview of the dataframe and the target label distribution.

        Args:
            label_column (str): Name of the column containing sentiment labels.
        """
        print(self.train_df.head())
        print(self.train_df.describe())
        plt.hist(self.train_df[label_column])
        plt.show()

    def numerize_labels(self, label_column):
        """
        Converts string labels to numerical values (1 for positive, 0 for neutral, -1 for negative).

        Args:
            label_column (str): Name of the column containing sentiment labels.
        """
        invalid_labels = []
        for label in self.train_df[label_column].unique():
            if label.lower() in ["positive", "pos"]:
                pos_label = label
            elif label.lower() in ["negative", "neg"]:
                neg_label = label
            elif label.lower() in ["neutral", "neut", "neu"]:
                neu_label = label
            else:
                invalid_labels.append(label)

        self.train_df = self.train_df[~self.train_df[label_column].isin(invalid_labels)]
        self.valid_df = self.valid_df[~self.valid_df[label_column].isin(invalid_labels)]

        label_mapping = {pos_label: 1, neu_label: 0, neg_label: -1}

        self.true_labels = label_column + "_numerized"
        self.train_df[self.true_labels] = self.train_df[label_column].replace(
            label_mapping
        )
        self.valid_df[self.true_labels] = self.valid_df[label_column].replace(
            label_mapping
        )

    def tokenize(self, column_name):
        """
        Tokenizes text data and performs various cleaning steps prior.

        Args:
            column_name (str): Name of the column containing text data.
        """

        def clean_text(text):
            if isinstance(text, str):  # skip nan
                text = re.sub(
                    r"[^\x00-\x7F]|<unk>|\[UNK\]", "", text
                )  # remove non-ascii + inline unk tokens
                text = re.sub(
                    r"\b\S*[a-zA-Z]+\S*[^\W\s]+\.[^\W\s]+\S*[a-zA-Z]+\S*\b",
                    "http",
                    text,
                )  # mask urls
                text = re.sub(r"^@\w+", "@user", text)  # mask usernames
                text = text.lower().strip()  # lowercase + remove trailing whitespaces
                text = text.translate(
                    str.maketrans("", "", string.punctuation)
                )  # remove punctuation
                text = word_tokenize(text, language="english")  # tokenize
                text = [
                    t for t in text if t not in stopwords.words("english")
                ]  # remove stopwords
                if len(text) < 1:
                    text = pd.NA
                else:
                    text = " ".join(text)
            return text

        print(f"tokenizing column {column_name}\n")
        print("train df:\n")
        self.train_df[column_name + "_tokenized"] = self.train_df[
            column_name
        ].progress_apply(clean_text)
        self.train_df = self.train_df.dropna()
        print("validation df: \n")
        self.valid_df[column_name + "_tokenized"] = self.valid_df[
            column_name
        ].progress_apply(clean_text)
        self.valid_df = self.valid_df.dropna()

    def vader_sentiment(self, in_column):
        """
        Performs sentiment analysis using VADER sentiment analyzer.

        Args:
            in_column (str): Name of the column containing text data.

        Returns:
            numpy.ndarray: Confusion matrix of true labels vs. predicted sentiments.
        """
        nltk.download("vader_lexicon")
        vader_sentiment = []
        for sentence in self.valid_df[in_column]:
            sid = SentimentIntensityAnalyzer()
            sent_scores = sid.polarity_scores(sentence)
            compound_score = sent_scores["compound"]
            if compound_score >= 0.5:
                vader_sentiment.append(1)
            elif compound_score <= -0.5:
                vader_sentiment.append(-1)
            else:
                vader_sentiment.append(0)
        self.valid_df["vader_sentiment"] = vader_sentiment
        print(
            classification_report(
                self.valid_df[self.true_labels], self.valid_df["vader_sentiment"]
            )
        )
        return confusion_matrix(
            self.valid_df[self.true_labels], self.valid_df["vader_sentiment"]
        )

    def textblob_sentiment(self, in_column):
        """
        Performs sentiment analysis using TextBlob library.

        Args:
            in_column (str): Name of the column containing text data.

        Returns:
            numpy.ndarray: Confusion matrix of true labels vs. predicted sentiments.
        """
        textblob_sentiment = []
        for sentence in self.valid_df[in_column]:
            blob = TextBlob(sentence)
            sent_score = blob.sentiment.polarity
            if sent_score >= 0.5:
                textblob_sentiment.append(1)
            elif sent_score <= -0.5:
                textblob_sentiment.append(-1)
            else:
                textblob_sentiment.append(0)
        self.valid_df["textblob_sentiment"] = textblob_sentiment
        print(
            classification_report(
                self.valid_df[self.true_labels], self.valid_df["textblob_sentiment"]
            )
        )
        return confusion_matrix(
            self.valid_df[self.true_labels], self.valid_df["textblob_sentiment"]
        )

    def huggingface_sentiment(self, in_column):
        """
        Performs sentiment analysis using a Hugging Face transformer model.

        Args:
            in_column (str): Name of the column containing text data.

        Returns:
            numpy.ndarray: Confusion matrix of true labels vs. predicted sentiments.
        """
        pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )
        label_mapping = {"positive": 1, "neutral": 0, "negative": -1}
        huggingface_sentiment = []
        for sentence in tqdm(self.valid_df[in_column]):
            sent_score = pipe(sentence)
            huggingface_sentiment.append(label_mapping[sent_score[0]["label"]])

        self.valid_df["huggingface_sentiment"] = huggingface_sentiment
        print(
            classification_report(
                self.valid_df[self.true_labels], self.valid_df["huggingface_sentiment"]
            )
        )
        return confusion_matrix(
            self.valid_df[self.true_labels], self.valid_df["huggingface_sentiment"]
        )
