import pandas as pd

from btnlp_lib import SentimentAnalyzer

train_df = pd.read_csv(
    "../data/twitter_training.csv",
    names=["ID", "platform", "sentiment", "comment"],
    na_values=["<unk>"],
    dtype={"comment": str},
)
valid_df = pd.read_csv(
    "../data/twitter_validation.csv",
    names=["ID", "platform", "sentiment", "comment"],
    na_values=["<unk>"],
    dtype={"comment": str},
)

analyzer = SentimentAnalyzer(train_df, valid_df)

analyzer.summarize("sentiment")
analyzer.numerize_labels("sentiment")
analyzer.tokenize("comment")
analyzer.vader_sentiment("comment_tokenized")
analyzer.textblob_sentiment("comment_tokenized")
analyzer.huggingface_sentiment("comment_tokenized")
