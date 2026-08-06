import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

train["name"] = train["name"].fillna("")
val["name"] = val["name"].fillna("")
train["item_description"] = train["item_description"].fillna("")
val["item_description"] = val["item_description"].fillna("")

name_vectorizer = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2),
    min_df=2, max_df=0.9, stop_words="english"
)

desc_vectorizer = TfidfVectorizer(
    max_features=20000, ngram_range=(1, 2),
    min_df=2, max_df=0.9, stop_words="english"
)

X_name_train = name_vectorizer.fit_transform(train["name"])
X_name_val = name_vectorizer.transform(val["name"])

X_desc_train = desc_vectorizer.fit_transform(train["item_description"])
X_desc_val = desc_vectorizer.transform(val["item_description"])