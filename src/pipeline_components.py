import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def fillna_text(x):
    return x.fillna("")


def fillna_cat(x):
    return x.fillna("missing")


class LeakSafeTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, smoothing=10):
        self.group_col = group_col
        self.smoothing = smoothing

    def fit(self, X, y=None):
        self.global_mean_ = X["price"].mean()
        stats = X.groupby(self.group_col)["price"].agg(["mean", "count"])
        stats["smoothed"] = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (stats["count"] + self.smoothing)
        self.mapping_ = stats["smoothed"]
        return self

    def transform(self, X):
        encoded = X[self.group_col].map(self.mapping_).fillna(self.global_mean_)
        return encoded.to_numpy().reshape(-1, 1)