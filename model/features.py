# -*- coding: utf-8 -*-
from sklearn.pipeline import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class Feature(TransformerMixin):
    """Feature Interface."""
    def fit(self, X, y=None, **fit_params):
        return self


class CustomTfidfVectorizer(Feature):
    def __init__(self, df):
        self.tr = TfidfVectorizer()
        data = [row for row in df['Content']]
        self.tr.fit_transform(data)

    def transform(self, df):
        return self.tr.transform([row for row in df['Content']])


