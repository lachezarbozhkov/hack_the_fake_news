# -*- coding: utf-8 -*-
from sklearn.pipeline import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

import numpy as np

W2V = Word2Vec.load("../data/embedings-bg/w2v_model")
SW = open('../data/stopwords.txt').read().split("\n")

class Feature(TransformerMixin):
    """Feature Interface."""
    def fit(self, X, y=None, **fit_params):
        return self


class Word2VecAverageContentVector(Feature):
    def transform(self, df):
        res = np.zeros((len(df.index), 300))
        for i, sent in enumerate(df['Content']):
            res[i] = np.average([W2V[word] for word in sent.lower().split(" ") if word in W2V])
        return res


class Word2VecTitleContent(Feature):
    def transform(self, df):
        out = []
        for i, row in df.iterrows():
            sent_v_c = np.average([W2V[word] for word in row['Content'].lower().split(" ") if word in W2V])
            sent_v_t = np.average([W2V[word] for word in row['Content Title'].lower().split(" ") if word in W2V])
            if not np.isnan(sent_v_t):
                out.append(1 - cosine(sent_v_c, sent_v_t))
            else:
                out.append(0)
        return np.array(out).reshape(len(df.index), 1)


class TypeTokenRatio(Feature):
    def transform(self, df):
        out = []
        for i, row in df.iterrows():
            tokens_c = len(row['Content'].lower().split(" "))
            types_c = len(set(row['Content'].lower().split(" ")))
            tokens_v = len(row['Content Title'].lower().split(" "))
            types_v = len(set(row['Content Title'].lower().split(" ")))
            out.append([types_c/tokens_c, types_v/tokens_v])
        return np.array(out).reshape(len(df.index), 2)


class Word2VecAverageTitleVector(Feature):
    def transform(self, df):
        res = np.zeros((len(df.index), 300))
        for i, sent in enumerate(df['Content Title']):
            sent_v = np.average([W2V[word] for word in sent.lower().split(" ") if word in W2V])
            if not np.isnan(sent_v):
                res[i] = sent_v
        return res


class CustomTfidfVectorizerTitle(Feature):
    def __init__(self, df):
        self.tr = TfidfVectorizer(max_features=500)
        data = [row for row in df['Content Title']]
        self.tr.fit_transform(data)

    def transform(self, df):
        res = self.tr.transform([row for row in df['Content Title']])
        return res

class WMDDistance(Feature):
    def transform(self, df):
        out = []
        for i, row in df.iterrows():
            out.append(W2V.wmdistance(row['Content'].lower().split(), row['Content Title'].lower().split()))
        return np.array(out).reshape(len(df.index), 1)

class StopWordsCount(Feature):
    def transform(self, df):
        return np.array([len([w for w in sent.lower().split() if w in SW]) for sent in df['Content']])\
            .reshape(len(df.index), 1)

class StopWordsTitle(Feature):
    def transform(self, df):
        return np.array([len([w for w in sent.lower().split() if w in SW]) for sent in df['Content Title']])\
            .reshape(len(df.index), 1)


class CustomTfidfVectorizer(Feature):
    def __init__(self, df):
        self.tr = TfidfVectorizer(max_features=500)
        data = [row for row in df['Content']]
        self.tr.fit_transform(data)

    def transform(self, df):
        return self.tr.transform([row for row in df['Content']])


