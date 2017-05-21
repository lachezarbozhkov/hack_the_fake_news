# -*- coding: utf-8 -*-
from sklearn.pipeline import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import re

import numpy as np

def load_pmi(name):
    res = dict()
    with open('../data/' + name, 'r') as pfile:
        r = pfile.read().split('\n')
        for line in r:
            split = line.split('\t')
            res[split[0]] = np.asarray(split[1:], dtype='float')
    return res



W2V = Word2Vec.load("../data/embedings-bg/w2v_model")
SW = open('../data/stopwords.txt').read().split("\n")
PMI_CONTENT_CLICKBAIT = load_pmi('pmi_content_clickbait')
PMI_CONTENT_FACT = load_pmi('pmi_content_fact')
PMI_HEADERS_CLICKBAIT = load_pmi('pmi_headers_clickbait')
PMI_HEADERS_FACT = load_pmi('pmi_headers_fact')
REGEX_CLEAN = '[\n„\".,!?“:\-\/_\xa0\(\)…]'

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

class PMI(Feature):
    def transform(self, df):

        out = []
        for i, row in df.iterrows():
            res = []
            tokens_content = re.sub(REGEX_CLEAN, '', str(row['Content'])).lower().split(" ")
            tokens_title = re.sub(REGEX_CLEAN, '', str(row['Content Title'])).lower().split(" ")

            pmi_header_bait = []
            pmi_header_NONbait = []
            pmi_header_fact = []
            pmi_header_NONfact = []
            pmi_content_bait = []
            pmi_content_NONbait = []
            pmi_content_fact = []
            pmi_content_NONfact = []

            for token in tokens_title:
                if token in PMI_HEADERS_CLICKBAIT and len(PMI_HEADERS_CLICKBAIT[token]) == 2:
                    pmi_header_bait.append(PMI_HEADERS_CLICKBAIT[token][0])
                    pmi_header_NONbait.append(PMI_HEADERS_CLICKBAIT[token][1])
                if token in PMI_HEADERS_FACT and len(PMI_HEADERS_FACT[token]) == 2:
                    pmi_header_fact.append(PMI_HEADERS_FACT[token][1])
                    pmi_header_NONfact.append(PMI_HEADERS_FACT[token][0])

            for token in tokens_content:
                if token in PMI_CONTENT_CLICKBAIT and len(PMI_CONTENT_CLICKBAIT[token]) == 2:
                    pmi_content_bait.append(PMI_CONTENT_CLICKBAIT[token][0])
                    pmi_content_NONbait.append(PMI_CONTENT_CLICKBAIT[token][1])
                if token in PMI_CONTENT_FACT and len(PMI_CONTENT_FACT[token]) == 2:
                    pmi_content_fact.append(PMI_CONTENT_FACT[token][1])
                    pmi_content_NONfact.append(PMI_CONTENT_FACT[token][0])

            if len(pmi_header_NONfact) == 0:
                pmi_header_NONfact.append(0)
                pmi_header_fact.append(0)
                pmi_header_bait.append(0)
                pmi_header_NONbait.append(0)


            res = [max(pmi_header_bait), np.mean(pmi_header_bait),
                   max(pmi_header_NONbait), np.mean(pmi_header_NONbait),
                   max(pmi_header_fact), np.mean(pmi_header_fact),
                   max(pmi_header_NONfact), np.mean(pmi_header_NONfact),
                   max(pmi_content_bait), np.mean(pmi_content_bait),
                   max(pmi_content_NONbait), np.mean(pmi_content_NONbait),
                   max(pmi_content_fact), np.mean(pmi_content_fact),
                   max(pmi_content_NONfact), np.mean(pmi_content_NONfact)
                   ]

            out.append(res)
        return out

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


