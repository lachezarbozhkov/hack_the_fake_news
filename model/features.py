# -*- coding: utf-8 -*-
import re
import numpy as np
from urllib import parse

from sklearn.pipeline import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, LdaModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
import fasttext
import nltk.data
import re

JARGON_DICT = open("../data/dicts/jargon.txt").read().split("\n")
FOREIGN_DICT = open("../data/dicts/foreign_words.txt").read().split("\n")
EN_DICT = open("../data/dicts/en_words_with_bg_equivalents.txt").read().split("\n")

SENT_SPLITTER = nltk.data.load('tokenizers/punkt/english.pickle')
PERSONAL_PRONOUNS = ['аз', 'ти', 'той', 'тя', 'то', 'ние', 'вие', 'те', 'мен', 'мене', 'тебе', 'теб', 'него',
                     'нея', 'нас', 'вас', 'тях', 'ме', 'те', 'го', 'я', 'ни', 'ви', 'ги', 'нему', 'ней', 'нам',
                     'вам', 'тям', 'ми', 'му', 'ѝ', 'им']
VOWELS = 'аъоуеияю'



def load_pmi(name):
    res = dict()
    with open('../data/' + name, 'r') as pfile:
        r = pfile.read().split('\n')
        for line in r:
            split = line.split('\t')
            res[split[0]] = np.asarray(split[1:], dtype='float')
    return res

FAKE_DICT = Dictionary.load("../data/dict")
LDA = LdaModel.load("../data/lda/lda_model")
W2V = Word2Vec.load("../data/embedings-bg/w2v_model")
FASTTEXT_MODEL = None # fasttext.load_model('../data/fasttext/modelfst.bin')
FASTTEXT_SUP = None #fasttext.load_model("../data/fasttext/supervised_model.bin")
SW = open('../data/stopwords.txt').read().split("\n")
PMI_CONTENT_CLICKBAIT = load_pmi('pmi_content_clickbait')
PMI_CONTENT_FACT = load_pmi('pmi_content_fact')
PMI_HEADERS_CLICKBAIT = load_pmi('pmi_headers_clickbait')
PMI_HEADERS_FACT = load_pmi('pmi_headers_fact')
PMI_CONTENT_CLICKBAIT_NE = load_pmi('pmi_content_clickbait_NE')
PMI_CONTENT_FACT_NE = load_pmi('pmi_content_fact_NE')
PMI_HEADERS_CLICKBAIT_NE = load_pmi('pmi_headers_clickbait_NE')
PMI_HEADERS_FACT_NE = load_pmi('pmi_headers_fact_NE')
PMI_CONTENT_CLICKBAIT_BI = load_pmi('pmi_content_clickbait_BI')
PMI_CONTENT_FACT_BI = load_pmi('pmi_content_fact_BI')
PMI_HEADERS_CLICKBAIT_BI = load_pmi('pmi_headers_clickbait_BI')
PMI_HEADERS_FACT_BI = load_pmi('pmi_headers_fact_BI')

REGEX_CLEAN = '[\n„\".,!?“:\-\/_\xa0\(\)…]'
TYPOS = [word.strip() for word in open("../data/dicts/typos.txt").read().split("\n")]

class Feature(BaseEstimator, TransformerMixin):
    """Feature Interface."""
    def fit(self, X, y=None, **fit_params):
        return self


class Word2VecAverageContentVector(Feature):
    def transform(self, df):
        res = np.zeros((len(df.index), 300))
        for i, sent in enumerate(df['Content']):
            res[i] = np.average([W2V[word] for word in sent.lower().split(" ") if word in W2V])
        return res


class TyposCount(Feature):
    @staticmethod
    def count_typos(text):
        return len([word for word in text.lower().split(" ") if word in TYPOS])

    def transform(self, df):
        return df.Content.apply(lambda sent: self.count_typos(sent)).values.reshape(len(df), 1)


class EnglishInTitle(Feature):
    def transform(self, df):
        return df['Content Title'].apply(lambda title:
                                         int(re.match(pattern='[a-zA-Z]+', string=title) != None))\
            .values.reshape(len(df), 1)


class FastTextSupervised(Feature):
    def __init__(self):
        global FASTTEXT_SUP
        if not FASTTEXT_SUP:
            FASTTEXT_SUP = fasttext.load_model("../data/fasttext/supervised_model.bin")

    def transform(self, df):
        res = []
        for sent in df['Content']:
            pred_prob = FASTTEXT_SUP.predict_proba(sent)[0][0]
            # print(pred_prob[0][0])
            if pred_prob[0] == '__label__1':
                res.append(1)
            else:
                res.append(3)
            # if pred_prob[0] == '__label__1':
            #     res.append(1-pred_prob[1])
            # else:
            #     res.append(pred_prob[1])
        # print(res)
        return np.array(res).reshape((len(df), 1))


class FastTextAverageContentVector(Feature):
    def __init__(self):
        global FASTTEXT_MODEL
        if not FASTTEXT_MODEL:
            FASTTEXT_MODEL = fasttext.load_model('../data/fasttext/modelfst.bin')

    def transform(self, df):
        res = np.zeros((len(df.index), 300))
        for i, sent in enumerate(df['Content']):
            res[i] = np.average([FASTTEXT_MODEL[word] for word in sent.lower().split(" ") if word in FASTTEXT_MODEL])
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
            title = re.sub(REGEX_CLEAN, '', str(row['Content'])).lower()
            content = re.sub(REGEX_CLEAN, '', str(row['Content Title'])).lower()
            tokens_content = title.split(" ")
            tokens_title = content.split(" ")

            pmi_header_bait = []
            pmi_header_NONbait = []
            pmi_header_fact = []
            pmi_header_NONfact = []
            pmi_content_bait = []
            pmi_content_NONbait = []
            pmi_content_fact = []
            pmi_content_NONfact = []


            pmi_header_bait_NE = [0]
            pmi_header_NONbait_NE = [0]
            pmi_header_fact_NE = [0]
            pmi_header_NONfact_NE = [0]
            pmi_content_bait_NE = [0]
            pmi_content_NONbait_NE = [0]
            pmi_content_fact_NE = [0]
            pmi_content_NONfact_NE = [0]

            pmi_header_bait_BI = [0]
            pmi_header_NONbait_BI = [0]
            pmi_header_fact_BI = [0]
            pmi_header_NONfact_BI = [0]
            pmi_content_bait_BI = [0]
            pmi_content_NONbait_BI = [0]
            pmi_content_fact_BI = [0]
            pmi_content_NONfact_BI = [0]


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

            for ne in PMI_HEADERS_CLICKBAIT_NE:
                if ne in title and len(PMI_HEADERS_CLICKBAIT_NE[ne]) == 2:
                    pmi_header_bait_NE.append(PMI_HEADERS_CLICKBAIT_NE[ne][0])
                    pmi_header_NONbait_NE.append(PMI_HEADERS_CLICKBAIT_NE[ne][1])
            for ne in PMI_HEADERS_FACT_NE:
                if ne in title and len(PMI_HEADERS_FACT_NE[ne]) == 2:
                    pmi_header_fact_NE.append(PMI_HEADERS_FACT_NE[ne][0])
                    pmi_header_NONfact_NE.append(PMI_HEADERS_FACT_NE[ne][1])
            for ne in PMI_CONTENT_CLICKBAIT_NE:
                if ne in content and len(PMI_CONTENT_CLICKBAIT_NE[ne]) == 2:
                    pmi_content_bait_NE.append(PMI_CONTENT_CLICKBAIT_NE[ne][0])
                    pmi_content_NONbait_NE.append(PMI_CONTENT_CLICKBAIT_NE[ne][1])
            for ne in PMI_CONTENT_FACT_NE:
                if ne in content and len(PMI_CONTENT_FACT_NE[ne]) == 2:
                    pmi_content_fact_NE.append(PMI_CONTENT_FACT_NE[ne][0])
                    pmi_content_NONfact_NE.append(PMI_CONTENT_FACT_NE[ne][1])


            for ne in PMI_HEADERS_CLICKBAIT_BI:
                if ne in title and len(PMI_HEADERS_CLICKBAIT_BI[ne]) == 2:
                    pmi_header_bait_BI.append(PMI_HEADERS_CLICKBAIT_BI[ne][0])
                    pmi_header_NONbait_BI.append(PMI_HEADERS_CLICKBAIT_BI[ne][1])
            for ne in PMI_HEADERS_FACT_BI:
                if ne in title and len(PMI_HEADERS_FACT_BI[ne]) == 2:
                    pmi_header_fact_BI.append(PMI_HEADERS_FACT_BI[ne][0])
                    pmi_header_NONfact_BI.append(PMI_HEADERS_FACT_BI[ne][1])
            for ne in PMI_CONTENT_CLICKBAIT_BI:
                if ne in content and len(PMI_CONTENT_CLICKBAIT_BI[ne]) == 2:
                    pmi_content_bait_BI.append(PMI_CONTENT_CLICKBAIT_BI[ne][0])
                    pmi_content_NONbait_BI.append(PMI_CONTENT_CLICKBAIT_BI[ne][1])
            for ne in PMI_CONTENT_FACT_BI:
                if ne in content and len(PMI_CONTENT_FACT_BI[ne]) == 2:
                    pmi_content_fact_BI.append(PMI_CONTENT_FACT_BI[ne][0])
                    pmi_content_NONfact_BI.append(PMI_CONTENT_FACT_BI[ne][1])

            #print(pmi_content_NONfact_BI)
            res = [
                    max(pmi_header_bait), np.mean(pmi_header_bait),
                    max(pmi_header_NONbait), np.mean(pmi_header_NONbait),
                    max(pmi_header_fact), np.mean(pmi_header_fact),
                    max(pmi_header_NONfact), np.mean(pmi_header_NONfact),
                    max(pmi_content_bait), np.mean(pmi_content_bait),
                    max(pmi_content_NONbait), np.mean(pmi_content_NONbait),
                    max(pmi_content_fact), np.mean(pmi_content_fact),
                    max(pmi_content_NONfact), np.mean(pmi_content_NONfact),

                    max(pmi_header_bait_NE), np.mean(pmi_header_bait_NE),
                    max(pmi_header_NONbait_NE), np.mean(pmi_header_NONbait_NE),
                    max(pmi_header_fact_NE), np.mean(pmi_header_fact_NE),
                    max(pmi_header_NONfact_NE), np.mean(pmi_header_NONfact_NE),
                    max(pmi_content_bait_NE), np.mean(pmi_content_bait_NE),
                    max(pmi_content_NONbait_NE), np.mean(pmi_content_NONbait_NE),
                    max(pmi_content_fact_NE), np.mean(pmi_content_fact_NE),
                    max(pmi_content_NONfact_NE), np.mean(pmi_content_NONfact_NE),

                    max(pmi_header_bait_BI), np.mean(pmi_header_bait_BI),
                    max(pmi_header_NONbait_BI), np.mean(pmi_header_NONbait_BI),
                    max(pmi_header_fact_BI), np.mean(pmi_header_fact_BI),
                    max(pmi_header_NONfact_BI), np.mean(pmi_header_NONfact_BI),
                    max(pmi_content_bait_BI), np.mean(pmi_content_bait_BI),
                    max(pmi_content_NONbait_BI), np.mean(pmi_content_NONbait_BI),
                    max(pmi_content_fact_BI), np.mean(pmi_content_fact_BI),
                    max(pmi_content_NONfact_BI), np.mean(pmi_content_NONfact_BI)
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

TFIDF_CONTENT = None
TFIDF_TITLE = None
TFIDF_URL = None


class CustomTfidfVectorizerTitle(Feature):
    def transform(self, df):
        global TFIDF_TITLE
        if not TFIDF_TITLE:
            TFIDF_TITLE = TfidfVectorizer(max_features=500)
            data = [row for row in df['Content Title']]
            TFIDF_TITLE.fit(data)
        res = TFIDF_TITLE.transform([row for row in df['Content Title']])
        return res


class CustomTfidfVectorizer(Feature):
    def transform(self, df):
        global TFIDF_CONTENT
        if not TFIDF_CONTENT:
            TFIDF_CONTENT = TfidfVectorizer(max_features=500)
            data = [row for row in df['Content']]
            TFIDF_CONTENT.fit(data)
        return TFIDF_CONTENT.transform([row for row in df['Content']])


class LDAVectorContent(Feature):
    def transform(self, df):
        res = np.zeros((len(df.index), 10))
        for i, sent in enumerate(df['Content']):
            sent_v = [t[1] for t in LDA.get_document_topics(FAKE_DICT.doc2bow(sent.lower().split(" ")), minimum_probability=-1)]
            # if not np.isnan(sent_v):
            res[i] = np.array(sent_v).reshape(10)
        return res


class LDAVectorContentTitle(Feature):
    def transform(self, df):
        res = []
        for i, sent in df.iterrows():
            sent_t = [t[1] for t in
                      LDA.get_document_topics(FAKE_DICT.doc2bow(sent['Content'].lower().split(" ")), minimum_probability=-1)]
            sent_c = [t[1] for t in
                      LDA.get_document_topics(FAKE_DICT.doc2bow(sent['Content Title'].lower().split(" ")), minimum_probability=-1)]

            sim = cosine(sent_t, sent_c)

            # if not np.isnan(sent_v):
            res.append(sim)
        return np.array(res).reshape(len(df.index), 1)

class LDAVectorTitle(Feature):
    def transform(self, df):
        res = np.zeros((len(df.index), 10))
        for i, sent in enumerate(df['Content Title']):
            sent_v = [t[1] for t in
                      LDA.get_document_topics(FAKE_DICT.doc2bow(sent.lower().split(" ")), minimum_probability=-1)]
            # if not np.isnan(sent_v):
            res[i] = np.array(sent_v).reshape(10)
        return res


class CustomTfidfVectorizer_URL(Feature):
    def transform(self, df):
        global TFIDF_URL
        if not TFIDF_URL:
            TFIDF_URL = TfidfVectorizer(max_features=300)
            data = [row for row in df['Content Url']]
            TFIDF_URL.fit(data)
        return TFIDF_URL.transform([row for row in df['Content Url']])


class Dicts(Feature):
    def transform(self, df):
        results = np.zeros((len(df), 6))
        for i, row in enumerate(df['Content']):
            results[i][0] = sum([1 for word in row.lower().split(" ") if word in JARGON_DICT])
            results[i][1] = sum([1 for word in row.lower().split(" ") if word in EN_DICT])
            results[i][2] = sum([1 for word in row.lower().split(" ") if word in FOREIGN_DICT])
        for i, row in enumerate(df['Content Title']):
            results[i][3] = sum([1 for word in row.lower().split(" ") if word in JARGON_DICT])
            results[i][4] = sum([1 for word in row.lower().split(" ") if word in EN_DICT])
            results[i][5] = sum([1 for word in row.lower().split(" ") if word in FOREIGN_DICT])
        return results

class CountingWords(Feature):

    def extract_urls(self, text):
        """Return a list of urls from a text string. False positives exist."""
        out = []
        for word in text.split(' '):
            thing = parse.urlparse(word.strip())
            if thing.scheme:
                out.append(word)
        return out

    def sim_title_content(self, title, content):
        title_words = title.lower().split()
        content_words = content.lower().split()
        common_words = sum([word in content_words for word in title_words])
        return common_words / len(title_words)

    def transform(self, df):
        df = df.copy()
        df['Content'] = df['Content'].astype(str)
        df['Content Title'] = df['Content Title'].astype(str)

        df['len_words'] = df['Content'].apply(lambda t: len(t.split()))
        df['len_words_title'] = df['Content Title'].apply(lambda t: len(t.split()))

        df['len_chars'] = df['Content'].apply(len)
        df['len_chars_title'] = df['Content Title'].apply(len)

        df['len_symbols'] = df['Content'].apply(lambda t: sum([c in ['$.!;#?:-+@%^&*(),'] for c in t]))
        df['len_symbols_title'] = df['Content Title'].apply(lambda t: sum([c in ['$.!;#?:-+@%^&*(),'] for c in t]))

        df['len_capitals'] = df['Content'].apply(lambda t: sum([str.isupper(c) for c in t]))
        df['len_capitals_title'] = df['Content Title'].apply(lambda t: sum([str.isupper(c) for c in t]))

        df['fraction_capitals'] = (df['len_capitals'] + 1) / (df['len_chars'] + 1)
        df['fraction_capitals_title'] = (df['len_capitals_title'] + 1) / (df['len_chars_title'] + 1)

        df['len_url'] = df.Content.apply(lambda t: len(self.extract_urls(t)))
        df['title_sim'] = df.apply(lambda row: self.sim_title_content(row['Content Title'], row['Content']), axis=1)

        columns = ['len_words', 'len_words_title',
                'len_chars', 'len_chars_title', 'len_symbols', 'len_symbols_title',
                'len_capitals', 'len_capitals_title', 'fraction_capitals',
                'fraction_capitals_title', 'len_url', 'title_sim']

        return df[columns]


def num_syllables(word):
    return sum([word.count(vowel) for vowel in VOWELS])


def num_pronouns(split_sentences):
    return sum([sent.count(pronoun) for sent in split_sentences for pronoun in PERSONAL_PRONOUNS])


class ReadabilityFeatures(Feature):
    """
    1.    Average sentence length (ASL)
    2.    Number of different hard words
    3.    Number of personal pronouns (PP)
    4.    Percentage of unique words (TYPES)
    5.    Number of prepositional phrases
    6.    Average word length (syllables) (ASW)
    7.    Flesch–Kincaid readability tests : 206.835 − (1.015 × ASL) − (84.6 × ASW)
    8.    (Gunning Fog Index)
          Grade level= 0.4 * ( (average sentence length) + (percentage of Hard Words) )
          Where: Hard Words = words with more than two syllables.
    9.    SMOG grading = 3 + √polysyllable count.
          Where polysyllable count = number of words of more than two syllables in a sample of 30 sentences
    10.   Forecast : Grade level = 20 − (N / 10)
          Where N = number of single-syllable words in a 150-word sample.

    Reference: https://en.wikipedia.org/wiki/Readability

    """

    def transform(self, df):
        result = []
        for i, row in df.iterrows():
            sentences = SENT_SPLITTER.tokenize(row['Content'])
            words_in_sents = [sent.lower().split(" ") for sent in sentences]
            words = [word for sent in words_in_sents for word in sent]
            asl = np.average([len(sent) for sent in words_in_sents])
            asw = np.average([num_syllables(word) for word in words])
            pp = num_pronouns(words_in_sents)
            types = len(set(words))
            fkr = 206.835 - (1.015 * asl) - (84.6 * asw)
            polly = len([word for sent in words_in_sents[:30] for word in sent if num_syllables(word) > 2])
            smog = polly ** (-3) + 3
            N = len([word for word in words[:150] if num_syllables(word) == 1])
            forecast = 20 - (N / 10)

            result.append([asl, asw, pp, types, fkr, smog, forecast])

        return np.array(result).reshape((len(result), len(result[0])))