import pandas as pd
import numpy as np
from sklearn.preprocessing.data import MaxAbsScaler, StandardScaler

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.svm.classes import SVC

from model.features import CustomTfidfVectorizer, Word2VecAverageContentVector, CustomTfidfVectorizerTitle, \
    Word2VecAverageTitleVector, Word2VecTitleContent, TypeTokenRatio, StopWordsCount, StopWordsTitle, WMDDistance, \
    PMI, CustomTfidfVectorizer_URL

df = pd.read_excel("../data/FN_Training_Set.xlsx")

train, test = train_test_split(df, test_size=0.2, random_state=999)
train = train.dropna(subset=['Content', 'Content Title'])

pipe = Pipeline([
    ('union', FeatureUnion([
        ('tfidf', CustomTfidfVectorizer(train)),
        ('tf-idf-url', CustomTfidfVectorizer_URL(train)),
        ('tf-idf_title', CustomTfidfVectorizerTitle(train)),
        ('w2v_vector_content', Word2VecAverageContentVector()),
        ('w2v_vector_title', Word2VecAverageTitleVector()),
        ('type_token', TypeTokenRatio()),
        ('w2v_title_content', Word2VecTitleContent()),
        ('sw', StopWordsCount()),
        ('sw_title', StopWordsTitle()),
        ('pmi', PMI())
        # ('wmd', WMDDistance())
    ])),
    ('scaler', MaxAbsScaler()),
    ('clf', LinearSVC(random_state=42))
])

print("training...")
pipe.fit_transform(train, train['click_bait_score'])
print("testing...")
score = pipe.score(test, test['click_bait_score'])

print(score)
