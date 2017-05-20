import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split

from model.features import CustomTfidfVectorizer

df = pd.read_excel("../data/FN_Training_Set.xlsx")

train, test = train_test_split(df, test_size=0.1, random_state=999)
train = train.dropna(subset=['Content'])

pipe = Pipeline([
    ('union', FeatureUnion([
        ('tfidf', CustomTfidfVectorizer(train))
    ])),
    ('clf', LinearSVC())
])

pipe.fit_transform(train, train['fake_news_score'])
score = pipe.score(test, test['fake_news_score'])

print(score)
