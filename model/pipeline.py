import pandas as pd
import numpy as np
from sklearn.preprocessing.data import MaxAbsScaler, StandardScaler

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

from features import CustomTfidfVectorizer, Word2VecAverageContentVector, CustomTfidfVectorizerTitle, \
    Word2VecAverageTitleVector, Word2VecTitleContent, TypeTokenRatio, StopWordsCount, StopWordsTitle, WMDDistance, \
    PMI, CustomTfidfVectorizer_URL, LDAVectorContent, CountingWords, FastTextSupervised, ReadabilityFeatures, Dicts, \
    FastTextAverageContentVector


df = pd.read_excel("../data/FN_Training_Set.xlsx")

train, test = train_test_split(df, test_size=0.2, random_state=999)

tuned_parameters = [{'clf__kernel': ['rbf'], 'clf__gamma': [1e-3, 1e-4],
                     'clf__C': [1,  16, 32, 64, 100, 1000]}]


train = train.dropna(subset=['Content', 'Content Title'])

pipe = Pipeline([
    ('union', FeatureUnion([
        ('tfidf', CustomTfidfVectorizer()),
        # ('tf-idf-url', CustomTfidfVectorizer_URL()),
        ('tf-idf_title', CustomTfidfVectorizerTitle()),
        ('w2v_vector_content', Word2VecAverageContentVector()),
        ('w2v_vector_title', Word2VecAverageTitleVector()),
        ('type_token', TypeTokenRatio()),
        ('w2v_title_content', Word2VecTitleContent()),
        ('sw', StopWordsCount()),
        ('sw_title', StopWordsTitle()),
        ('pmi', PMI()),
        ('lda', LDAVectorContent()),
        ('CountingWords', CountingWords()),
        ('readability', ReadabilityFeatures()),
        # ('fastext_sup', FastTextSupervised()),
        # ('fast_text', FastTextAverageContentVector()),
        # ('dicts', Dicts()),
        # ('wmd', WMDDistance()),
    ])),
    ('scaler', MaxAbsScaler()),
    ('clf', LinearSVC())
])

# grid_search = GridSearchCV(pipe, tuned_parameters, cv=5,
#                         scoring='accuracy', verbose=1, n_jobs=-1)
# grid_search.fit(train, train['click_bait_score'])


# print(grid_search.best_params_)


print("training...")
pipe.fit_transform(train, train['click_bait_score'])
print("testing...")
score = pipe.score(test, test['click_bait_score'])

print(score)
