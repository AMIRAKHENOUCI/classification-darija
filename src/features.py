from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
def build_tfidf():
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3
    )
    return tfidf

def fit_transform_tfidf(texts):
    tfidf = build_tfidf()
    X = tfidf.fit_transform(texts)
    return X, tfidf


def get_vocabulary_info(texts):
    cv = CountVectorizer()
    cv.fit(texts)
    return len(cv.vocabulary_)