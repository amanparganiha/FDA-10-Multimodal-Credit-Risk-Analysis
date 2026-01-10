# src/nlp_features.py
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import numpy as np
import pandas as pd

def compute_tfidf(corpus, max_features=5000, ngram_range=(1,2)):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_tfidf = vec.fit_transform(corpus)
    return vec, X_tfidf

def save_vectorizer(vec, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vec, path)

def transform_with_vectorizer(vec, corpus):
    return vec.transform(corpus)

def compute_simple_nlp_scores(text_series):
    df = pd.DataFrame({
        'tok_count': text_series.fillna('').str.split().map(len),
        'avg_word_len': text_series.fillna('').map(lambda s: np.mean([len(w) for w in s.split()]) if s.strip() else 0)
    })
    return df
