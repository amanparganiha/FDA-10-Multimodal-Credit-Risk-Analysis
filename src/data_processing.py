# src/data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

def load_processed(path):
    return pd.read_csv(path)

def prepare_targets(df, rating_col='rating', inv_grade_col='investment_grade'):
    y_binary = df[inv_grade_col].astype(int)
    y_multi = df[rating_col].astype(str)
    return y_binary, y_multi

def prepare_features(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['adsh','company_name','rating','investment_grade','financial_score']
    features = [c for c in df.columns if c not in exclude_cols]
    X = df[features]
    return X, features

def split_for_tasks(X, y_binary, y_multi, test_size=0.2, random_state=42):
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_binary, test_size=test_size, stratify=y_binary, random_state=random_state
    )
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X, y_multi, test_size=test_size, stratify=y_multi, random_state=random_state
    )
    return (X_train_b, X_test_b, y_train_b, y_test_b,
            X_train_m, X_test_m, y_train_m, y_test_m)

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
