# src/model_training.py
import os, joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from pathlib import Path

class ModelTrainer:
    def __init__(self, cfg, artifact_dir):
        self.cfg = cfg
        self.artifact_dir = artifact_dir
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, n_jobs=cfg.get('n_jobs', 1), random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, n_jobs=1, random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        Path(self.artifact_dir).mkdir(parents=True, exist_ok=True)

    def train_single(self, model_key, X_train, y_train, X_test, y_test, task_name, feature_set):
        model = self.models[model_key]
        if model_key in ('logistic_regression', 'svm'):
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
            X_test_proc = scaler.transform(X_test)
            joblib.dump(scaler, os.path.join(self.artifact_dir, f'scaler_{feature_set}_{task_name}_{model_key}.pkl'))
        else:
            X_train_proc, X_test_proc = X_train, X_test

        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
        acc = accuracy_score(y_test, y_pred)
        metrics = {
            'accuracy': acc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        model_path = os.path.join(self.artifact_dir, f"{feature_set}__{task_name}__{model_key}.pkl")
        joblib.dump(model, model_path)
        metrics_path = model_path.replace('.pkl', '.metrics.pkl')
        joblib.dump(metrics, metrics_path)
        return metrics, y_pred

    def run(self, X_train, X_test, y_train, y_test, feature_set, task_name):
        results = {}
        for key in self.models:
            metrics, y_pred = self.train_single(key, X_train, y_train, X_test, y_test, task_name, feature_set)
            results[key] = {'metrics': metrics, 'predictions': y_pred, 'model_path': os.path.join(self.artifact_dir, f"{feature_set}__{task_name}__{key}.pkl")}
        return results
