"""
Train 5 classifiers on the UCI Spambase dataset and save models + metrics.

Usage:
    python train_models.py

Outputs:
    model/<name>.joblib        - one tuned Pipeline(log1p + StandardScaler + classifier) per model
    model/feature_names.joblib - list of the 57 feature names (column order)
    model/metrics_summary.csv  - Accuracy, AUC, Precision, Recall, F1, MCC per model, plus CV_F1
    model/best_params.csv      - best hyperparameters from GridSearchCV
    test_data.csv              - held-out 20% test set (raw features + "spam" target)
"""

import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
)
DATA_FILE = "spambase.data"
MODEL_DIR = "model"
TEST_FILE = "test_data.csv"
TARGET = "spam"
RANDOM_STATE = 5709  # last 4 digits of my BITS ID (2026AJ05709)

WORDS = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses", "free",
    "business", "email", "you", "credit", "your", "font", "000", "money",
    "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct",
    "cs", "meeting", "original", "project", "re", "edu", "table", "conference",
]
CHARS = [";", "(", "[", "!", "$", "#"]
FEATURE_NAMES = (
    [f"word_freq_{w}" for w in WORDS]
    + [f"char_freq_{c}" for c in CHARS]
    + [
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
    ]
)

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
}

# Hyperparameter grids searched with 5-fold CV on the training set only
PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "Decision Tree": {"clf__max_depth": [5, 10, 15, None], "clf__min_samples_leaf": [1, 5, 10]},
    "KNN": {"clf__n_neighbors": [3, 5, 7, 9, 11], "clf__weights": ["uniform", "distance"]},
    "Naive Bayes": {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]},
    "SVM": {"clf__C": [0.5, 1, 5, 10], "clf__gamma": ["scale", 0.01, 0.05]},
}


def model_filename(name):
    return name.lower().replace(" ", "_") + ".joblib"


def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    df = pd.read_csv(DATA_FILE, header=None, names=FEATURE_NAMES + [TARGET])
    print(f"Loaded {df.shape[0]} rows x {df.shape[1] - 1} features")
    return df


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    df = load_data()
    X, y = df[FEATURE_NAMES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    rows, params = [], []
    for name, clf in MODELS.items():
        pipe = Pipeline([
            ("log", FunctionTransformer(np.log1p)),  # features are heavily right-skewed
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        search = GridSearchCV(pipe, PARAM_GRIDS[name], cv=5, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        metrics = evaluate(best, X_test, y_test)
        rows.append({"Model": name, **metrics, "CV_F1": search.best_score_})
        params.append({"Model": name, "Best params": str(search.best_params_)})
        joblib.dump(best, os.path.join(MODEL_DIR, model_filename(name)))
        print(f"{name:<20} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    joblib.dump(FEATURE_NAMES, os.path.join(MODEL_DIR, "feature_names.joblib"))
    summary = pd.DataFrame(rows).round(4)
    summary.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)
    pd.DataFrame(params).to_csv(os.path.join(MODEL_DIR, "best_params.csv"), index=False)

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(TEST_FILE, index=False)

    print("\nMetrics summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved models to {MODEL_DIR}/ and test set ({len(test_df)} rows) to {TEST_FILE}")


if __name__ == "__main__":
    main()
