"""
Streamlit app: evaluate pre-trained spam classifiers on uploaded test data.

Run locally:
    streamlit run app.py

Models are loaded from model/*.joblib (produced by train_models.py); nothing is retrained here.
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

MODEL_DIR = "model"
TEST_FILE = "test_data.csv"
TARGET = "spam"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="wide")


@st.cache_resource
def load_models():
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in MODEL_FILES.items()
    }
    features = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return models, features


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    # AUC is undefined if the uploaded data contains only one class
    auc = roc_auc_score(y, y_prob) if y.nunique() > 1 else float("nan")
    return y_pred, {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


st.title("📧 Spam Filter Model Lab")
st.caption("Malyala Krishna Chaithanya · 2026AJ05709 · BITS Pilani M.Tech AIML · ML Assignment 2")

models, features = load_models()

# ---- Sidebar: data + model selection ----
st.sidebar.header("1. Test data")
uploaded = st.sidebar.file_uploader(
    "Upload CSV (57 feature columns + 'spam' target)", type="csv"
)
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = f"Uploaded file: {uploaded.name}"
else:
    df = pd.read_csv(TEST_FILE)
    source = f"Bundled test set: {TEST_FILE}"

missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(
        f"The CSV is missing {len(missing)} required column(s), e.g. {missing[:5]}. "
        f"Download the bundled test_data.csv from the repo for the expected format."
    )
    st.stop()

X, y = df[features], df[TARGET]

st.sidebar.header("2. Model")
model_name = st.sidebar.selectbox("Choose a model", list(models))

st.sidebar.download_button(
    "Download sample test CSV",
    data=open(TEST_FILE, "rb").read(),
    file_name="test_data.csv",
    mime="text/csv",
)

st.info(f"{source} — {len(df)} rows, {int(y.sum())} spam / {int((y == 0).sum())} not spam")

with st.expander("Dataset overview"):
    st.write("Class balance in this data:")
    st.bar_chart(y.map({0: "Not spam", 1: "Spam"}).value_counts())
    st.write("Feature summary (first 10 columns):")
    st.dataframe(X.iloc[:, :10].describe().T.round(3), width="stretch")

params_file = os.path.join(MODEL_DIR, "best_params.csv")
if os.path.exists(params_file):
    best_params = pd.read_csv(params_file).set_index("Model")["Best params"]
    st.sidebar.caption(f"Tuned hyperparameters: {best_params.get(model_name, 'n/a')}")

# ---- Selected model ----
y_pred, metrics = compute_metrics(models[model_name], X, y)

st.subheader(f"Evaluation metrics — {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Not spam", "Spam"], yticklabels=["Not spam", "Spam"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y, y_pred, labels=[0, 1], target_names=["Not spam", "Spam"],
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

# ---- All-model comparison ----
st.subheader("Comparison of all 5 models on this data")
comparison = pd.DataFrame(
    [{"Model": name, **compute_metrics(m, X, y)[1]} for name, m in models.items()]
).set_index("Model").round(4)
st.dataframe(
    comparison.style.highlight_max(axis=0, color="#c6efce"), width="stretch"
)
st.caption("Green = best score in each column.")

st.subheader("ROC curves — all models")
if y.nunique() > 1:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, m in models.items():
        fpr, tpr, _ = roc_curve(y, m.predict_proba(X)[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC={comparison.loc[name, 'AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    st.pyplot(fig)
else:
    st.warning("ROC curve needs both classes in the data.")
