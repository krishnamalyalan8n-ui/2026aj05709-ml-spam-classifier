# Spam Email Classification — ML Assignment 2

**Name:** Malyala Krishna Chaithanya  **BITS ID:** 2026AJ05709  **Program:** M.Tech (AIML), BITS Pilani WILP

## Links

- **GitHub Repository:** https://github.com/krishnamalyalan8n-ui/2026aj05709-ml-spam-classifier
- **Live Streamlit App:** https://2026aj05709-spam-classifier.streamlit.app/

## a. Problem Statement

Unwanted bulk email (spam) wastes time and is a common way to deliver phishing and malware. The goal is to build a **binary classifier** that predicts whether an email is **spam (1)** or **not spam (0)** from word- and character-frequency statistics. Five classification models are trained and tuned on the same dataset and compared on six metrics. The trained models are served through an interactive Streamlit web app.

## b. Dataset Description

| Property | Value |
|---|---|
| Name | Spambase (UCI Machine Learning Repository) |
| Source | https://archive.ics.uci.edu/dataset/94/spambase |
| Instances | 4,601 emails (1,813 spam = 39.4%, 2,788 not spam = 60.6%) |
| Features | 57 continuous numeric features (no missing values) |
| Target | `spam` — 1 = spam, 0 = not spam |

Feature groups:

- **48 `word_freq_<WORD>`** — percentage of words in the email that match WORD (e.g. `free`, `money`, `remove`, `george`, `hp`).
- **6 `char_freq_<CHAR>`** — percentage of characters that match `; ( [ ! $ #`.
- **3 `capital_run_length_*`** — average, longest, and total length of uninterrupted runs of capital letters.

**Preprocessing:** 80/20 stratified train/test split (`random_state=5709`), giving 3,680 training rows and 921 test rows. Each model is a `Pipeline(log1p → StandardScaler → classifier)`. The log1p step reduces the heavy right skew of the frequency features, and scaling is fitted on the training data only. Hyperparameters are tuned per model with **5-fold GridSearchCV** (scoring = F1) on the training set. The held-out test set is saved as `test_data.csv`.

## c. GitHub Repository Link

**Repository:** https://github.com/krishnamalyalan8n-ui/2026aj05709-ml-spam-classifier

The repository contains the complete source code, `requirements.txt`, this README, the test data (`test_data.csv`) and the saved models for all five classifiers:

```
ML-SPAM-CLASSIFIER/
├── app.py                 # Streamlit app (loads saved models, no retraining)
├── train_models.py        # Downloads data, tunes + trains 5 models, saves models + metrics
├── print_table.py         # Helper to print the metrics table
├── requirements.txt
├── README.md
├── test_data.csv          # Held-out 20% test set (57 features + target)
├── .gitignore
├── .streamlit/
│   └── config.toml        # App theme
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── feature_names.joblib
    ├── metrics_summary.csv
    └── best_params.csv
```

## d. Models Used and Evaluation Metrics

Models (each tuned with 5-fold GridSearchCV): Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine (RBF kernel). All metrics are computed on the 921-row held-out test set. AUC uses `predict_proba[:, 1]`.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9468 | 0.9816 | 0.9266 | 0.9394 | 0.9330 | 0.8889 |
| Decision Tree | 0.9218 | 0.9332 | 0.9145 | 0.8843 | 0.8992 | 0.8357 |
| KNN | 0.9555 | 0.9822 | 0.9423 | 0.9449 | 0.9436 | 0.9068 |
| Naive Bayes | 0.8274 | 0.9606 | 0.7073 | **0.9587** | 0.8140 | 0.6863 |
| **Support Vector Machine** | **0.9577** | **0.9868** | **0.9475** | 0.9449 | **0.9462** | **0.9113** |

(Best value in each column in bold.)

Best hyperparameters found (see `model/best_params.csv`):

| Model | Best params |
|---|---|
| Logistic Regression | C = 1 |
| Decision Tree | max_depth = 10, min_samples_leaf = 5 |
| KNN | n_neighbors = 9, weights = distance |
| Naive Bayes | var_smoothing = 1e-3 |
| SVM | C = 10, gamma = scale |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.947, AUC 0.982. It is a simple linear model that is fast and easy to interpret, and it came 3rd, behind SVM and KNN. It does well because, after the log1p transform and scaling, spam and non-spam emails are already close to linearly separable. Its straight-line boundary cannot capture the remaining non-linear patterns. |
| Decision Tree | Lowest AUC of all five models (0.933), even after tuning (max_depth = 10, min_samples_leaf = 5). A single tree splits the data into a limited number of leaves, so its probability scores are coarse and step-like. It also tends to memorise the training data (overfit) instead of learning smooth patterns. |
| KNN | 2nd best (Accuracy 0.956, F1 0.944). Tuning helped a lot (k = 9, distance weighting): looking at more neighbours and trusting the closer ones more reduces noise from single odd emails. It only works well because the features are log-transformed and scaled, otherwise large columns like capital_run_length_total would dominate the distance. It is slow to predict because it compares each email with every training email. |
| Naive Bayes | Highest recall (0.959), so it catches almost all spam, but the lowest precision (0.707), so it wrongly flags many good emails. This is because it assumes all features are independent, which is not true here (words like "free", "money" and "$" usually appear together), so it over-counts the spam evidence. It is still useful when missing spam is the costliest mistake. |
| Support Vector Machine | Best on 5 of the 6 metrics (Accuracy 0.958, AUC 0.987, Precision 0.948, F1 0.946, MCC 0.911). The RBF kernel can draw curved decision boundaries, and C = 10 lets it fit the training data closely without overfitting the test data. It has the highest precision, so it loses the fewest good emails, but it is the slowest model to train. |
| **Overall winner** | **Support Vector Machine** is the winner, with **KNN** a close second. For a spam filter, precision matters most, because sending a real email (for example a bank or job email) to the spam folder is worse than letting one spam email through, and SVM has the highest precision. |
