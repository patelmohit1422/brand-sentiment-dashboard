from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    TOP_KEYWORDS_PATH,
    RANDOM_STATE,
)
from .data_generator import save_dataset
from .text_processing import preprocess_series, preprocess_text


def train_and_save_model(force_regenerate_data: bool = False):
    if force_regenerate_data or not RAW_DATA_PATH.exists():
        df = save_dataset()
    else:
        df = pd.read_csv(RAW_DATA_PATH)

    df["clean_text"] = preprocess_series(df["text"])

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=6000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_samples": int(len(df)),
        "class_distribution": df["label"].value_counts().to_dict(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model, "labels": list(model.classes_)}, MODEL_PATH)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    feature_names = vectorizer.get_feature_names_out()
    class_keywords = {}
    for class_idx, class_name in enumerate(model.classes_):
        coef = model.coef_[class_idx]
        top_positive_idx = coef.argsort()[-20:][::-1]
        top_negative_idx = coef.argsort()[:20]
        class_keywords[class_name] = {
            "top_positive_terms": [
                {"term": feature_names[i], "weight": round(float(coef[i]), 4)} for i in top_positive_idx
            ],
            "top_negative_terms": [
                {"term": feature_names[i], "weight": round(float(coef[i]), 4)} for i in top_negative_idx
            ],
        }

    TOP_KEYWORDS_PATH.write_text(json.dumps(class_keywords, indent=2), encoding="utf-8")
    return df, metrics, class_keywords


def load_artifacts():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not PROCESSED_DATA_PATH.exists():
        df, metrics, top_keywords = train_and_save_model()
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        top_keywords = json.loads(TOP_KEYWORDS_PATH.read_text(encoding="utf-8")) if TOP_KEYWORDS_PATH.exists() else {}
    artifacts = joblib.load(MODEL_PATH)
    return df, metrics, top_keywords, artifacts


def predict_text(text: str):
    if not MODEL_PATH.exists():
        train_and_save_model()
    artifacts = joblib.load(MODEL_PATH)
    vectorizer = artifacts["vectorizer"]
    model = artifacts["model"]
    labels = artifacts["labels"]

    clean = preprocess_text(text)
    X = vectorizer.transform([clean])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    prob_map = {label: float(prob) for label, prob in zip(labels, probs)}
    return pred, prob_map, clean
