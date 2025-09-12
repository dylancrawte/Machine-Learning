"""Train, save, and load the PubMed relevance classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .features import combine_features

RELEVANT_LABEL = "Relevant"
NOT_RELEVANT_LABEL = "Not Relevant"


def build_classifier_pipeline() -> Pipeline:
    """Sklearn pipeline over all text fields (same inputs as Pubmed-Pipeline)."""
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "text",
                FunctionTransformer(combine_features, validate=False),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=1,
                ),
            ),
            ("clf", clf),
        ]
    )


def train_classifier(
    df: pd.DataFrame,
    labels: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, dict]:
    """Train on labeled papers. Labels must be 'Relevant' or 'Not Relevant'."""
    y = labels.astype(str)

    if y.nunique() < 2:
        raise ValueError("Training labels must include at least two classes.")

    split_kwargs = dict(test_size=test_size, random_state=random_state)
    if len(y) >= 10 and y.value_counts().min() >= 2:
        split_kwargs["stratify"] = y

    X_train, X_test, y_train, y_test = train_test_split(df, y, **split_kwargs)

    pipeline = build_classifier_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "report": classification_report(y_test, y_pred, zero_division=0),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    return pipeline, metrics


def save_classifier(pipeline: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_classifier(path: str | Path) -> Pipeline:
    return joblib.load(path)


def predict_relevance(pipeline: Pipeline, df: pd.DataFrame) -> pd.Series:
    return pd.Series(pipeline.predict(df), index=df.index)


def filter_relevant(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows classified as Relevant (same behavior as Pubmed-Pipeline)."""
    predictions = predict_relevance(pipeline, df)
    out = df.copy()
    out["prediction"] = predictions
    return out[out["prediction"] == RELEVANT_LABEL].drop(columns=["prediction"])
