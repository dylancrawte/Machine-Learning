"""Multi-class mood classifier from Spotify audio features."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from .features import MOOD_FEATURES, MOOD_LABELS


def load_mood_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_mood_pipeline(
    *,
    hidden_layer_sizes: tuple[int, ...] = (8,),
    max_iter: int = 300,
    random_state: int = 15,
) -> Pipeline:
    """
    Scikit-learn replacement for the original Keras model:
    - MinMaxScaler on 10 audio features
    - MLP: 8 hidden units + 4-class softmax (via logistic multi-class)
    """
    return Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    max_iter=max_iter,
                    random_state=random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )


def train_mood_classifier(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 15,
    cv_folds: int = 10,
) -> Tuple[Pipeline, LabelEncoder, dict]:
    """Train mood classifier and return metrics."""
    X = df[MOOD_FEATURES].values
    y_raw = df["mood"].astype(str)

    encoder = LabelEncoder()
    encoder.fit(MOOD_LABELS)
    y = encoder.transform(y_raw)

    pipeline = build_mood_pipeline(random_state=random_state)

    cv_scores = cross_val_score(
        pipeline, X, y, cv=cv_folds, scoring="accuracy", n_jobs=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(
            y_test, y_pred, target_names=encoder.classes_
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classes": list(encoder.classes_),
    }
    return pipeline, encoder, metrics


def save_mood_model(
    pipeline: Pipeline,
    encoder: LabelEncoder,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "encoder": encoder}, path)


def load_mood_model(path: str | Path) -> Tuple[Pipeline, LabelEncoder]:
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["encoder"]


def predict_mood_from_features(
    features: dict | pd.Series | np.ndarray,
    pipeline: Pipeline,
    encoder: LabelEncoder,
) -> str:
    """Predict mood from one row of audio features."""
    if isinstance(features, dict):
        row = np.array([[features[c] for c in MOOD_FEATURES]], dtype=float)
    elif isinstance(features, pd.Series):
        row = features[MOOD_FEATURES].values.reshape(1, -1)
    else:
        row = np.asarray(features, dtype=float).reshape(1, -1)

    pred = int(pipeline.predict(row)[0])
    return str(encoder.inverse_transform([pred])[0])
