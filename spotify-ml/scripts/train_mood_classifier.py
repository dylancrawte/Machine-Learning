#!/usr/bin/env python3
"""Train the Spotify mood classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mood_model import load_mood_data, save_mood_model, train_mood_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mood classifier (4 classes).")
    parser.add_argument("--data", default="data/data_moods.csv")
    parser.add_argument("--output", default="artifacts/mood_model.joblib")
    args = parser.parse_args()

    df = load_mood_data(ROOT / args.data)
    pipeline, encoder, metrics = train_mood_classifier(df)
    save_mood_model(pipeline, encoder, ROOT / args.output)

    print(f"CV accuracy: {metrics['cv_mean_accuracy']:.2%} (+/- {metrics['cv_std_accuracy']:.2%})")
    print(f"Test accuracy: {metrics['test_accuracy']:.2%}")
    print(metrics["classification_report"])
    print(f"Saved model to {ROOT / args.output}")


if __name__ == "__main__":
    main()
