#!/usr/bin/env python3
"""Predict mood for a Spotify track ID or from the bundled dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

import spotipy

from src.mood_model import load_mood_data, load_mood_model, predict_mood_from_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict song mood.")
    parser.add_argument("--model", default="artifacts/mood_model.joblib")
    parser.add_argument("--track-id", help="Spotify track ID (requires API credentials)")
    parser.add_argument("--from-csv", help="Song name to look up in data_moods.csv")
    parser.add_argument("--data", default="data/data_moods.csv")
    args = parser.parse_args()

    pipeline, encoder = load_mood_model(ROOT / args.model)

    if args.track_id:
        env_path = ROOT / ".env"
        if not env_path.exists():
            raise SystemExit(
                f"No .env file at {env_path}\n"
                "Copy config.example.env to .env and add your Spotify Client ID and Secret.\n"
                "Or run without the API: python scripts/predict_mood.py --from-csv \"1999\""
            )
        load_dotenv(env_path)
        from src.spotify_client import features_row_for_prediction

        try:
            row, name, artist = features_row_for_prediction(args.track_id)
        except EnvironmentError as exc:
            raise SystemExit(str(exc)) from exc
        except spotipy.SpotifyException as exc:
            if exc.http_status == 403:
                raise SystemExit(
                    "Spotify blocked audio-features (403). See README — use --from-csv instead."
                ) from exc
            raise
        mood = predict_mood_from_features(row, pipeline, encoder)
        print(f"{name} by {artist} is a {mood.upper()} song")
        return

    if args.from_csv:
        df = load_mood_data(ROOT / args.data)
        match = df[df["name"].str.lower() == args.from_csv.lower()]
        if match.empty:
            match = df[df["name"].str.contains(args.from_csv, case=False, na=False)]
        if match.empty:
            raise SystemExit(f"No song matching '{args.from_csv}' in {args.data}")
        row = match.iloc[0]
        mood = predict_mood_from_features(row, pipeline, encoder)
        print(f"{row['name']} by {row['artist']} is a {mood.upper()} song (actual: {row['mood']})")
        return

    parser.error("Provide --track-id or --from-csv")


if __name__ == "__main__":
    main()
