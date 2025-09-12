"""Optional Spotify API helpers (from the original helpers.py)."""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import pandas as pd

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except ImportError as exc:
    raise ImportError("spotipy is required. pip install spotipy") from exc

from .features import CLUSTER_FEATURES, METADATA_COLUMNS, MOOD_FEATURES

AUDIO_COLUMNS = METADATA_COLUMNS + ["length"] + MOOD_FEATURES

_PLACEHOLDER_MARKERS = ("your_client", "your_spotify", "changeme", "xxx")


def _load_credentials() -> tuple[str, str]:
    client_id = (os.environ.get("SPOTIFY_CLIENT_ID") or "").strip().strip("'\"")
    client_secret = (os.environ.get("SPOTIFY_CLIENT_SECRET") or "").strip().strip("'\"")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "Missing Spotify credentials.\n"
            "  1. Create an app at https://developer.spotify.com/dashboard\n"
            "  2. Copy config.example.env to .env in the spotify-ml folder\n"
            "  3. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET (Client ID + Client Secret)\n"
            "Or use: python scripts/predict_mood.py --from-csv \"Song Name\" (no API needed)"
        )

    combined = (client_id + client_secret).lower()
    if any(marker in combined for marker in _PLACEHOLDER_MARKERS):
        raise EnvironmentError(
            "Spotify credentials in .env still look like placeholders.\n"
            "Replace them with the real Client ID and Client Secret from your Spotify app dashboard."
        )

    return client_id, client_secret


def _client() -> spotipy.Spotify:
    client_id, client_secret = _load_credentials()
    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    try:
        return spotipy.Spotify(client_credentials_manager=auth)
    except spotipy.SpotifyOauthError as exc:
        raise EnvironmentError(
            "Spotify rejected your API credentials (invalid_client).\n"
            "  - Confirm Client ID and Client Secret in .env match the Spotify Developer Dashboard\n"
            "  - If you clicked 'Reset client secret', update .env with the new secret\n"
            "  - No extra quotes or spaces around values in .env\n"
            "  - The app must be created under your Spotify account (not a deleted/old app)\n"
            f"Details: {exc}"
        ) from exc


_AUDIO_FEATURES_403_HELP = """
Spotify returned 403 Forbidden on /v1/audio-features.

Since Nov 2024, Spotify blocks Audio Features for most new apps and apps in
Development Mode. Track metadata may still work; audio features will not.

What you can do:
  1. Use bundled data (no API): python scripts/predict_mood.py --from-csv "1999"
  2. Request Extended Quota / legacy access in the Spotify Developer Dashboard
     https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api
  3. Use the pre-built CSVs in data/ (same as the original Spotify-ML repo)
"""


def _fetch_audio_features(sp: spotipy.Spotify, track_id: str) -> dict:
    try:
        feats_list = sp.audio_features([track_id])
    except spotipy.SpotifyException as exc:
        if exc.http_status == 403:
            raise EnvironmentError(_AUDIO_FEATURES_403_HELP.strip()) from exc
        raise

    if not feats_list or feats_list[0] is None:
        raise ValueError(f"No audio features returned for track id {track_id!r}")

    return feats_list[0]


def track_features(track_id: str, sp: Optional[spotipy.Spotify] = None) -> dict:
    """Fetch metadata + audio features for one Spotify track ID."""
    sp = sp or _client()
    meta = sp.track(track_id)
    feats = _fetch_audio_features(sp, track_id)

    return {
        "name": meta["name"],
        "album": meta["album"]["name"],
        "artist": meta["album"]["artists"][0]["name"],
        "id": meta["id"],
        "release_date": meta["album"]["release_date"],
        "popularity": meta["popularity"],
        "length": meta["duration_ms"],
        "danceability": feats["danceability"],
        "acousticness": feats["acousticness"],
        "energy": feats["energy"],
        "instrumentalness": feats["instrumentalness"],
        "liveness": feats["liveness"],
        "valence": feats["valence"],
        "loudness": feats["loudness"],
        "speechiness": feats["speechiness"],
        "tempo": feats["tempo"],
        "key": feats["key"],
        "time_signature": feats["time_signature"],
    }


def download_album_tracks(
    album_id: str,
    *,
    delay_seconds: float = 0.6,
    sp: Optional[spotipy.Spotify] = None,
) -> pd.DataFrame:
    """Download all tracks from one album ID into a DataFrame."""
    sp = sp or _client()
    rows: List[dict] = []

    album = sp.album(album_id)
    for track in sp.album_tracks(album_id)["items"]:
        time.sleep(delay_seconds)
        tid = track["id"]
        row = track_features(tid, sp=sp)
        rows.append(row)

    return pd.DataFrame(rows)


def features_row_for_prediction(track_id: str) -> Tuple[dict, str, str]:
    """Return features dict, song name, and artist for predict_mood."""
    row = track_features(track_id)
    name = row["name"]
    artist = row["artist"]
    return row, name, artist
