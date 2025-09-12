"""Feature column definitions shared by clustering and mood models."""

from __future__ import annotations

# Used in clustering2.ipynb — 4 audio features, 3D plots
CLUSTER_FEATURES = ["danceability", "energy", "valence", "loudness"]

# Used in Keras-Classification.ipynb — df.columns[6:-1]
MOOD_FEATURES = [
    "danceability",
    "acousticness",
    "energy",
    "instrumentalness",
    "liveness",
    "valence",
    "loudness",
    "speechiness",
    "tempo",
    "key",
]

MOOD_LABELS = ["Calm", "Energetic", "Happy", "Sad"]

METADATA_COLUMNS = [
    "name",
    "album",
    "artist",
    "id",
    "release_date",
    "popularity",
]
