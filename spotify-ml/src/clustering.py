"""K-means clustering of Spotify audio features (playlist generation)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from .features import CLUSTER_FEATURES, METADATA_COLUMNS


def load_clustering_data(
    df1_path: str | Path,
    df2_path: str | Path,
) -> pd.DataFrame:
    """Load and merge artist/album CSVs like clustering2.ipynb."""
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    df = pd.concat([df1, df2], ignore_index=True)
    keep = METADATA_COLUMNS + CLUSTER_FEATURES
    existing = [c for c in keep if c in df.columns]
    return df[existing].copy()


def run_clustering(
    df: pd.DataFrame,
    *,
    n_clusters: int = 2,
    random_state: int = 15,
) -> Tuple[pd.DataFrame, KMeans, MinMaxScaler]:
    """
    Cluster songs by audio features.

    Mirrors the original notebook:
    - MinMaxScaler on danceability, energy, valence, loudness
    - KMeans(k=2, init='k-means++')
    """
    missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing clustering features: {missing}")

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[CLUSTER_FEATURES])

    kmeans = KMeans(
        init="k-means++",
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    kmeans.fit(X)

    out = df.copy()
    out["kmeans"] = kmeans.labels_
    return out, kmeans, scaler


def save_cluster_outputs(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write df.csv, cluster0.csv, cluster1.csv like the original repo."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "all": output_dir / "df.csv",
        "cluster_0": output_dir / "cluster0.csv",
        "cluster_1": output_dir / "cluster1.csv",
    }
    df.to_csv(paths["all"], index=False)
    df[df["kmeans"] == 0].to_csv(paths["cluster_0"], index=False)
    df[df["kmeans"] == 1].to_csv(paths["cluster_1"], index=False)
    return paths


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean audio features per cluster (groupby from original notebook)."""
    return df.groupby("kmeans")[CLUSTER_FEATURES].mean()
