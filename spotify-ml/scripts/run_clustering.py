#!/usr/bin/env python3
"""Run K-means clustering on Spotify audio features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clustering import cluster_summary, load_clustering_data, run_clustering, save_cluster_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster songs with K-means (k=2).")
    parser.add_argument("--df1", default="data/df1.csv")
    parser.add_argument("--df2", default="data/df2.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--clusters", type=int, default=2)
    args = parser.parse_args()

    df = load_clustering_data(ROOT / args.df1, ROOT / args.df2)
    clustered, _model, _scaler = run_clustering(df, n_clusters=args.clusters)
    paths = save_cluster_outputs(clustered, ROOT / args.output_dir)

    print(f"Songs clustered: {len(clustered)}")
    print("\nCluster means:")
    print(cluster_summary(clustered).to_string())
    print("\nWrote:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
