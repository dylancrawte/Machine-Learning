#!/usr/bin/env python3
"""Train the relevance classifier from labeled data or two PubMed queries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.download import download_pubmed_xml
from src.model import NOT_RELEVANT_LABEL, RELEVANT_LABEL, save_classifier, train_classifier
from src.parse_papers import parse_xml_directory, preprocess_dataframe


def _load_labels(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    if "pmid" not in df.columns or "label" not in df.columns:
        raise ValueError("labels CSV must have columns: pmid, label")
    return df.set_index("pmid")["label"]


def _has_xml_files(directory: Path) -> bool:
    return directory.exists() and bool(list(directory.glob("*.xml")) or list(directory.glob("*.xml.gz")))


def _weak_labels_from_queries(
    positive_xml: Path,
    negative_xml: Path,
    positive_queries: list[str],
    negative_queries: list[str],
    api_key: str | None,
    max_per_query: int | None,
    skip_download: bool = False,
):
    if not (skip_download and _has_xml_files(positive_xml)):
        download_pubmed_xml(positive_xml, positive_queries, api_key, max_records=max_per_query)
    if not (skip_download and _has_xml_files(negative_xml)):
        download_pubmed_xml(negative_xml, negative_queries, api_key, max_records=max_per_query)

    pos = preprocess_dataframe(parse_xml_directory(positive_xml))
    neg = preprocess_dataframe(parse_xml_directory(negative_xml))

    pos["label"] = RELEVANT_LABEL
    neg["label"] = NOT_RELEVANT_LABEL

    import pandas as pd

    combined = pd.concat([pos, neg], ignore_index=True)
    combined = combined.drop_duplicates(subset=["pmid"], keep="first")
    return combined, combined["label"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PubMed relevance classifier.")
    parser.add_argument("--output-model", default="artifacts/classifier.joblib")
    parser.add_argument("--labels-csv", help="CSV with pmid,label (Relevant / Not Relevant)")
    parser.add_argument("--xml-dir", help="Directory of XML if labels-csv only has pmids")
    parser.add_argument("--positive-query", action="append", dest="positive_queries")
    parser.add_argument("--negative-query", action="append", dest="negative_queries")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-per-query", type=int, default=500)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing XML in data/train_*_xml (useful after a failed parse/train step)",
    )
    args = parser.parse_args()

    if args.labels_csv:
        import pandas as pd

        labels = _load_labels(Path(args.labels_csv))
        if args.xml_dir:
            df = preprocess_dataframe(parse_xml_directory(args.xml_dir))
            df = df[df["pmid"].isin(labels.index)]
            y = df["pmid"].map(labels)
        else:
            full = pd.read_csv(args.labels_csv)
            if not {"pmid", "label"}.issubset(full.columns):
                raise ValueError("labels-csv must include pmid and label columns")
            df = full
            y = full["label"]
    elif args.positive_queries and args.negative_queries:
        pos_xml = ROOT / "data" / "train_positive_xml"
        neg_xml = ROOT / "data" / "train_negative_xml"
        df, y = _weak_labels_from_queries(
            pos_xml,
            neg_xml,
            args.positive_queries,
            args.negative_queries,
            args.api_key,
            args.max_per_query,
            skip_download=args.skip_download,
        )
    else:
        parser.error(
            "Provide --labels-csv (+ optional --xml-dir) OR "
            "--positive-query and --negative-query"
        )

    pipeline, metrics = train_classifier(df, y)
    save_classifier(pipeline, ROOT / args.output_model)

    print(metrics["report"])
    print(f"Saved model to {ROOT / args.output_model}")


if __name__ == "__main__":
    main()
