#!/usr/bin/env python3
"""Incremental PubMed update pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import PubmedPipelineUpdate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PubMed update pipeline.")
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--xml-dir", default="data/xml_update")
    parser.add_argument("--classifier", default="artifacts/classifier.joblib")
    parser.add_argument("--main-output", default="data/relevant_papers.parquet")
    parser.add_argument("--delta-output", default="data/new_and_updated.parquet")
    parser.add_argument("--last-run", default="artifacts/last_run.pkl")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    job = PubmedPipelineUpdate(
        xml_dir=ROOT / args.xml_dir,
        classifier_path=ROOT / args.classifier,
        main_dataframe_path=ROOT / args.main_output,
        last_run_path=ROOT / args.last_run,
        new_and_updated_path=ROOT / args.delta_output,
    )

    if not args.skip_download:
        job.download_xml_from_pubmed(
            args.queries,
            args.api_key,
            max_records=args.max_records,
        )

    main_df, new_df = job.run_pipeline()
    print(f"Main dataset: {len(main_df)} papers")
    print(f"New/updated batch: {len(new_df)} papers")


if __name__ == "__main__":
    main()
