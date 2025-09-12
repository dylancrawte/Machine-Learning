#!/usr/bin/env python3
"""Initial PubMed download + classify + save parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import PubmedPipelineSetup


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PubMed setup pipeline.")
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--xml-dir", default="data/xml")
    parser.add_argument("--classifier", default="artifacts/classifier.joblib")
    parser.add_argument("--output", default="data/relevant_papers.parquet")
    parser.add_argument("--last-run", default="artifacts/last_run.pkl")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    job = PubmedPipelineSetup(
        xml_dir=ROOT / args.xml_dir,
        classifier_path=ROOT / args.classifier,
        dataframe_output_path=ROOT / args.output,
        last_run_path=ROOT / args.last_run,
    )

    if not args.skip_download:
        job.download_xml_from_pubmed(
            args.queries,
            args.api_key,
            max_records=args.max_records,
        )

    df = job.run_pipeline()
    print(f"Saved {len(df)} relevant papers to {ROOT / args.output}")


if __name__ == "__main__":
    main()
