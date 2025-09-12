"""Setup and update pipelines (pandas-based recreation of Pubmed-Pipeline)."""

from __future__ import annotations

import datetime
import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .download import download_pubmed_xml
from .model import filter_relevant, load_classifier
from .parse_papers import parse_xml_directory, preprocess_dataframe


class PubmedPipeline:
    def __init__(
        self,
        xml_dir: str | Path,
        classifier_path: str | Path,
        last_run_path: str | Path,
    ):
        self.xml_dir = Path(xml_dir)
        self.classifier_path = Path(classifier_path)
        self.last_run_path = Path(last_run_path)
        self.classifier = load_classifier(self.classifier_path)

    def save_last_run_date(self) -> None:
        self.last_run_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.last_run_path, "wb") as f:
            pickle.dump(datetime.date.today(), f)

    @staticmethod
    def load_last_run_date(path: str | Path) -> datetime.date:
        with open(path, "rb") as f:
            return pickle.load(f)

    def parse_and_classify(self) -> pd.DataFrame:
        raw = parse_xml_directory(self.xml_dir)
        df = preprocess_dataframe(raw)
        return filter_relevant(self.classifier, df)


class PubmedPipelineSetup(PubmedPipeline):
    def __init__(
        self,
        xml_dir: str | Path,
        classifier_path: str | Path,
        dataframe_output_path: str | Path,
        last_run_path: str | Path,
    ):
        super().__init__(xml_dir, classifier_path, last_run_path)
        self.dataframe_output_path = Path(dataframe_output_path)

    def download_xml_from_pubmed(
        self,
        search_queries: List[str],
        api_key: Optional[str] = None,
        *,
        max_records: Optional[int] = None,
    ) -> Path:
        download_pubmed_xml(
            self.xml_dir,
            search_queries,
            api_key,
            max_records=max_records,
        )
        self.save_last_run_date()
        return self.xml_dir

    def run_pipeline(self) -> pd.DataFrame:
        df = self.parse_and_classify()
        self.dataframe_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.dataframe_output_path, index=False)
        return df


class PubmedPipelineUpdate(PubmedPipeline):
    def __init__(
        self,
        xml_dir: str | Path,
        classifier_path: str | Path,
        main_dataframe_path: str | Path,
        last_run_path: str | Path,
        new_and_updated_path: str | Path,
    ):
        super().__init__(xml_dir, classifier_path, last_run_path)
        self.main_dataframe_path = Path(main_dataframe_path)
        self.new_and_updated_path = Path(new_and_updated_path)
        self.main_dataframe = pd.read_parquet(self.main_dataframe_path)

    def download_xml_from_pubmed(
        self,
        search_queries: List[str],
        api_key: Optional[str] = None,
        *,
        max_records: Optional[int] = None,
    ) -> Path:
        last_run = self.load_last_run_date(self.last_run_path)
        today = datetime.date.today()
        reldate = (today - last_run).days

        if reldate < 1:
            raise ValueError(
                "Days since last pipeline run is less than 1. "
                "Run at most once per day (PubMed updates daily)."
            )

        reldate += 1  # timezone buffer, same as original library

        download_pubmed_xml(
            self.xml_dir,
            search_queries,
            api_key,
            reldate=reldate,
            max_records=max_records,
        )
        self.save_last_run_date()
        return self.xml_dir

    def run_pipeline(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        new_df = self.parse_and_classify()

        if not new_df.empty and "pmid" in self.main_dataframe.columns:
            common = set(self.main_dataframe["pmid"]) & set(new_df["pmid"])
            self.main_dataframe = self.main_dataframe[
                ~self.main_dataframe["pmid"].isin(common)
            ]

        updated_main = pd.concat([self.main_dataframe, new_df], ignore_index=True)

        self.main_dataframe_path.parent.mkdir(parents=True, exist_ok=True)
        updated_main.to_parquet(self.main_dataframe_path, index=False)
        new_df.to_parquet(self.new_and_updated_path, index=False)

        return updated_main, new_df
