"""Parse PubMed Medline XML files into a pandas DataFrame."""

from __future__ import annotations

import gzip
from contextlib import contextmanager
from glob import glob
from pathlib import Path
from typing import Iterator, List

import pandas as pd
from lxml import etree

try:
    from pubmed_parser.medline_parser import parse_article_info, parse_grant_id
except ImportError as exc:
    raise ImportError(
        "pubmed-parser is required. Install with: pip install -r requirements.txt"
    ) from exc

TEXT_COLUMNS = [
    "abstract",
    "title",
    "medline_ta",
    "keywords",
    "publication_types",
    "chemical_list",
    "country",
    "authors",
    "mesh_terms",
]

METADATA_COLUMNS = [
    "pmid",
    "pmc",
    "pubdate",
    "doi",
    "other_id",
    "nlm_unique_id",
    "file_name",
]


@contextmanager
def _open_medline_xml(path: str | Path):
    """Open PubMed XML from E-utilities (plain .xml) or FTP dumps (.xml.gz)."""
    path_str = str(path)
    handle = gzip.open(path_str, "rb") if path_str.endswith(".gz") else open(path_str, "rb")
    try:
        yield handle
    finally:
        handle.close()


def iter_medline_records(path: str | Path) -> Iterator[dict]:
    """
    Parse one Medline XML file.

    pubmed_parser.parse_medline_xml only supports gzip; E-utilities returns plain XML.
    """
    with _open_medline_xml(path) as handle:
        for _event, element in etree.iterparse(handle, events=("end",)):
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
            if tag == "PubmedArticle":
                record = parse_article_info(
                    element,
                    year_info_only=True,
                    nlm_category=False,
                    author_list=False,
                    reference_list=False,
                    parse_subs=False,
                )
                record["grant_ids"] = parse_grant_id(element)
                element.clear()
                while element.getprevious() is not None:
                    del element.getparent()[0]
                yield record


def parse_xml_directory(xml_dir: str | Path) -> pd.DataFrame:
    """Parse all *.xml and *.xml.gz files in xml_dir into one DataFrame."""
    xml_path = Path(xml_dir)
    files = sorted(glob(str(xml_path / "*.xml")) + glob(str(xml_path / "*.xml.gz")))
    if not files:
        raise FileNotFoundError(f"No XML files found in {xml_path}")

    rows: List[dict] = []
    for file_path in files:
        for record in iter_medline_records(file_path):
            record["file_name"] = Path(file_path).name
            rows.append(record)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "pmid" in df.columns:
        df["pmid"] = pd.to_numeric(df["pmid"], errors="coerce").astype("Int64")
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Select and normalize columns to match the original Pubmed-Pipeline schema."""
    rename = {"authors": "author", "affiliations": "affiliation"}
    df = df.rename(columns=rename)

    keep = [
        "pmid",
        "pmc",
        "title",
        "medline_ta",
        "pubdate",
        "author",
        "affiliation",
        "publication_types",
        "mesh_terms",
        "keywords",
        "chemical_list",
        "abstract",
        "country",
        "other_id",
        "doi",
        "nlm_unique_id",
        "file_name",
    ]
    existing = [c for c in keep if c in df.columns]
    return df[existing].copy()
