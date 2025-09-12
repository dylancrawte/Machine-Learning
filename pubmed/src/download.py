"""Download PubMed records as XML via NCBI E-utilities."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import quote_plus

import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BATCH_SIZE = 500  # efetch max per request without history is 500; use history for larger sets
REQUEST_DELAY = 0.34  # ~3 req/s without API key


def _join_queries(queries: Iterable[str]) -> str:
    parts = [q.strip() for q in queries if q and q.strip()]
    if not parts:
        raise ValueError("At least one search query is required.")
    return "+OR+".join(quote_plus(p) for p in parts)


def _get(session: requests.Session, url: str, params: Optional[dict] = None) -> requests.Response:
    time.sleep(REQUEST_DELAY)
    response = session.get(url, params=params, timeout=120)
    response.raise_for_status()
    return response


def _esearch(
    session: requests.Session,
    term: str,
    *,
    api_key: Optional[str] = None,
    reldate: Optional[int] = None,
    datetype: Optional[str] = None,
    webenv: Optional[str] = None,
    usehistory: bool = True,
) -> dict:
    params = {"db": "pubmed", "term": term, "retmode": "json"}
    if usehistory:
        params["usehistory"] = "y"
    if api_key:
        params["api_key"] = api_key
    if reldate is not None:
        params["reldate"] = str(reldate)
    if datetype:
        params["datetype"] = datetype
    if webenv:
        params["webenv"] = webenv

    data = _get(session, f"{EUTILS_BASE}esearch.fcgi", params).json()
    result = data.get("esearchresult", {})
    return {
        "count": int(result.get("count", 0)),
        "webenv": result.get("webenv"),
        "query_key": result.get("querykey"),
    }


def _efetch_batch(
    session: requests.Session,
    *,
    query_key: str,
    webenv: str,
    retstart: int,
    retmax: int,
    api_key: Optional[str],
) -> str:
    params = {
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retmode": "xml",
        "retstart": retstart,
        "retmax": retmax,
    }
    if api_key:
        params["api_key"] = api_key
    return _get(session, f"{EUTILS_BASE}efetch.fcgi", params).text


def download_pubmed_xml(
    output_dir: str | Path,
    search_queries: List[str],
    api_key: Optional[str] = None,
    *,
    reldate: Optional[int] = None,
    max_records: Optional[int] = None,
) -> Path:
    """
    Download PubMed XML for search_queries into output_dir.

    If reldate is set, runs modification + entrez date searches (update mode).
  """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    query = _join_queries(search_queries)

    if reldate is not None:
        webenv = _esearch(
            session,
            query,
            api_key=api_key,
            reldate=reldate,
            datetype="mdat",
        )["webenv"]
        _esearch(
            session,
            query,
            api_key=api_key,
            reldate=reldate,
            datetype="edat",
            webenv=webenv,
        )
        search = _esearch(session, "%231+OR+%232", api_key=api_key, webenv=webenv)
    else:
        search = _esearch(session, query, api_key=api_key)

    total = search["count"]
    if total == 0:
        return output_path

    if max_records is not None:
        total = min(total, max_records)

    query_key = search["query_key"]
    webenv = search["webenv"]

    file_index = 0
    for retstart in range(0, total, BATCH_SIZE):
        retmax = min(BATCH_SIZE, total - retstart)
        xml_text = _efetch_batch(
            session,
            query_key=query_key,
            webenv=webenv,
            retstart=retstart,
            retmax=retmax,
            api_key=api_key,
        )
        out_file = output_path / f"results{file_index}.xml"
        out_file.write_text(xml_text, encoding="utf-8")
        file_index += 1

    return output_path


def count_pubmed_results(search_queries: List[str], api_key: Optional[str] = None) -> int:
    """Return how many PubMed records match the query (no download)."""
    session = requests.Session()
    query = _join_queries(search_queries)
    return _esearch(session, query, api_key=api_key)["count"]
