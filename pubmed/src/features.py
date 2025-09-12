"""Feature construction for the relevance classifier."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "abstract",
    "title",
    "medline_ta",
    "keywords",
    "publication_types",
    "chemical_list",
    "country",
    "author",
    "mesh_terms",
]


def _to_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value if v is not None)
    return str(value)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one string column per model feature."""
    out = pd.DataFrame(index=df.index)
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            out[col] = df[col].map(_to_text)
        else:
            out[col] = ""
    return out


def combine_features(df: pd.DataFrame) -> pd.Series:
    """Single text field for quick inspection or baselines."""
    parts = [build_feature_frame(df)[col] for col in FEATURE_COLUMNS]
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + " " + part
    return combined.str.strip()
