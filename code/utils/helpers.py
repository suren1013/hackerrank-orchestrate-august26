"""Small, reusable helper functions used across the project."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# ID / text helpers
# ---------------------------------------------------------------------------


def is_empty(value: Any) -> bool:
    """Return True if value is None, NaN, NaT, or an empty/whitespace string."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def safe_str(value: Any) -> str:
    """Convert a value to a clean string, returning '' for missing values."""
    if is_empty(value):
        return ""
    return str(value).strip()


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for lightweight comparisons."""
    return re.sub(r"\s+", " ", safe_str(text)).strip().lower()


def join_non_empty(values: list[Any], sep: str = ";") -> str:
    """Join non-empty string values with a separator."""
    cleaned = [safe_str(v) for v in values if not is_empty(v)]
    return sep.join(cleaned)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def ensure_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """Return the subset of `required` columns that exist in the DataFrame."""
    return [col for col in required if col in df.columns]


def rename_columns_with_prefix(
    df: pd.DataFrame, prefix: str, exclude: list[str] | None = None
) -> pd.DataFrame:
    """Rename all columns of a DataFrame with a prefix, except excluded ones.

    Used when merging context tables so that overlapping column names do not
    collide (e.g. group_id from groups.csv vs group_id in messages.csv).
    """
    exclude = exclude or []
    rename_map = {
        col: f"{prefix}{col}" for col in df.columns if col not in exclude
    }
    return df.rename(columns=rename_map)


def first_non_null(series: pd.Series) -> Any:
    """Return the first non-null value in a series, or None."""
    for value in series:
        if not is_empty(value):
            return value
    return None