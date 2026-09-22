"""The three rung-1 facts, computed for any dataset instead of hand-written.

The churn app had them welded to its columns: 11 blank total_charges, and
contract type against churn. Here each becomes a question asked of the data:

  what is missing      -- NULLs per column, and whether one other column
                          explains every one of them (tenure = 0 did)
  who goes with whom   -- the categorical columns whose event rate varies most
                          across their own values
  numeric or category  -- one shared rule, so the charts and the model's
                          encoder never disagree about a column
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

MIN_SHARE = 0.01          # a value held by fewer rows than this is noise
MAX_CATEGORIES = 15       # above this a column is an identifier, not a category


def to_snake(name: str) -> str:
    """CamelCase and customerID both become snake_case.

    Postgres lowercases unquoted identifiers, so a column imported as
    "MonthlyCharges" has to be quoted in every query afterwards. Renaming once,
    here, is cheaper than quoting forever.
    """
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.replace("__", "_").strip("_").lower()


def split_columns(df: pd.DataFrame, id_column: str, target_column: str,
                  categorical_int_max_unique: int) -> tuple[list[str], list[str]]:
    """Numeric columns first, categorical second, in the frame's own order.

    An integer column with only a handful of distinct values is a flag, not a
    measurement: senior_citizen holds 0 and 1 and belongs in the encoder, while
    tenure runs 0 to 72 and belongs on a scaler.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    for column in df.columns:
        if column in (id_column, target_column):
            continue
        series = df[column]
        if ptypes.is_float_dtype(series):
            numeric.append(column)
        elif ptypes.is_integer_dtype(series):
            if series.nunique(dropna=True) > categorical_int_max_unique:
                numeric.append(column)
            else:
                categorical.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def missing_report(df: pd.DataFrame, id_column: str,
                   target_column: str) -> list[dict]:
    """Every column that has NULLs, most first. Clean columns are left out."""
    rows = []
    for column in df.columns:
        nulls = int(df[column].isna().sum())
        if nulls:
            rows.append(
                {
                    "column": column,
                    "nulls": nulls,
                    "share": nulls / len(df) if len(df) else 0.0,
                }
            )
    return sorted(rows, key=lambda r: -r["nulls"])


def explains_missing(df: pd.DataFrame, column: str) -> dict | None:
    """Does one other column hold a single value on every null row?

    In the Telco set the answer was tenure = 0: the blanks were this month's
    joiners, not a data error. Returned only when the shared value is rarer in
    the table at large than it is among the null rows, so a constant column
    does not masquerade as an explanation.
    """
    nulls = df[df[column].isna()]
    if nulls.empty:
        return None
    candidates = []
    for other in df.columns:
        if other == column:
            continue
        values = nulls[other].dropna().unique()
        if len(values) != 1:
            continue
        value = values[0]
        overall = float((df[other] == value).mean())
        if overall < 1.0 and overall < len(nulls) / len(df) * 5 + 0.5:
            candidates.append((overall, other, value))
    if not candidates:
        return None
    # The tightest candidate wins, not the first one in column order. On the
    # Telco set dependents = Yes clears the threshold and covers 2,110 rows,
    # explaining nothing, while tenure = 0 covers exactly the 11 null rows.
    _, other, value = min(candidates, key=lambda candidate: candidate[0])
    return {"column": other, "value": value, "n": int(len(nulls))}


def discriminative_categoricals(df: pd.DataFrame, target_column: str,
                                positive_label: str, k: int = 3) -> list[dict]:
    """The k categorical columns whose event rate varies most across values."""
    event = (df[target_column].astype(str).str.strip() == positive_label).astype(int)
    ranked = []
    for column in df.columns:
        if column == target_column:
            continue
        series = df[column]
        if ptypes.is_float_dtype(series):
            continue
        counts = series.value_counts(dropna=True)
        keep = counts[counts >= max(1, int(MIN_SHARE * len(df)))]
        if not 2 <= len(keep) <= MAX_CATEGORIES:
            continue
        rates = (
            event[series.isin(keep.index)]
            .groupby(series[series.isin(keep.index)])
            .mean()
        )
        if rates.empty:
            continue
        ranked.append(
            {
                "column": column,
                "spread": float(rates.max() - rates.min()),
                "rates": {str(idx): float(value) for idx, value in rates.items()},
            }
        )
    return sorted(ranked, key=lambda r: -r["spread"])[:k]


# --------------------------------------------------------------------------
# What moves with the target -- the rung-1 story charts
# --------------------------------------------------------------------------
def _event(df: pd.DataFrame, target_column: str, positive_label: str) -> pd.Series:
    return (df[target_column].astype(str).str.strip() == positive_label).astype(int)


def _cramers_v(series: pd.Series, event: pd.Series) -> float:
    """Association between a categorical column and a yes/no target, 0 to 1."""
    table = pd.crosstab(series, event)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    return math.sqrt(chi2 / (total * (min(table.shape) - 1)))


def associations(df: pd.DataFrame, target_column: str, positive_label: str,
                 id_column: str, categorical_int_max_unique: int) -> list[dict]:
    """Every column ranked by how strongly it moves with the target.

    Point-biserial correlation for a measurement, Cramér's V for a category:
    both run 0 to 1, so one bar chart can hold the whole table. Columns that
    are identifiers (more distinct values than MAX_CATEGORIES) are skipped.
    """
    event = _event(df, target_column, positive_label)
    numeric, categorical = split_columns(
        df, id_column, target_column, categorical_int_max_unique
    )
    rows = []
    for column in numeric:
        series = df[column]
        if series.nunique(dropna=True) < 2:
            continue
        strength = abs(float(series.corr(event)))
        rows.append({"column": column, "kind": "numeric",
                     "strength": 0.0 if math.isnan(strength) else strength})
    for column in categorical:
        series = df[column].astype(str)
        if not 2 <= series.nunique(dropna=True) <= MAX_CATEGORIES:
            continue
        rows.append({"column": column, "kind": "categorical",
                     "strength": _cramers_v(series, event)})
    return sorted(rows, key=lambda r: -r["strength"])


def numeric_separation(df: pd.DataFrame, target_column: str, positive_label: str,
                       numeric: list[str]) -> list[dict]:
    """Measurements ranked by how far apart the two groups sit on them.

    `effect` is the gap between the medians in units of the column's spread,
    so tenure (0 to 72) and monthly charges (18 to 118) compare fairly.
    """
    event = _event(df, target_column, positive_label).astype(bool)
    rows = []
    for column in numeric:
        series = pd.to_numeric(df[column], errors="coerce")
        spread = float(series.std(ddof=0)) or 1.0
        positive = float(series[event].median())
        negative = float(series[~event].median())
        rows.append({
            "column": column,
            "median_positive": positive,
            "median_negative": negative,
            "effect": abs(positive - negative) / spread,
        })
    return sorted(rows, key=lambda r: -r["effect"])


def correlation_matrix(df: pd.DataFrame, target_column: str, positive_label: str,
                       numeric: list[str]) -> pd.DataFrame:
    """Pearson correlations among the measurements, with the target as 0/1."""
    frame = df[numeric].apply(pd.to_numeric, errors="coerce")
    frame[target_column] = _event(df, target_column, positive_label)
    return frame.corr()
