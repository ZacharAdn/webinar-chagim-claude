"""Step 1 -- raw CSV to a table Postgres and the model can both read.

Three rules, and each one is a talking point on rung 1:

  1. Column names become snake_case, because Postgres lowercases unquoted
     identifiers -- a column imported as "MonthlyCharges" has to be quoted in
     every query afterwards.
  2. A text column whose non-blank values are all numeric becomes numeric, and
     its blanks become real NULLs instead of silently becoming 0.0.
  3. The id column must be unique and the target column must have exactly two
     values. Either failure stops the run here rather than three steps later.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes

from ..context import LadderConfig, LadderError, write_result

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from insights import to_snake  # noqa: E402  -- src/ owns the naming rule

GENERATED_ID = "row_id"


def _numeric_or_none(series: pd.Series) -> pd.Series | None:
    """Return the coerced column when every non-blank value parses as a number.

    Text columns are `object` on pandas 2 and `str` on pandas 3, so the test is
    "not already numeric", never "dtype is object" -- that check silently did
    nothing on pandas 3 and left the blanks as the string " ".
    """
    if ptypes.is_numeric_dtype(series) or ptypes.is_bool_dtype(series):
        return None
    stripped = series.astype(str).str.strip()
    nonblank = stripped[stripped != ""]
    if nonblank.empty:
        return None
    if pd.to_numeric(nonblank, errors="coerce").notna().all():
        # "" coerces to NaN on its own, which is exactly the NULL we want.
        return pd.to_numeric(stripped, errors="coerce")
    return None


def prepare(cfg: LadderConfig) -> dict:
    if not cfg.raw_csv.exists():
        raise LadderError(f"raw file not found: {cfg.raw_csv}")

    df = pd.read_csv(cfg.raw_csv)
    df.columns = [to_snake(c) for c in df.columns]

    target = to_snake(cfg.target_column)
    if target not in df.columns:
        raise LadderError(
            f"target column '{cfg.target_column}' is not in {cfg.raw_csv.name}"
        )

    id_column = to_snake(cfg.id_column) if cfg.id_column else ""
    if id_column and id_column not in df.columns:
        raise LadderError(
            f"id column '{cfg.id_column}' is not in {cfg.raw_csv.name}"
        )
    if not id_column:
        id_column = GENERATED_ID
        df.insert(0, GENERATED_ID, range(1, len(df) + 1))
    elif df[id_column].duplicated().any():
        dupes = int(df[id_column].duplicated().sum())
        raise LadderError(
            f"id column '{id_column}' is not unique: {dupes} duplicate values"
        )

    df[target] = df[target].astype(str).str.strip()
    values = sorted(df[target].dropna().unique())
    if len(values) != 2:
        raise LadderError(
            f"target '{target}' has {len(values)} distinct values, expected 2: "
            + ", ".join(map(str, values[:5]))
        )
    if cfg.positive_label not in values:
        raise LadderError(
            f"positive_label '{cfg.positive_label}' is not one of {values}"
        )

    for column in df.columns:
        if column in (id_column, target):
            continue
        coerced = _numeric_or_none(df[column])
        if coerced is not None:
            df[column] = coerced

    cfg.prepared_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.prepared_csv, index=False)

    nulls = {c: int(n) for c, n in df.isna().sum().items() if n}
    positive_rate = float((df[target] == cfg.positive_label).mean())
    line = (
        f"{len(df):,} rows · {len(df.columns)} columns · "
        + (
            " · ".join(f"{n} NULLs in {c}" for c, n in nulls.items())
            if nulls
            else "no NULLs"
        )
        + f" · {positive_rate:.1%} positive"
    )
    return write_result(
        cfg,
        "prepare",
        "PASS",
        line,
        {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "nulls": nulls,
            "positive_rate": round(positive_rate, 4),
            "id_column": id_column,
            "target_column": target,
            "negative_label": next(v for v in values if v != cfg.positive_label),
        },
    )


def load_prepared(cfg: LadderConfig) -> pd.DataFrame:
    if not cfg.prepared_csv.exists():
        raise LadderError(
            f"{cfg.prepared_csv} is missing. Run: python ladder.py prepare"
        )
    return pd.read_csv(cfg.prepared_csv)


def copy_raw(cfg: LadderConfig, source: Path) -> Path:
    """Used by `init`: bring the user's CSV into the project as data/raw.csv."""
    cfg.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    if cfg.raw_csv.resolve() != Path(source).resolve():
        shutil.copyfile(source, cfg.raw_csv)
    return cfg.raw_csv
