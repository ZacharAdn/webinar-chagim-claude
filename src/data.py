"""Rung 1's plumbing -- read records from Supabase, fall back to the CSV.

Two things here are not obvious and both bite in front of an audience:

  1. st_supabase_connection 2.x removed conn.query(...). The Streamlit tutorial
     still shows it. In 2.1.3 the query surface is conn.table(...), which is the
     supabase-py builder.
  2. PostgREST caps a response at 1,000 rows. A dashboard that ignores this
     silently renders 1,000 of 7,043 rows and every number on the page is wrong.
     fetch_table pages through with .range() instead.

If Supabase is unreachable -- no secrets, wifi dies mid-demo -- the loader falls
back to the prepared CSV and says so out loud rather than showing an error.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from insights import to_snake

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PREDICTIONS = REPO_ROOT / "data" / "predictions.csv"
PAGE_SIZE = 1000
RECORDS_TABLE = "records"
PREDICTIONS_TABLE = "predictions"
FEEDBACK_TABLE = "feedback"


def project_name() -> str:
    with (REPO_ROOT / "ladder.toml").open("rb") as fh:
        return str(tomllib.load(fh)["project"]["name"])


def local_csv() -> Path:
    return REPO_ROOT / "data" / f"{project_name()}.csv"


def id_column() -> str:
    """The column every paged read is ordered by."""
    with (REPO_ROOT / "ladder.toml").open("rb") as fh:
        dataset = tomllib.load(fh)["dataset"]
    return to_snake(str(dataset.get("id_column", "") or "")) or "row_id"


@dataclass
class LoadResult:
    df: pd.DataFrame
    source: str          # "supabase" or "csv"
    detail: str = ""


def get_connection():
    """Return a live Supabase connection, or None if it cannot be built."""
    try:
        from st_supabase_connection import SupabaseConnection

        return st.connection("supabase", type=SupabaseConnection)
    except Exception:  # noqa: BLE001
        return None


def fetch_table(conn, table: str, order_by: str = "",
                page_size: int = PAGE_SIZE) -> pd.DataFrame:
    """Read a whole table, paging past the 1,000-row PostgREST ceiling.

    The ORDER BY is not decoration. PostgREST leaves row order unspecified
    without one, so paging with .range() walks an order the server never
    promised to keep -- a row can appear on two pages and be missing from the
    result. Even when the rows all arrive, they arrive in a different order than
    the CSV, and train_test_split splits by position: the app reported recall
    0.540 where REPORT.md said 0.559, on the same 7,043 rows and the same model.
    """
    frames: list[pd.DataFrame] = []
    start = 0
    while True:
        query = conn.table(table).select("*")
        if order_by:
            query = query.order(order_by)
        rows = query.range(start, start + page_size - 1).execute().data
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < page_size:
            break
        start += page_size
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl="10m", show_spinner="Reading records...")
def load_records() -> LoadResult:
    conn = get_connection()
    if conn is not None:
        try:
            df = fetch_table(conn, RECORDS_TABLE, order_by=id_column())
            if not df.empty:
                return LoadResult(df, "supabase", f"{len(df):,} rows from Supabase")
        except Exception as exc:  # noqa: BLE001 -- the fallback is the point
            return _local(
                f"Supabase unavailable ({type(exc).__name__}), using the local file"
            )
    return _local("no Supabase connection configured, using the local file")


def _local(detail: str) -> LoadResult:
    path = local_csv()
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: python ladder.py prepare")
    return LoadResult(pd.read_csv(path), "csv", detail)


def write_predictions(conn, rows: list[dict]) -> tuple[bool, str]:
    """Rung 4: the model's output becomes a row someone else can read.

    Without a Supabase connection the rows are appended to data/predictions.csv
    instead, so the write-then-read-back moment still happens on stage. That
    file lives only on the machine (or container) running the app -- which is
    exactly the difference between a local file and a table, and the caption in
    the app says so.
    """
    if conn is None:
        try:
            LOCAL_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
            header = not LOCAL_PREDICTIONS.exists()
            pd.DataFrame(rows).to_csv(
                LOCAL_PREDICTIONS, mode="a", header=header, index=False
            )
            return True, (
                f"{len(rows)} predictions written to data/{LOCAL_PREDICTIONS.name} "
                "(local file -- no Supabase connection)"
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"local write failed: {type(exc).__name__}: {exc}"
    try:
        conn.table(PREDICTIONS_TABLE).insert(rows).execute()
        return True, f"{len(rows)} predictions written to {PREDICTIONS_TABLE}"
    except Exception as exc:  # noqa: BLE001
        return False, f"write failed: {type(exc).__name__}: {exc}"


def read_predictions(conn, limit: int = 20) -> pd.DataFrame:
    if conn is None:
        if not LOCAL_PREDICTIONS.exists():
            return pd.DataFrame()
        try:
            local = pd.read_csv(LOCAL_PREDICTIONS)
            return (
                local.sort_values("created_at", ascending=False)
                .head(limit)
                .reset_index(drop=True)
            )
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    try:
        rows = (
            conn.table(PREDICTIONS_TABLE)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
            .data
        )
        return pd.DataFrame(rows)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


# --------------------------------------------------------------------------
# The loop -- feedback in, outcomes back, and the numbers rung 4 reports
# --------------------------------------------------------------------------
def write_feedback(conn, row: dict) -> tuple[bool, str]:
    """One person's verdict on one recommendation. Local mode keeps it in session only."""
    if conn is None:
        st.session_state.setdefault("local_feedback", []).append(row)
        return True, "recorded in this session only (no Supabase connection)"
    try:
        conn.table(FEEDBACK_TABLE).insert(row).execute()
        if row.get("prediction_id") and row.get("actual_outcome") in ("stayed", "left"):
            conn.table(PREDICTIONS_TABLE).update({
                "outcome": row["actual_outcome"],
                "outcome_at": row.get("created_at"),
            }).eq("id", row["prediction_id"]).execute()
        return True, "feedback written to Supabase"
    except Exception as exc:  # noqa: BLE001
        return False, f"feedback write failed: {type(exc).__name__}: {exc}"


def read_feedback(conn, limit: int = 200) -> list[dict]:
    if conn is None:
        return list(st.session_state.get("local_feedback", []))
    try:
        return (
            conn.table(FEEDBACK_TABLE).select("*")
            .order("created_at", desc=True).limit(limit).execute().data
        )[::-1]
    except Exception:  # noqa: BLE001
        return []


def latest_prediction_for(conn, record_id: str) -> dict | None:
    """The most recent logged decision for a record, so feedback can point at it."""
    if conn is None:
        return None
    try:
        rows = (
            conn.table(PREDICTIONS_TABLE).select("id,probability,recommended_action,created_at")
            .eq("record_id", record_id).order("created_at", desc=True).limit(1)
            .execute().data
        )
        return rows[0] if rows else None
    except Exception:  # noqa: BLE001
        return None


def loop_counts(conn) -> dict:
    """Decisions logged, how many carry an outcome, how many verdicts came in."""
    if conn is None:
        return {"logged": 0, "with_outcome": 0, "verdicts": len(read_feedback(conn))}
    try:
        logged = conn.table(PREDICTIONS_TABLE).select("id", count="exact").limit(1).execute().count or 0
        with_outcome = (
            conn.table(PREDICTIONS_TABLE).select("id", count="exact")
            .not_.is_("outcome", "null").limit(1).execute().count or 0
        )
        verdicts = conn.table(FEEDBACK_TABLE).select("id", count="exact").limit(1).execute().count or 0
        return {"logged": logged, "with_outcome": with_outcome, "verdicts": verdicts}
    except Exception:  # noqa: BLE001
        return {"logged": 0, "with_outcome": 0, "verdicts": 0}
