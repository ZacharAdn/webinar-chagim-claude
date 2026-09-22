"""The scoring job: score the table, keep the riskiest, write them down.

This used to be a button on rung 4. It moved here because a button that
writes to a table nobody in the room can open proves nothing on stage; the
ladder's app step runs the job instead, and the app shows the audit trail
that the job left behind. No Streamlit import, so the runner can call it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

import data as data_mod
import model as model_mod
from agent import Bands

DEFAULT_BATCH = 25


def score_and_write(df: pd.DataFrame, spec: model_mod.Spec,
                    trained: model_mod.TrainedModel, rules, conn,
                    n: int = DEFAULT_BATCH) -> tuple[bool, str, list[dict]]:
    """Score every record, write the n riskiest with their action. (ok, message, rows)."""
    scored = model_mod.score_records(trained, df, spec, [])
    bands = Bands.from_rules(rules)
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "record_id": str(row[spec.id_column]),
            "probability": round(float(row["probability"]), 4),
            "recommended_action": bands.recommend(row).action,
            "model_version": f"{trained.version} · rules v{rules.version}",
            "created_at": now,
        }
        for row in scored.head(n).to_dict("records")
    ]
    ok, message = data_mod.write_predictions(conn, rows)
    return ok, message, rows
