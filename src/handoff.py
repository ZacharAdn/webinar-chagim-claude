"""The record that crosses from rung 2 to rung 3.

Rung 2's form scores a customer who is not in the table. Rung 3 opens on that
same customer, so the demo reads as one story rather than two tabs that happen
to share a model. The state lives in st.session_state; this module only ever
sees a dict, so it can be tested without Streamlit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

KEY = "scored"
ID_PREFIX = "rung2:"


def stash_scored(session: dict, record: dict, probabilities: dict[str, float],
                 typed: dict) -> dict:
    """Remember the customer rung 2 just scored. Returns what was stored.

    The id is a hash of the typed fields, so scoring the same customer twice
    yields the same id -- which is what lets feedback rows attach to it.
    """
    digest = hashlib.sha1(
        json.dumps(typed, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    session[KEY] = {
        "record": dict(record),
        "probability": {k: float(v) for k, v in probabilities.items()},
        "typed": dict(typed),
        "id": ID_PREFIX + digest,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return session[KEY]


def default_record(session: dict, estimator: str, id_column: str) -> dict | None:
    """The rung-3 record built from the stash, or None when rung 2 has not run.

    The recommender reads one probability -- the configured estimator's -- so
    that is the one written into the record; the other family's number is
    display only and stays in the stash.
    """
    scored = session.get(KEY)
    if not scored or estimator not in scored.get("probability", {}):
        return None
    record = dict(scored["record"])
    record[id_column] = scored["id"]
    record["probability"] = float(scored["probability"][estimator])
    return record
