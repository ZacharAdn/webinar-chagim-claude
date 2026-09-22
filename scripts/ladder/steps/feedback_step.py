"""Step 8 -- the loop, measured.

Every rung below this one teaches the system from someone else's past. This one
asks the only question that uses its own present: of the decisions it has
actually made, how many turned out to be right?

On the day the ladder is built the answer is "none yet, and here is the empty
column that will hold it". That is a PASS, not a failure -- the column existing
and being reachable is the whole deliverable, because a loop nobody can close is
the most common way a model dies quietly in production.

The order this step implies is the lesson: a disagreement between the
recommendation and the outcome is a rung 3 fix long before it is a rung 2 fix.
Ten rows are enough to correct a band or a prompt; a retrain needs thousands.
"""

from __future__ import annotations

from .. import context
from ..context import LadderConfig, write_result
from ..schema import PREDICTIONS_TABLE
from .verify_step import parse_count, rest, secrets


def counts(url: str, key: str) -> tuple[int, int, int]:
    """Total predictions, how many carry an outcome, and how many agreed.

    'Agreed' is deliberately crude: the model said more likely than not, and the
    outcome says the event happened. A sharper metric belongs on rung 2 once
    there are enough rows to justify one.
    """
    _, headers, _ = rest(
        url, key, f"{PREDICTIONS_TABLE}?select=id&limit=1",
        headers={"Prefer": "count=exact", "Range": "0-0"},
    )
    total = parse_count(headers.get("content-range"))

    _, headers, _ = rest(
        url, key,
        f"{PREDICTIONS_TABLE}?select=id&outcome=not.is.null"
        "&record_id=neq.ladder-verify&limit=1",
        headers={"Prefer": "count=exact", "Range": "0-0"},
    )
    labelled = parse_count(headers.get("content-range"))

    agreed = 0
    if labelled:
        _, _, rows = rest(
            url, key,
            f"{PREDICTIONS_TABLE}?select=probability,outcome"
            "&outcome=not.is.null&record_id=neq.ladder-verify&limit=1000",
        )
        for row in rows or []:
            probability = row.get("probability")
            outcome = str(row.get("outcome", "")).strip().lower()
            if probability is None or outcome == "":
                continue
            predicted_yes = float(probability) >= 0.5
            happened = outcome in ("1", "yes", "true", "y", "כן")
            if predicted_yes == happened:
                agreed += 1
    return total, labelled, agreed


def run(cfg: LadderConfig) -> dict:
    url, key = secrets(cfg)
    total, labelled, agreed = counts(url, key)

    if labelled == 0:
        line = (
            f"{total:,} decisions logged · 0 outcomes yet — "
            "the column is open and waiting"
        )
    else:
        rate = agreed / labelled
        fix = "rung 3, the bands or the prompt" if rate < 0.7 else "nothing yet"
        line = (
            f"{total:,} decisions logged · {labelled:,} with an outcome · "
            f"{rate:.0%} agreed · fix first: {fix}"
        )

    return write_result(
        cfg, "feedback", "PASS", line,
        {"logged": total, "with_outcome": labelled, "agreed": agreed},
    )
