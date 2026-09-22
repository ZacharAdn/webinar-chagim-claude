"""The customer scored on rung 2 is the customer rung 3 opens on."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import handoff  # noqa: E402

TYPED = {"contract": "Month-to-month", "tenure": 1.0, "monthly_charges": 95.0}
RECORD = {**TYPED, "internet_service": "Fiber optic", "total_charges": 95.0}


def test_default_record_is_none_before_rung_2_ran():
    assert handoff.default_record({}, "logreg", "customer_id") is None


def test_stash_then_default_carries_the_record_id_and_the_estimators_probability():
    session: dict = {}
    handoff.stash_scored(session, RECORD, {"logreg": 0.613, "tree": 0.892}, TYPED)

    record = handoff.default_record(session, "logreg", "customer_id")

    assert record["customer_id"].startswith("rung2:")
    assert record["probability"] == 0.613
    assert record["contract"] == "Month-to-month"
    assert record["total_charges"] == 95.0


def test_default_record_is_none_when_the_estimator_was_not_scored():
    session: dict = {}
    handoff.stash_scored(session, RECORD, {"tree": 0.892}, TYPED)

    assert handoff.default_record(session, "logreg", "customer_id") is None


def test_the_id_is_deterministic_in_the_typed_values():
    a, b = {}, {}
    handoff.stash_scored(a, RECORD, {"logreg": 0.6}, TYPED)
    handoff.stash_scored(b, dict(RECORD), {"logreg": 0.7}, dict(TYPED))

    assert a[handoff.KEY]["id"] == b[handoff.KEY]["id"]
