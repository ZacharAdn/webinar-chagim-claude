"""Step 3 -- run the app headless, then run the scoring job, count what happened.

AppTest executes the whole script, so every tab's body runs even though nothing
is clicked. That is the point: an exception hiding in tab 3 fails here rather
than on stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from ..context import LadderConfig, LadderError, write_result

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import batch  # noqa: E402
import model as model_mod  # noqa: E402
import rules_store  # noqa: E402
from rungs import SUBHEADS  # noqa: E402  -- no Streamlit import, no page config

TIMEOUT = 300


def run(cfg: LadderConfig) -> dict:
    app_file = cfg.root / "src" / "app.py"
    if not app_file.exists():
        raise LadderError(f"{app_file} is missing")

    at = AppTest.from_file(str(app_file), default_timeout=TIMEOUT).run()
    if at.exception:
        raise LadderError(f"app raised on load: {at.exception[0].value.splitlines()[0]}")

    rendered = [element.value for element in at.subheader]
    rungs = sum(1 for head in SUBHEADS if head in rendered)
    if rungs != 4:
        raise LadderError(f"{rungs} of 4 rungs rendered: {rendered}")

    # The job. It used to be a button on rung 4; the step presses nothing now
    # and runs the same function the button ran, against the saved model.
    from .prepare import load_prepared
    df = load_prepared(cfg)
    spec = model_mod.Spec.from_config(cfg)
    trained = model_mod.load_model(model_mod.model_path(cfg.root, spec.estimator))
    if trained is None:
        raise LadderError("no saved model -- run: python ladder.py model")
    conn = _client(cfg)
    rules = rules_store.load_active(conn) if conn is not None else rules_store.from_toml(cfg.root)

    before = _prediction_count(cfg)
    ok, message, _rows = batch.score_and_write(df, spec, trained, rules, conn)
    if not ok:
        raise LadderError(message)
    after = _prediction_count(cfg)
    written = max(after - before, 0)

    destination = "Supabase" if conn is not None else "data/predictions.csv"
    line = f"4 rungs · 0 exceptions · {written} rows to {destination}"
    return write_result(
        cfg,
        "app",
        "PASS",
        line,
        {
            "rungs": rungs,
            "exceptions": 0,
            "rows_written": written,
            "destination": destination,
            "model_version": trained.version,
            "rules_version": rules.version,
        },
    )


def _client(cfg: LadderConfig):
    """A plain Supabase client from the secrets file, or None outside the cloud."""
    from . import verify_step

    try:
        url, key = verify_step.secrets(cfg)
    except LadderError:
        return None
    from supabase import create_client

    return create_client(url, key)


def _prediction_count(cfg: LadderConfig) -> int:
    """Rows in both destinations, summed.

    The app writes to Supabase the moment secrets exist and falls back to the
    CSV when they do not. Counting only the CSV made rows_written 0 on every
    live run -- the step still passed, but the assertion built on it could not.
    Summing both means the number grows wherever the click actually landed.
    """
    total = 0
    path = cfg.root / "data" / "predictions.csv"
    if path.exists():
        total += max(sum(1 for _ in path.open(encoding="utf-8")) - 1, 0)
    return total + _supabase_count(cfg)


def _supabase_count(cfg: LadderConfig) -> int:
    """Rows in the live predictions table, or 0 when there is no live table."""
    from . import verify_step
    from ..schema import PREDICTIONS_TABLE

    try:
        url, key = verify_step.secrets(cfg)
        _, headers, _ = verify_step.rest(
            url, key, f"{PREDICTIONS_TABLE}?select=id&limit=1",
            headers={"Prefer": "count=exact", "Range": "0-0"},
        )
        return verify_step.parse_count(headers.get("content-range"))
    except LadderError:
        return 0
