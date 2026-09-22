"""The scoring job is a function the ladder runs, not a button on the screen."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import batch  # noqa: E402
import data as data_mod  # noqa: E402
import model as model_mod  # noqa: E402
import rules_store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_score_and_write_puts_the_riskiest_rows_first_and_stamps_both_versions(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(data_mod, "LOCAL_PREDICTIONS", tmp_path / "predictions.csv")
    df = pd.read_csv(ROOT / "data" / "webina-sep.csv")
    spec = model_mod.Spec.from_toml(ROOT)
    trained = model_mod.load_model(model_mod.model_path(ROOT, spec.estimator))
    rules = rules_store.from_toml(ROOT)

    ok, message, rows = batch.score_and_write(df, spec, trained, rules, conn=None, n=7)

    assert ok, message
    assert len(rows) == 7
    probabilities = [row["probability"] for row in rows]
    assert probabilities == sorted(probabilities, reverse=True)
    assert rows[0]["model_version"] == f"{trained.version} · rules v{rules.version}"
    assert set(rows[0]) == {
        "record_id", "probability", "recommended_action", "model_version", "created_at",
    }
    written = pd.read_csv(tmp_path / "predictions.csv")
    assert len(written) == 7
