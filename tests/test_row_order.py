"""The live table must come back in the same order as the prepared CSV.

PostgREST leaves row order unspecified without an ORDER BY, and train_test_split
splits by position. An unordered read therefore produced different metrics in
the app than in REPORT.md -- same rows, same model, different numbers -- which
is the kind of discrepancy that destroys trust in a demo.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SECRETS = ROOT / ".streamlit" / "secrets.toml"


@pytest.fixture(scope="module")
def live_and_local():
    if not SECRETS.exists():
        pytest.skip("no .streamlit/secrets.toml -- nothing live to compare against")
    for key, value in tomllib.loads(SECRETS.read_text()).items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)

    import data as data_mod

    load = data_mod.load_records()
    if load.source != "supabase":
        pytest.skip(f"loader fell back to {load.source}: {load.detail}")
    local = pd.read_csv(ROOT / "data" / f"{data_mod.project_name()}.csv")
    return load.df, local, data_mod


def test_the_live_table_returns_every_row_exactly_once(live_and_local):
    live, local, data_mod = live_and_local
    key = data_mod.id_column()

    assert len(live) == len(local)
    assert not live[key].duplicated().any()


def test_model_metrics_do_not_depend_on_row_order():
    """The real invariant. Ordering the read is not enough on its own.

    Supabase hands back the same rows in a different order than the CSV, and
    train_test_split splits by position -- so without this, the app and
    REPORT.md quote different recall for the same model on the same data.
    """
    import model as model_mod

    spec = model_mod.Spec.from_toml(ROOT)
    frame = pd.read_csv(ROOT / "data" / f"{ROOT.name}.csv")
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    first = model_mod.train_model(frame, spec).metrics
    second = model_mod.train_model(shuffled, spec).metrics

    assert (first.accuracy, first.recall, first.roc_auc) == (
        second.accuracy, second.recall, second.roc_auc
    )
