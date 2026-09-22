"""Rung 2's 'score a new customer' helpers.

These test behaviour a demo depends on: that a partial record can be completed
from the table itself, that completing it produces a real probability, and that
two estimators asked the same question genuinely answer it differently -- which
is the assertion that catches the Streamlit cache returning one model twice.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model as model_mod  # noqa: E402


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    spec = model_mod.Spec.from_toml(ROOT)
    path = ROOT / "data" / f"{ROOT.name}.csv"
    if not path.exists():
        pytest.skip(f"{path} is missing -- run: python ladder.py prepare")
    return pd.read_csv(path), spec


def test_feature_defaults_covers_every_model_column_and_holds_no_nulls(frame):
    df, spec = frame
    numeric, categorical = model_mod.split_columns(df, spec)

    defaults = model_mod.feature_defaults(df, spec)

    assert set(defaults) == set(numeric + categorical)
    assert not [k for k, v in defaults.items() if pd.isna(v)]


def test_score_record_returns_a_probability_between_zero_and_one(frame):
    df, spec = frame
    trained = model_mod.train_model(df, spec)
    record = model_mod.feature_defaults(df, spec)

    probability = model_mod.score_record(trained, record)

    assert 0.0 <= probability <= 1.0


def test_the_two_estimators_score_the_same_record_differently(frame):
    df, spec = frame
    record = model_mod.feature_defaults(df, spec)
    record["contract"] = "Month-to-month"
    record["tenure"] = 1

    logreg = model_mod.score_record(
        model_mod.train_model(df, replace(spec, estimator="logreg")), record
    )
    tree = model_mod.score_record(
        model_mod.train_model(df, replace(spec, estimator="tree")), record
    )

    assert logreg != tree
