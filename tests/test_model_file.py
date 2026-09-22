"""The trained model is a file in the repo, and its version is a fingerprint.

Retraining is deterministic, so the file is not about reproducibility; it is
about a version that changes exactly when the model does, and an app that
scores the same customer the same way after every reboot.
"""

import dataclasses
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import model as model_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frame():
    return pd.read_csv(ROOT / "data" / "webina-sep.csv")


@pytest.fixture(scope="module")
def spec():
    return model_mod.Spec.from_toml(ROOT)


def test_save_then_load_scores_the_same_and_keeps_the_version(frame, spec, tmp_path):
    trained = model_mod.train_model(frame, spec)
    path = model_mod.save_model(trained, tmp_path / "logreg.joblib")
    loaded = model_mod.load_model(path)
    record = model_mod.feature_defaults(frame, spec)
    assert loaded.version == trained.version
    assert model_mod.score_record(loaded, record) == pytest.approx(
        model_mod.score_record(trained, record)
    )
    assert loaded.metrics.as_row() == trained.metrics.as_row()


def test_version_is_a_fingerprint_of_the_model_not_a_constant(frame, spec):
    twice = [model_mod.train_model(frame, spec).version for _ in range(2)]
    assert twice[0] == twice[1]
    assert twice[0].startswith("logreg-")
    assert twice[0] != "logreg-v1"
    tree = model_mod.train_model(frame, dataclasses.replace(spec, estimator="tree"))
    assert tree.version != twice[0]


def test_load_model_returns_none_when_there_is_no_file(tmp_path):
    assert model_mod.load_model(tmp_path / "missing.joblib") is None


def test_model_path_lives_under_models_by_estimator():
    assert model_mod.model_path(ROOT, "tree") == ROOT / "models" / "tree.joblib"
