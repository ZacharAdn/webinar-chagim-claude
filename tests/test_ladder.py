"""The checks REPORT.md is built from. Run: pytest tests/test_ladder.py

Every assertion here is about a step's recorded result, never about a number
this project happens to have produced -- so the same file is meaningful on any
dataset. Cloud steps are skipped when they were skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ladder.context import LadderConfig, read_result  # noqa: E402

CFG = LadderConfig.load(ROOT)


def result(step: str) -> dict:
    payload = read_result(CFG, step)
    if payload is None:
        pytest.skip(f"step '{step}' has not run")
    if payload["status"] == "SKIPPED":
        pytest.skip(f"step '{step}' was skipped: {payload['line']}")
    return payload


def test_prepare_produced_a_non_empty_two_class_table():
    data = result("prepare")["data"]
    assert data["rows"] > 0
    assert data["columns"] >= 2
    assert 0.0 < data["positive_rate"] < 1.0
    assert CFG.prepared_csv.exists()


def test_the_model_beats_its_own_baseline_on_recall():
    data = result("model")["data"]
    assert data["model"]["recall"] > data["baseline"]["recall"]
    assert data["model"]["roc_auc"] > 0.5


def test_the_leaky_split_scores_higher_than_the_honest_one():
    data = result("model")["data"]
    assert data["leaky"]["recall"] > data["honest"]["recall"]


def test_the_app_rendered_four_rungs_and_raised_nothing():
    data = result("app")["data"]
    assert data["rungs"] == 4
    assert data["exceptions"] == 0
    assert data["rows_written"] > 0


def test_supabase_imported_every_prepared_row():
    assert result("supabase")["data"]["rows"] == result("prepare")["data"]["rows"]


def test_the_repo_is_public_when_the_config_says_so():
    assert result("github")["data"]["visibility"] == CFG.github_visibility


def test_the_deployed_app_answered_and_a_write_came_back():
    data = result("verify")["data"]
    assert data["rows"] > 0
    assert data["prediction_id"]
    if data["app_url"]:
        assert data["app_status"] == "live"


def test_every_recorded_result_is_valid_json_with_a_status():
    for path in sorted(CFG.results_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        assert payload["status"] in {"PASS", "FAIL", "SKIPPED", "PENDING"}
