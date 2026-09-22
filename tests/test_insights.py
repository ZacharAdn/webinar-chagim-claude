"""Rung 1's story helpers: what moves with the target, and how."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import insights  # noqa: E402


def _frame() -> pd.DataFrame:
    # `tracks` follows the target exactly; `noise` does not; `flag` is a
    # categorical that follows it too.
    return pd.DataFrame(
        {
            "id": range(12),
            "tracks": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            "noise": [5.0, 3.0, 5.0, 3.0, 5.0, 3.0, 5.0, 3.0, 5.0, 3.0, 5.0, 3.0],
            "flag": list("aaaaaabbbbbb"),
            "y": list("NNNNNNYYYYYY"),
        }
    )


def test_associations_rank_every_column_by_its_link_to_the_target():
    ranked = insights.associations(_frame(), "y", "Y", "id", 10)
    names = [row["column"] for row in ranked]
    assert set(names) == {"tracks", "noise", "flag"}
    assert names[-1] == "noise"
    by_name = {row["column"]: row for row in ranked}
    assert by_name["tracks"]["kind"] == "numeric"
    assert by_name["flag"]["kind"] == "categorical"
    assert by_name["tracks"]["strength"] > 0.99
    assert by_name["flag"]["strength"] > 0.99
    assert by_name["noise"]["strength"] < 0.05


def test_numeric_separation_reports_the_medians_of_the_two_groups():
    rows = insights.numeric_separation(_frame(), "y", "Y", ["tracks", "noise"])
    assert rows[0]["column"] == "tracks"
    assert rows[0]["median_positive"] == 9.0
    assert rows[0]["median_negative"] == 1.0
    assert rows[0]["effect"] > rows[1]["effect"]


def test_correlation_matrix_carries_the_target_as_a_number():
    matrix = insights.correlation_matrix(_frame(), "y", "Y", ["tracks", "noise"])
    assert list(matrix.columns) == ["tracks", "noise", "y"]
    assert matrix.loc["tracks", "y"] > 0.99
    assert abs(matrix.loc["noise", "y"]) < 0.05
