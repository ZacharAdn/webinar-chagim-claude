"""Agent two, offline.

The learner has to be trustworthy with no key and no network, because that is
the state it will be in the moment a demo needs it. So every path through
propose_rules is pinned here: it moves an action when people said which one was
better, it raises a floor when 'high risk' customers kept staying, and it
leaves alone what the evidence does not touch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import learner  # noqa: E402
import rules_store  # noqa: E402


@pytest.fixture
def rules() -> rules_store.RuleSet:
    return rules_store.from_toml(ROOT)


def fb(probability, verdict="wrong", outcome="", better=""):
    return {
        "probability": probability, "verdict": verdict,
        "actual_outcome": outcome, "better_action": better,
    }


def test_toml_rules_are_version_one_and_sorted_high_to_low(rules):
    assert rules.version == 1
    floors = [b.min for b in rules.bands]
    assert floors == sorted(floors, reverse=True)
    assert floors[-1] == 0.0


def test_normalise_rejects_a_set_with_no_zero_floor():
    with pytest.raises(ValueError):
        rules_store.normalise([{"name": "a", "min": 0.3, "action": "x"}])


def test_digest_folds_each_row_onto_the_band_its_probability_falls_in(rules):
    rows = [fb(0.9, "wrong", "stayed", "Call"), fb(0.4, "right", "left"), fb(0.1, "right")]
    d = learner.digest(rules, rows)
    assert d["high"].total == 1 and d["high"].wrong == 1 and d["high"].stayed == 1
    assert d["medium"].total == 1 and d["medium"].left == 1
    assert d["low"].total == 1 and d["low"].wrong == 0


def test_rules_learner_leaves_everything_alone_below_the_threshold(rules):
    rows = [fb(0.9, "wrong", "stayed", "Call"), fb(0.9, "wrong", "stayed", "Call")]
    got = learner.propose_rules(rules, rows)
    assert not got.changed
    assert "no band crossed" in got.rationale


def test_rules_learner_adopts_the_action_people_kept_suggesting(rules):
    rows = [fb(0.7, "wrong", "stayed", "Call before offering anything") for _ in range(3)]
    got = learner.propose_rules(rules, rows)
    high = next(b for b in got.bands if b.name == "high")
    assert high.action == "Call before offering anything"
    assert high.min == 0.60
    assert any("high: action" in c for c in got.changes)
    assert got.source == "learner:rules"


def test_rules_learner_raises_a_floor_when_high_risk_customers_kept_staying(rules):
    rows = [fb(0.65, "wrong", "stayed") for _ in range(4)]
    got = learner.propose_rules(rules, rows)
    high = next(b for b in got.bands if b.name == "high")
    assert high.min == pytest.approx(0.65)
    assert high.action == rules.band_for(0.65).action


def test_rules_learner_never_touches_the_zero_floor(rules):
    rows = [fb(0.05, "wrong", "stayed") for _ in range(5)]
    got = learner.propose_rules(rules, rows)
    assert got.bands[-1].min == 0.0


def test_proposal_diff_names_every_change(rules):
    after = rules_store.normalise([
        {"name": "high", "min": 0.7, "action": "Call"},
        {"name": "medium", "min": 0.35, "action": "Add to next month's outreach list"},
        {"name": "low", "min": 0.0, "action": "No action"},
    ])
    changes = learner._diff(rules, after)
    assert changes == ["high: floor 60% -> 70%", "high: action 'Contact this week and make a concrete offer' -> 'Call'"]


def test_llm_learner_is_none_without_a_key(rules, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert learner.propose_llm(rules, [fb(0.9)]) is None


def test_propose_falls_back_to_rules_when_llm_is_absent(rules, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    got = learner.propose(rules, [fb(0.7, "wrong", "stayed", "Call")] * 3)
    assert got.source == "learner:rules"


def test_three_who_left_after_a_soft_call_lower_the_floor_of_the_band_above(rules):
    """The mirror of the floor-rises rule: medium customers kept leaving, so
    the high band must start lower and catch them."""
    feedback = [fb(0.577, "wrong", "left") for _ in range(3)]
    proposal = learner.propose_rules(rules, feedback)
    floors = {b.name: b.min for b in proposal.bands}
    assert proposal.changed
    assert floors["high"] == pytest.approx(0.55)
    assert floors["medium"] == pytest.approx(0.35)
    assert "left" in proposal.rationale
    assert proposal.changes == ["high: floor 60% -> 55%"]


def test_propose_prefers_a_rule_that_moved_over_a_model_that_shrugged(rules, monkeypatch):
    """The LLM is a second opinion, not a veto: when the deterministic learner
    has crossed its threshold and moved a floor, that is the proposal."""
    feedback = [fb(0.577, "wrong", "left") for _ in range(3)]
    unchanged = learner.Proposal(rules, rules.bands, "no evidence", "learner:groq", {}, [])
    monkeypatch.setattr(learner, "propose_llm", lambda *a, **k: unchanged)
    proposal = learner.propose(rules, feedback, prefer_llm=True)
    assert proposal.changes == ["high: floor 60% -> 55%"]
    assert proposal.source == "learner:rules"


def test_propose_asks_the_model_only_when_the_rules_find_nothing(rules, monkeypatch):
    feedback = [fb(0.577, "right", "stayed")]
    moved = rules_store.normalise([{"name": "high", "min": 0.5, "action": "x"},
                                   {"name": "medium", "min": 0.35, "action": "y"},
                                   {"name": "low", "min": 0.0, "action": "z"}])
    from_model = learner.Proposal(rules, moved, "model says so", "learner:groq", {}, ["high: floor 60% -> 50%"])
    monkeypatch.setattr(learner, "propose_llm", lambda *a, **k: from_model)
    assert learner.propose(rules, feedback, prefer_llm=True) is from_model
    assert learner.propose(rules, feedback, prefer_llm=False).source == "learner:rules"
