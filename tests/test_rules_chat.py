"""The console: says what the rules are, proposes changes, never writes."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules_chat  # noqa: E402
from rules_store import Band, RuleSet  # noqa: E402

V1 = RuleSet(1, (
    Band("high", 0.60, "Contact this week and make a concrete offer"),
    Band("medium", 0.35, "Add to next month's outreach list"),
    Band("low", 0.00, "No action"),
), "toml", "ladder.toml")

GOOD_BLOCK = (
    "I would lower the high floor.\n\n```json\n"
    '{"bands": [{"name": "high", "min": 0.5, "action": "Contact this week and make a concrete offer"},'
    '{"name": "medium", "min": 0.35, "action": "Add to next month\'s outreach list"},'
    '{"name": "low", "min": 0.0, "action": "No action"}], "rationale": "Two verdicts said high fired too late."}\n```'
)


class StubClient:
    """Looks enough like the OpenAI client for reply() to call it."""

    def __init__(self, content: str):
        self.content = content
        self.chat = self
        self.completions = self
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_opening_message_names_every_band_and_the_customers_band(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    text = rules_chat.opening_message(V1, {"probability": 0.613})

    for band in V1.bands:
        assert band.name in text and band.action in text
    assert "61%" in text and "'high'" in text
    assert "v1" in text


def test_extract_bands_reads_the_last_block_and_validates_it():
    bands, rationale = rules_chat.extract_bands(GOOD_BLOCK, current=V1)

    assert bands is not None
    assert bands[0] == Band("high", 0.5, "Contact this week and make a concrete offer")
    assert rationale.startswith("Two verdicts")


def test_extract_bands_returns_none_when_the_block_fails_normalise():
    bad = GOOD_BLOCK.replace('"name": "low", "min": 0.0', '"name": "low", "min": 0.1')

    assert rules_chat.extract_bands(bad, current=V1) == (None, "")


def test_extract_bands_returns_none_when_band_names_change():
    renamed = GOOD_BLOCK.replace('"name": "medium"', '"name": "mid"')

    assert rules_chat.extract_bands(renamed, current=V1) == (None, "")


def test_extract_bands_returns_none_without_a_block():
    assert rules_chat.extract_bands("The rules look fine to me.") == (None, "")


def test_reply_returns_prose_without_the_block_and_the_parsed_bands():
    client = StubClient(GOOD_BLOCK)
    history = [{"role": "user", "content": "Is 60% too high a floor?"}]

    turn = rules_chat.reply(history, "system text", current=V1, client=client)

    assert turn is not None
    assert turn.text == "I would lower the high floor."
    assert turn.bands is not None and turn.bands[0].min == 0.5
    assert client.calls[0]["messages"][0] == {"role": "system", "content": "system text"}
    assert client.calls[0]["messages"][1] == history[0]
    assert "response_format" not in client.calls[0]


def test_reply_is_none_without_a_key_and_without_a_client(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert rules_chat.reply([{"role": "user", "content": "hi"}], "s") is None


def test_chat_key_changes_with_the_customer_and_with_the_rules_version():
    """Streamlit renders every tab on every run, so the console can be born on a
    different customer than the one rung 3 later shows. The session key must
    move with the customer, or the opening message goes stale."""
    v2 = RuleSet(2, V1.bands, "learner:rules", "moved")

    same = rules_chat.chat_key(V1, "rung2:abc123")
    other_customer = rules_chat.chat_key(V1, "7590-VHVEG")
    other_version = rules_chat.chat_key(v2, "rung2:abc123")

    assert same == rules_chat.chat_key(V1, "rung2:abc123")
    assert same != other_customer
    assert same != other_version
    assert "v1" in same and "rung2:abc123" in same


class FailingClient(StubClient):
    """The provider rejects the call -- what a revoked key looks like."""

    def create(self, **kwargs):
        raise RuntimeError("Error code: 401 - Invalid API Key")


def test_reply_carries_the_providers_error_instead_of_swallowing_it():
    turn = rules_chat.reply([{"role": "user", "content": "so 15%?"}], "s",
                            current=V1, client=FailingClient(""))
    assert turn is not None
    assert turn.bands is None
    assert "401" in turn.error
    assert turn.text == ""
