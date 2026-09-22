"""The gate: strangers may read, score, chat and send verdicts; only the owner publishes."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import access  # noqa: E402


class FakeQueryParams(dict):
    pass


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    import streamlit as st
    st.session_state.clear()
    access._chat_meter.clear()
    monkeypatch.setattr(st, "query_params", FakeQueryParams(), raising=False)
    yield
    st.session_state.clear()
    access._chat_meter.clear()


def _set_key(monkeypatch, value):
    monkeypatch.setattr(access, "_expected_key", lambda: value)


def _set_url_key(value):
    import streamlit as st
    st.query_params[access.KEY_PARAM] = value


def test_no_key_configured_fails_closed(monkeypatch):
    """A deployment that lost its secret goes quiet instead of opening up."""
    _set_key(monkeypatch, "")
    assert access.is_editor() is False


def test_stranger_without_the_key_cannot_publish(monkeypatch):
    _set_key(monkeypatch, "s3cret")
    assert access.is_editor() is False


def test_wrong_key_cannot_publish(monkeypatch):
    _set_key(monkeypatch, "s3cret")
    _set_url_key("s3cre")
    assert access.is_editor() is False


def test_owner_with_the_key_can_publish(monkeypatch):
    _set_key(monkeypatch, "s3cret")
    _set_url_key("s3cret")
    assert access.is_editor() is True


def test_chat_budget_runs_out_per_session(monkeypatch):
    for _ in range(access.SESSION_CHAT_BUDGET):
        allowed, _ = access.chat_allowed()
        assert allowed
        access.note_chat_call()
    allowed, reason = access.chat_allowed()
    assert allowed is False
    assert "session" in reason


def test_chat_budget_has_an_app_wide_ceiling(monkeypatch):
    """One visitor cannot burn the key for everyone; the app-wide meter stops it."""
    monkeypatch.setattr(access, "SESSION_CHAT_BUDGET", 10_000)
    for _ in range(access.HOURLY_CHAT_BUDGET):
        allowed, _ = access.chat_allowed()
        assert allowed
        access.note_chat_call()
    allowed, reason = access.chat_allowed()
    assert allowed is False
    assert "hour" in reason


def test_visitor_text_is_trimmed_and_flattened():
    assert access.clean_text("  two   words \n here ") == "two words here"
    assert access.clean_text("x" * 500, limit=200) == "x" * 200
    assert access.clean_text("") is None
    assert access.clean_text(None) is None


def test_key_is_derived_when_none_is_configured(monkeypatch):
    """No dashboard step needed: the gate derives its key from a secret already there."""
    monkeypatch.setattr(access, "_secret", lambda name, section=None: (
        "sb_publishable_example" if name == "SUPABASE_KEY" else ""))
    monkeypatch.delenv("EDITOR_KEY", raising=False)
    derived = access._expected_key()
    assert len(derived) == 16
    _set_url_key(derived)
    assert access.is_editor() is True


def test_an_explicit_key_wins_over_the_derived_one(monkeypatch):
    monkeypatch.setattr(access, "_secret", lambda name, section=None: (
        "chosen" if name == "EDITOR_KEY" else "sb_publishable_example"))
    _set_url_key("chosen")
    assert access.is_editor() is True


def test_no_secrets_at_all_stays_locked(monkeypatch):
    monkeypatch.setattr(access, "_secret", lambda name, section=None: "")
    monkeypatch.delenv("EDITOR_KEY", raising=False)
    assert access._expected_key() == ""
    assert access.is_editor() is False
