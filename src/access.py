"""Who is allowed to change what everyone else sees.

The demo is meant to be tried by strangers, so almost everything stays open:
reading, scoring a customer, getting a recommendation, talking to the rules and
sending a verdict are all anonymous. Two things are not, because they replace the
active rules for every other visitor at once - applying the console's proposal and
publishing the learner's revision. Those sit behind a key the owner carries in the
link (`?k=...`), checked against EDITOR_KEY in the app's secrets.

The chat is open but not unlimited: every message is a model call on the owner's
API key, so there is a per-visitor budget and an app-wide hourly ceiling. A visitor
who runs out is told plainly; the rest of the page keeps working.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import tomllib
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
KEY_PARAM = "k"
SESSION_CHAT_BUDGET = 12
HOURLY_CHAT_BUDGET = 150
_WINDOW = 3600.0

# What a visitor sees where a publish button would be. Says what the button does
# and why it is held, so the loop still reads as real rather than broken.
LOCKED_NOTE = (
    "Publishing replaces the live rules for everyone looking at this app right "
    "now, so it stays with the owner during the demo. Everything above is real: "
    "the proposal you are reading was computed from the verdicts in the database."
)


def _secret(name: str, section: str | None = None) -> str:
    try:
        block = st.secrets[section] if section else st.secrets
        return str(block.get(name, "") or "")
    except Exception:  # noqa: BLE001 - no secrets file, or no such section
        return ""


def _expected_key() -> str:
    """The value a link must carry in ?k= to publish.

    EDITOR_KEY wins when it is set. Otherwise the key is derived from a secret the
    deployment already holds, so the gate works without anyone touching the hosting
    dashboard: the derivation lives in this file, the input does not, and a repo
    reader cannot reproduce it. Rotating that secret changes the link - which is
    the safe direction, since the gate then locks rather than opens.
    """
    explicit = str(_secret("EDITOR_KEY") or os.environ.get("EDITOR_KEY", "")).strip()
    if explicit:
        return explicit
    seed = _secret("SUPABASE_KEY", "connections.supabase") or _secret("GROQ_API_KEY")
    if not seed:
        return ""
    return derive_key(_project_name(), seed)


def derive_key(project: str, seed: str) -> str:
    """The publish key for one project. supabase_setup.sh computes the same value
    and stores it in the database's publish_gate, so the app and the database agree
    without the key ever being written into the repository."""
    return hashlib.sha256(f"{project}|publish-gate|{seed}".encode()).hexdigest()[:16]


def _project_name() -> str:
    try:
        with (REPO_ROOT / "ladder.toml").open("rb") as fh:
            return str(tomllib.load(fh)["project"]["name"])
    except Exception:  # noqa: BLE001 - no ladder.toml, no name; the seed still differs
        return "ladder"


def is_editor() -> bool:
    """True when the visitor carries the editor key in the URL.

    Fails closed on purpose: with no EDITOR_KEY configured nobody can publish.
    A deployment that lost its secret should go quiet, not hand the live rules to
    whoever opens the link. Local work puts the key in .streamlit/secrets.toml.
    """
    expected = _expected_key()
    if not expected:
        return False
    try:
        supplied = st.query_params.get(KEY_PARAM, "")
    except Exception:  # noqa: BLE001 - older Streamlit
        supplied = ""
    if isinstance(supplied, list):
        supplied = supplied[0] if supplied else ""
    return hmac.compare_digest(str(supplied).strip(), expected)


def publish_token() -> str:
    """What the database asks for before it will replace the rules.

    Same value the URL must carry, so one secret governs both sides: the app will
    not draw the button, and the database will not honour the write.
    """
    return _expected_key()


@st.cache_resource
def _chat_meter() -> dict:
    """Shared across every session in this app process; resets on reboot."""
    return {"window_started": time.time(), "calls": 0}


def chat_allowed() -> tuple[bool, str]:
    """Budget check for one chat message. Returns (allowed, reason if not)."""
    used = st.session_state.get("chat_calls_used", 0)
    if used >= SESSION_CHAT_BUDGET:
        return False, (
            f"You have used this session's {SESSION_CHAT_BUDGET} questions to the "
            "rules console. Reload the page to start a new session, or read the "
            "current bands above - they are the whole rule set."
        )
    meter = _chat_meter()
    now = time.time()
    if now - meter["window_started"] > _WINDOW:
        meter["window_started"] = now
        meter["calls"] = 0
    if meter["calls"] >= HOURLY_CHAT_BUDGET:
        return False, (
            "The console has answered a lot of questions this hour and is resting "
            "so the demo stays up for everyone. Try again shortly - the rules, the "
            "recommendation and the feedback form all keep working."
        )
    return True, ""


def note_chat_call() -> None:
    st.session_state["chat_calls_used"] = st.session_state.get("chat_calls_used", 0) + 1
    _chat_meter()["calls"] += 1


def clean_text(value: str | None, limit: int = 200) -> str | None:
    """Visitor free text on its way to the database and to a model prompt."""
    if not value:
        return None
    flat = " ".join(str(value).split())
    flat = "".join(ch for ch in flat if ch.isprintable())
    flat = flat[:limit].strip()
    return flat or None
