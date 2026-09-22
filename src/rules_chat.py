"""Rung 3's console: the agent says what the rules are and advises how to change them.

Approach A from the design: the conversation is free, production is not. When
the agent suggests a change it ends its reply with one fenced JSON block holding
the bands. extract_bands() runs that block through rules_store.normalise -- the
same gate the learner passes -- and the app shows before/after with an Apply
button. Nothing here writes to Supabase, and nothing here imports Streamlit, so
every function is testable with a dict and a stub client.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import rules_store
from rules_store import Band, RuleSet, normalise

DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


@dataclass
class ChatTurn:
    text: str
    bands: tuple[Band, ...] | None = None
    rationale: str = ""
    error: str = ""


def chat_key(rules: RuleSet, record_id: str) -> str:
    """The session key the console's conversation lives under.

    It moves with the rules version and with the customer: Streamlit renders
    every tab on every run, so the console can be born on the table's top
    customer before rung 2 has scored anyone, and a key that only tracked the
    version would keep that first opening message forever.
    """
    return f"rules_chat_v{rules.version}_{record_id}"


def llm_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def describe_bands(rules: RuleSet) -> str:
    return "\n".join(f"- {b.name}: from {b.min:.0%} -> {b.action}" for b in rules.bands)


def opening_message(rules: RuleSet, record: dict,
                    probability_key: str = "probability") -> str:
    """Built locally, no model call -- so the console is never empty."""
    probability = float(record.get(probability_key) or 0.0)
    band = rules.band_for(probability)
    return (
        f"Rules v{rules.version}, highest floor first:\n{describe_bands(rules)}\n\n"
        f"This customer is at {probability:.0%}, which lands in '{band.name}' "
        f"-> {band.action}.\n\n"
        "Ask me why, or tell me what you would rather see happen, and I will "
        "propose a change to the bands. Nothing changes until you apply it."
    )


def system_prompt(rules: RuleSet, digest_rows: list[dict], record: dict) -> str:
    return (
        "You are the console of a recommendation agent. The agent turns the "
        f"probability of {rules_store.event_phrase()} into an action using probability bands, highest floor first:\n"
        f"{json.dumps(rules.as_rows(), ensure_ascii=False)}\n"
        f"This is rules version {rules.version}.\n\n"
        "The customer under discussion:\n"
        f"{json.dumps(record, default=str, ensure_ascii=False)}\n\n"
        "What people said about earlier recommendations, folded per band:\n"
        f"{json.dumps(digest_rows, default=str, ensure_ascii=False)}\n\n"
        "Answer in the language the user writes in. Be concrete and short. State "
        "the current rules when asked. When you recommend changing them, end your "
        "reply with exactly one fenced json block of the form "
        '{"bands": [{"name": ..., "min": ..., "action": ...}, ...], '
        '"rationale": "..."} keeping the same band names, floors between 0 and 1 '
        "with the lowest at 0. If no change is justified, say so and do not emit "
        "a block. Never invent feedback that is not listed above."
    )


def extract_bands(text: str, current: RuleSet | None = None
                  ) -> tuple[tuple[Band, ...] | None, str]:
    """The last fenced json block in the reply, validated. (None, '') otherwise."""
    blocks = BLOCK.findall(text)
    if not blocks:
        return None, ""
    try:
        payload = json.loads(blocks[-1])
        bands = normalise(payload["bands"])
    except (ValueError, KeyError, TypeError):
        return None, ""
    if current is not None and {b.name for b in bands} != {b.name for b in current.bands}:
        return None, ""
    return bands, str(payload.get("rationale", "")).strip()


def strip_block(text: str) -> str:
    return BLOCK.sub("", text).strip()


def reply(history: list[dict], system: str, current: RuleSet | None = None,
          client=None, model: str = DEFAULT_MODEL) -> ChatTurn | None:
    """One assistant turn. `history` is [{"role", "content"}, ...]. None when it cannot run.

    `client` is injectable for tests; in the app it is built from GROQ_API_KEY
    exactly the way learner.propose_llm builds its own.
    """
    if client is None:
        if not llm_available():
            return None
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=model, max_tokens=900, reasoning_effort="low",
            messages=[{"role": "system", "content": system}, *history],
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 -- on stage a failure must not stop the demo
        return ChatTurn("", error=str(exc))
    if not text:
        return ChatTurn("", error="the model returned an empty reply")
    bands, rationale = extract_bands(text, current)
    return ChatTurn(strip_block(text), bands, rationale)
