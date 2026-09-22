"""Rung 3 -- the agent that recommends an action.

The house rule from the script: a rules layer is the baseline of a
recommendation exactly the way a dummy model is the baseline of a prediction.
The language model is only worth its latency if it beats the rules, and on
stage that is an open question, not a claim.

The rules are probability bands. On 9.9 they were read out of ladder.toml;
since the loop was added they are the active row of the `rules` table, with
ladder.toml as version 1 and the offline fallback (rules_store.py). The learner
(learner.py) writes the next version; this module only ever reads.

Bands.recommend always works, offline, with no key.
recommend_llm is the optional second opinion, served by gpt-oss-120b on Groq
over its OpenAI-compatible endpoint; without GROQ_API_KEY it returns None and
the app says so instead of pretending. It is handed the last few feedback rows
so the second opinion is the first thing in the system that has read them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import rules_store

DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Recommendation:
    action: str
    reason: str
    source: str            # "rules" or "groq"
    unknowns: str = ""


@dataclass(frozen=True)
class Band:
    name: str
    minimum: float
    action: str


@dataclass(frozen=True)
class Bands:
    bands: tuple[Band, ...]
    version: int = 1
    source: str = "toml"

    @classmethod
    def load(cls, root: Path | None = None, conn=None) -> "Bands":
        """The active rule set: Supabase when a connection is given, else ladder.toml."""
        rules = rules_store.load_active(conn, root)
        return cls.from_rules(rules)

    @classmethod
    def from_rules(cls, rules: rules_store.RuleSet) -> "Bands":
        return cls(
            tuple(Band(b.name, b.min, b.action) for b in rules.bands),
            rules.version, rules.source,
        )

    def recommend(self, record: dict,
                  probability_key: str = "probability") -> Recommendation:
        probability = float(record.get(probability_key) or 0.0)
        for band in self.bands:
            if probability >= band.minimum:
                return Recommendation(
                    band.action,
                    f"Risk {probability:.0%} is in the '{band.name}' band, "
                    f"which starts at {band.minimum:.0%} (rules v{self.version}).",
                    "rules",
                )
        last = self.bands[-1]
        return Recommendation(
            last.action, f"Risk {probability:.0%} is below every band.", "rules"
        )


def llm_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def recommend_llm(record: dict, rules: Recommendation,
                  model: str = DEFAULT_MODEL,
                  feedback: list[dict] | None = None) -> Recommendation | None:
    """Ask gpt-oss-120b for a second opinion. Returns None when it cannot run.

    Groq speaks the OpenAI wire format, so the OpenAI SDK pointed at
    GROQ_BASE_URL is the whole integration. reasoning_effort is a gpt-oss knob
    and json_object mode is what keeps the reply parseable without a retry.
    """
    if not llm_available():
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL
        )
        prompt = (
            f"You are advising a team acting on {rules_store.event_phrase()}. "
            "Here is one record:\n"
            f"{json.dumps(record, default=str, ensure_ascii=False)}\n\n"
            f"A rules engine recommends: {rules.action} ({rules.reason})\n\n"
            + (
                "What people said about earlier recommendations on similar records "
                "(verdict, what actually happened, what would have been better):\n"
                f"{json.dumps(feedback[-8:], default=str, ensure_ascii=False)}\n\n"
                if feedback else ""
            )
            + "Reply as JSON with exactly these keys: action (one concrete step), "
            "reason (one sentence, grounded in the fields above), unknowns "
            "(what this data does not tell you and would change the answer). "
            "Do not invent facts that are not in the record."
        )
        response = client.chat.completions.create(
            model=model,
            max_tokens=500,
            reasoning_effort="low",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            text = text.split("```")[1].removeprefix("json").strip()
        payload = json.loads(text)
        return Recommendation(
            action=str(payload.get("action", "")).strip(),
            reason=str(payload.get("reason", "")).strip(),
            source="groq",
            unknowns=str(payload.get("unknowns", "")).strip(),
        )
    except Exception:  # noqa: BLE001 -- on stage a failure must not stop the demo
        return None
