"""Agent two -- the one that learns.

Agent one (agent.py) recommends. It is deliberately dumb about its own track
record: it reads the active band set and applies it. This module is the other
half. It reads what people said about those recommendations -- right, wrong,
what actually happened, what would have been better -- and rewrites the band
set the recommender will use from now on.

Two learners, same contract, so the loop never depends on a key:

  propose_rules   deterministic. Counts the verdicts per band. A band whose
                  recommendation was called wrong often enough gets the action
                  people said would have been better. When nobody named one,
                  the outcome decides: they left, so the band takes the stronger
                  action of the band above; they stayed, so it takes the softer
                  action of the band below. Floors never move here -- a moved
                  floor is invisible to the customer on the screen (23.9.2026).
                  This is the one that runs when Groq is not there.
  propose_llm     gpt-oss-120b on Groq reads the same digest and writes the
                  revision as JSON with a rationale. Validated by the same
                  normalise() before it is allowed anywhere near the recommender.

Neither touches the model. That is rung 2's problem and it needs thousands of
outcomes, not a dozen verdicts -- which is the whole point of the slide.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import rules_store
from rules_store import Band, RuleSet, normalise

DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

MIN_VERDICTS_TO_ACT = 3       # fewer than this and a band is left alone
WRONG_SHARE_TO_ACT = 0.5      # half the verdicts on a band say wrong -> act


@dataclass(frozen=True)
class Thresholds:
    """The three numbers the learner acts on. Defaults above; ladder.toml's
    [learner] table overrides them, because a number that decides when the
    rules change belongs in the one configuration file, not in Python."""

    min_verdicts: int = MIN_VERDICTS_TO_ACT
    wrong_share: float = WRONG_SHARE_TO_ACT

    def rule_text(self) -> str:
        """One sentence the screen shows next to the button, so the threshold
        is never a surprise: it says what the learner will and will not do."""
        return (
            f"A band changes only once at least {self.min_verdicts} verdicts on it "
            f"say 'wrong' (at least {self.wrong_share:.0%} of its verdicts). With a "
            "better action named, the band's action becomes that. Without one: they "
            "left, so the band takes the stronger action of the band above; they "
            "stayed, so it takes the softer action of the band below."
        )


def thresholds(root: Path | None = None) -> Thresholds:
    try:
        with (Path(root or rules_store.REPO_ROOT) / "ladder.toml").open("rb") as fh:
            raw = tomllib.load(fh).get("learner", {})
    except Exception:  # noqa: BLE001 - no file, defaults
        raw = {}
    return Thresholds(
        int(raw.get("min_verdicts", MIN_VERDICTS_TO_ACT)),
        float(raw.get("wrong_share", WRONG_SHARE_TO_ACT)),
    )


@dataclass
class BandDigest:
    band: str
    floor: float
    action: str
    total: int = 0
    wrong: int = 0
    stayed: int = 0
    left: int = 0
    better: Counter = field(default_factory=Counter)

    @property
    def wrong_share(self) -> float:
        return self.wrong / self.total if self.total else 0.0

    def as_row(self) -> dict:
        top = self.better.most_common(1)
        return {
            "band": self.band,
            "from": f"{self.floor:.0%}",
            "verdicts": self.total,
            "wrong": self.wrong,
            "stayed": self.stayed,
            "left": self.left,
            "most suggested": top[0][0] if top else "",
        }


@dataclass
class Proposal:
    before: RuleSet
    bands: tuple[Band, ...]
    rationale: str
    source: str                      # 'learner:rules' or 'learner:groq'
    evidence: dict
    changes: list[str]

    @property
    def changed(self) -> bool:
        return bool(self.changes)


def digest(rules: RuleSet, feedback: list[dict]) -> dict[str, BandDigest]:
    """Fold the feedback rows onto the band each one fell in."""
    out = {b.name: BandDigest(b.name, b.min, b.action) for b in rules.bands}
    for row in feedback:
        probability = row.get("probability")
        if probability is None:
            continue
        band = rules.band_for(float(probability))
        d = out[band.name]
        d.total += 1
        if row.get("verdict") == "wrong":
            d.wrong += 1
        outcome = str(row.get("actual_outcome") or "").lower()
        if outcome == "stayed":
            d.stayed += 1
        elif outcome == "left":
            d.left += 1
        better = str(row.get("better_action") or "").strip()
        if better:
            d.better[better] += 1
    return out


def _diff(before: RuleSet, after: tuple[Band, ...]) -> list[str]:
    old = {b.name: b for b in before.bands}
    new = {b.name: b for b in after}
    lines = []
    for name, band in new.items():
        prev = old.get(name)
        if prev is None:
            lines.append(f"{name}: new band from {band.min:.0%} -> '{band.action}'")
            continue
        if prev.min != band.min:
            lines.append(f"{name}: floor {prev.min:.0%} -> {band.min:.0%}")
        if prev.action != band.action:
            lines.append(f"{name}: action '{prev.action}' -> '{band.action}'")
    for name in old:
        if name not in new:
            lines.append(f"{name}: removed")
    return lines


def propose_rules(rules: RuleSet, feedback: list[dict]) -> Proposal:
    """The deterministic learner. Reads the digest, moves what the evidence moves."""
    digested = digest(rules, feedback)
    t = thresholds()
    floors = {b.name: b.min for b in rules.bands}
    actions = {b.name: b.action for b in rules.bands}
    notes: list[str] = []
    crossed = 0
    for position, band in enumerate(rules.bands):       # highest floor first
        d = digested[band.name]
        if d.total < t.min_verdicts or d.wrong_share < t.wrong_share:
            continue
        crossed += 1
        top = d.better.most_common(1)
        if top and top[0][0] != band.action:
            actions[band.name] = top[0][0]
            notes.append(
                f"{band.name}: {d.wrong} of {d.total} verdicts said wrong and "
                f"{top[0][1]} of them suggested '{actions[band.name]}'."
            )
        elif d.left > d.stayed and position > 0:
            # They left after this call: too soft. Take the next band's action.
            above = rules.bands[position - 1]
            actions[band.name] = above.action
            notes.append(
                f"{band.name}: {d.left} of {d.total} customers left after "
                f"'{band.action}', so it now gets '{above.name}''s action: "
                f"'{above.action}'."
            )
        elif d.stayed > d.left and position < len(rules.bands) - 1:
            # They stayed anyway: too much. Take the softer action below.
            below = rules.bands[position + 1]
            actions[band.name] = below.action
            notes.append(
                f"{band.name}: {d.stayed} of {d.total} customers stayed after "
                f"'{band.action}', so it now gets '{below.name}''s action: "
                f"'{below.action}'."
            )
    revised = [Band(b.name, floors[b.name], actions[b.name]) for b in rules.bands]
    bands = normalise([b.__dict__ for b in revised])
    changes = _diff(rules, bands)
    total = sum(d.total for d in digested.values())
    rationale = " ".join(notes) if notes else (
        f"{total} verdicts read; no band crossed the threshold of "
        f"{t.min_verdicts} verdicts with {t.wrong_share:.0%} or more wrong."
        if not crossed else
        f"{total} verdicts read; {crossed} band(s) crossed the threshold but the "
        "verdicts point nowhere: no better action, and either stayed and left are "
        "in balance or there is no neighbouring band to borrow an action from."
    )
    return Proposal(
        rules, bands, rationale, "learner:rules",
        {"bands": [d.as_row() for d in digested.values()]}, changes,
    )


def llm_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def propose_llm(rules: RuleSet, feedback: list[dict],
                model: str = DEFAULT_MODEL) -> Proposal | None:
    """The same job, done by gpt-oss-120b. None when it cannot run or answers badly."""
    if not llm_available():
        return None
    digested = digest(rules, feedback)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL)
        prompt = (
            "You maintain the rules an agent uses to turn the probability of "
            f"{rules_store.event_phrase()} into an action. Current bands, highest floor first:\n"
            f"{json.dumps(rules.as_rows(), ensure_ascii=False)}\n\n"
            "What people said about the recommendations each band produced "
            "(verdicts, what actually happened, what they said would have been better):\n"
            f"{json.dumps([d.as_row() for d in digested.values()], ensure_ascii=False)}\n\n"
            "Recent individual feedback rows:\n"
            f"{json.dumps(feedback[-12:], default=str, ensure_ascii=False)}\n\n"
            "Revise the bands. Rules: keep the same band names; floors between 0 and 1, "
            "the lowest must be 0; change only what the feedback supports; if the "
            "feedback does not justify a change, return the bands unchanged and say so. "
            "Reply as JSON with exactly these keys: bands (a list of objects with "
            "name, min, action) and rationale (two sentences, citing the counts)."
        )
        response = client.chat.completions.create(
            model=model, max_tokens=700, reasoning_effort="low",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1].removeprefix("json").strip()
        payload = json.loads(text)
        bands = normalise(payload["bands"])
        if {b.name for b in bands} != {b.name for b in rules.bands}:
            return None
        return Proposal(
            rules, bands, str(payload.get("rationale", "")).strip(), "learner:groq",
            {"bands": [d.as_row() for d in digested.values()]}, _diff(rules, bands),
        )
    except Exception:  # noqa: BLE001 -- a bad answer means the rules learner runs
        return None


def propose(rules: RuleSet, feedback: list[dict], prefer_llm: bool = True) -> Proposal:
    """The learner the app calls. Rules first; the model only when they find nothing.

    The order used to be the other way round, and on stage the model read
    three verdicts, called them "insufficient consistent data" and shrugged,
    while the deterministic learner had already crossed its threshold. A rule
    that fired is evidence counted; a model's opinion is a second reading of
    the same table. The second reading is worth having only when the first
    found nothing.
    """
    from_rules = propose_rules(rules, feedback)
    if from_rules.changed or not prefer_llm:
        return from_rules
    got = propose_llm(rules, feedback)
    return got if got is not None else from_rules
