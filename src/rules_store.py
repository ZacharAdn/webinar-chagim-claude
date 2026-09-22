"""The rules the recommender reads, versioned, so a second agent can rewrite them.

On stage on 9.9 the bands lived in ladder.toml and the live-fix moment was
editing a line by hand. That is fine for one person at a laptop and useless
for a loop: nothing can write to a file inside a Streamlit container and have
it survive. So the active band set lives in a Supabase table now, ladder.toml
is version 1 and the offline fallback, and every revision is a new row with the
evidence it was made from. The recommender reads the active row; the learner
writes the next one. Neither imports the other.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_TABLE = "rules"


@dataclass(frozen=True)
class Band:
    name: str
    min: float
    action: str


@dataclass(frozen=True)
class RuleSet:
    version: int
    bands: tuple[Band, ...]          # highest floor first
    source: str = "toml"
    rationale: str = ""

    def as_rows(self) -> list[dict]:
        return [asdict(b) for b in self.bands]

    def band_for(self, probability: float) -> Band:
        for band in self.bands:
            if probability >= band.min:
                return band
        return self.bands[-1]


def normalise(bands: list[dict]) -> tuple[Band, ...]:
    """Sort highest floor first and reject anything the recommender cannot run on."""
    cleaned = []
    for raw in bands:
        name = str(raw.get("name", "")).strip()
        action = str(raw.get("action", "")).strip()
        minimum = float(raw.get("min", raw.get("minimum", 0.0)))
        if not name or not action:
            raise ValueError(f"band without a name or an action: {raw}")
        if not 0.0 <= minimum <= 1.0:
            raise ValueError(f"band {name} has a floor outside [0, 1]: {minimum}")
        cleaned.append(Band(name, round(minimum, 4), action))
    if not cleaned:
        raise ValueError("no bands")
    cleaned.sort(key=lambda b: -b.min)
    if cleaned[-1].min != 0.0:
        raise ValueError("the lowest band must start at 0 so every probability lands somewhere")
    if len({b.min for b in cleaned}) != len(cleaned):
        raise ValueError("two bands share a floor")
    return tuple(cleaned)


def event_phrase(root: Path | None = None) -> str:
    """What the model predicts, in words a prompt can use: `churn = Yes`.

    The prompts used to say "churn" and "retention team" outright; reading the
    target from ladder.toml keeps them honest on any dataset.
    """
    try:
        with (Path(root or REPO_ROOT) / "ladder.toml").open("rb") as fh:
            ds = tomllib.load(fh)["dataset"]
        return f"{ds['target_column']} = {ds['positive_label']}"
    except Exception:  # noqa: BLE001
        return "the event"


def from_toml(root: Path | None = None) -> RuleSet:
    with (Path(root or REPO_ROOT) / "ladder.toml").open("rb") as fh:
        raw = tomllib.load(fh)["bands"]
    rows = [{"name": k, "min": v["min"], "action": v["action"]} for k, v in raw.items()]
    return RuleSet(1, normalise(rows), "toml", "ladder.toml")


def from_row(row: dict) -> RuleSet:
    bands = row["bands"]
    if isinstance(bands, str):
        bands = json.loads(bands)
    return RuleSet(
        int(row["version"]), normalise(bands),
        str(row.get("source", "supabase")), str(row.get("rationale") or ""),
    )


def load_active(conn, root: Path | None = None) -> RuleSet:
    """The active row from Supabase; ladder.toml when there is no connection."""
    if conn is None:
        return from_toml(root)
    try:
        rows = (
            conn.table(RULES_TABLE).select("*").eq("active", True)
            .order("version", desc=True).limit(1).execute().data
        )
        if rows:
            return from_row(rows[0])
    except Exception:  # noqa: BLE001 -- the fallback is the point
        pass
    return from_toml(root)


def history(conn, limit: int = 10) -> list[dict]:
    if conn is None:
        return []
    try:
        return (
            conn.table(RULES_TABLE)
            .select("version,source,rationale,active,created_at,bands")
            .order("version", desc=True).limit(limit).execute().data
        )
    except Exception:  # noqa: BLE001
        return []


def publish(conn, bands: tuple[Band, ...], source: str, rationale: str,
            evidence: dict | None = None) -> RuleSet:
    """Write the next version and make it the active one. Returns what was written.

    The table itself refuses anonymous writes; a publish goes through the database
    function `publish_rules`, which checks a token before it touches anything. So a
    client that has the connection but not the token - any visitor, or a stale copy
    of this app - cannot replace the rules the room is looking at.
    """
    import access

    current = load_active(conn)
    payload = {
        "p_bands": [asdict(b) for b in bands],
        "p_source": source,
        "p_rationale": rationale,
        "p_evidence": evidence or {},
        "p_token": access.publish_token(),
    }
    result = conn.client.rpc("publish_rules", payload).execute()
    version = int(result.data if isinstance(result.data, int) else current.version + 1)
    return RuleSet(version, bands, source, rationale)
