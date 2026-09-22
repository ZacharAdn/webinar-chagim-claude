"""Shared plumbing for every ladder step: config, paths, results.

`ladder.toml` is the only file that differs between datasets, so it is the only
file this module reads. Everything downstream takes a LadderConfig and nothing
else -- that is what makes two runs on two datasets comparable line by line.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class LadderError(Exception):
    """A step failure. Printed as one line by the CLI, never as a traceback."""


@dataclass(frozen=True)
class Band:
    """One probability band from [bands]: the floor, and what to do above it."""

    name: str
    minimum: float
    action: str


@dataclass(frozen=True)
class LadderConfig:
    root: Path
    raw: str
    id_column: str
    target_column: str
    positive_label: str
    name: str
    region: str
    github_visibility: str
    bands: tuple[Band, ...]
    seed: int
    test_size: float
    categorical_int_max_unique: int
    estimator: str = "logreg"

    @classmethod
    def load(cls, root: Path) -> "LadderConfig":
        path = Path(root) / "ladder.toml"
        if not path.exists():
            raise LadderError(f"ladder.toml not found at {path}")
        try:
            with path.open("rb") as fh:
                raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise LadderError(f"ladder.toml is not valid TOML: {exc}") from exc

        try:
            dataset = raw["dataset"]
            project = raw["project"]
            model = raw.get("model", {})
            bands = tuple(
                Band(key, float(value["min"]), str(value["action"]))
                for key, value in sorted(
                    raw["bands"].items(), key=lambda kv: -float(kv[1]["min"])
                )
            )
            return cls(
                root=Path(root),
                raw=str(dataset["raw"]),
                id_column=str(dataset.get("id_column", "") or ""),
                target_column=str(dataset["target_column"]),
                positive_label=str(dataset["positive_label"]),
                name=str(project["name"]),
                region=str(project.get("region", "eu-central-1")),
                github_visibility=str(project.get("github_visibility", "public")),
                bands=bands,
                seed=int(model.get("seed", 42)),
                test_size=float(model.get("test_size", 0.2)),
                categorical_int_max_unique=int(
                    model.get("categorical_int_max_unique", 10)
                ),
                estimator=str(model.get("estimator", "logreg")),
            )
        except KeyError as exc:
            raise LadderError(f"ladder.toml is missing {exc}") from exc

    @property
    def raw_csv(self) -> Path:
        return self.root / self.raw

    @property
    def prepared_csv(self) -> Path:
        return self.root / "data" / f"{self.name}.csv"

    @property
    def results_dir(self) -> Path:
        return self.root / ".ladder"


def band_for(cfg: LadderConfig, probability: float) -> Band:
    """The first band whose floor the probability clears. Bands run high to low."""
    for band in cfg.bands:
        if probability >= band.minimum:
            return band
    return cfg.bands[-1]


def write_result(cfg: LadderConfig, step: str, status: str, line: str,
                 data: dict) -> dict:
    """One JSON per step. REPORT.md is assembled from these and nothing else."""
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "status": status,
        "line": line,
        "data": data,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (cfg.results_dir / f"{step}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return payload


def read_result(cfg: LadderConfig, step: str) -> dict | None:
    path = cfg.results_dir / f"{step}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
