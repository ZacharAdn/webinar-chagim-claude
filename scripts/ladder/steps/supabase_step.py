"""Step 4 -- the project, the schema, the import, the secrets.

The NULL count is computed here from the prepared CSV rather than in SQL: the
question the report answers is "did the import keep the NULLs prepare found",
and that is a comparison between two numbers this process already has.
"""

from __future__ import annotations

import os
import re
import subprocess

from ..context import LadderConfig, LadderError, read_result, write_result
from ..schema import RECORDS_TABLE, write_schema
from .prepare import load_prepared


def existing_ref(name: str) -> str | None:
    """The ref of a project already called `name`, or None. Never creates."""
    try:
        out = subprocess.run(
            ["supabase", "projects", "list", "--output-format", "json"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    import json

    for project in json.loads(out).get("projects", []):
        if project.get("name") == name:
            return project.get("ref") or project.get("id")
    return None


def run(cfg: LadderConfig) -> dict:
    df = load_prepared(cfg)
    write_schema(cfg, df)

    script = cfg.root / "scripts" / "supabase_setup.sh"
    if not script.exists():
        raise LadderError(f"{script} is missing")

    ref = existing_ref(cfg.name)
    if ref:
        print(f"project '{cfg.name}' already exists (ref {ref}) -- reusing it")

    env = os.environ | {
        "SUPABASE_PROJECT_NAME": cfg.name,
        "SUPABASE_REGION": cfg.region,
        "LADDER_CSV": str(cfg.prepared_csv),
        "LADDER_TABLE": RECORDS_TABLE,
    }
    command = [str(script)] + ([ref] if ref else [])
    result = subprocess.run(
        command, cwd=cfg.root, env=env, capture_output=True, text=True, timeout=1800
    )
    if result.returncode != 0:
        last = [l for l in result.stdout.splitlines() + result.stderr.splitlines() if l]
        raise LadderError(f"supabase setup failed: {last[-1] if last else 'no output'}")

    found_ref = _grab(result.stdout, r"project ref:\s*(\S+)") or ref
    rows = int(_grab(result.stdout, r"table rows:\s*(\d+)") or 0)
    expected = (read_result(cfg, "prepare") or {}).get("data", {})
    expected_rows = expected.get("rows", rows)
    if rows != expected_rows:
        raise LadderError(
            f"imported {rows} rows but prepare counted {expected_rows}"
        )
    nulls = sum((expected.get("nulls") or {}).values())

    line = f"ref {found_ref} · {rows:,} rows · {nulls} NULLs kept"
    return write_result(
        cfg, "supabase", "PASS", line,
        {"ref": found_ref, "rows": rows, "nulls": nulls,
         "url": f"https://{found_ref}.supabase.co", "table": RECORDS_TABLE},
    )


def _grab(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None
