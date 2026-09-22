"""The report -- one file, one table, eight rows.

The whole review is: scan the middle column, read the numbers. A FAIL stops the
reading, because everything below it is SKIPPED by definition and pretending
otherwise would be the one dishonest thing this file could do.
"""

from __future__ import annotations

from datetime import datetime

from .context import LadderConfig, read_result, write_result

STEPS = (
    ("prepare", "1 prepare"),
    ("model", "2 model"),
    ("app", "3 app"),
    ("supabase", "4 supabase"),
    ("github", "5 github"),
    ("deploy", "6 deploy"),
    ("verify", "7 verify"),
    ("feedback", "8 feedback"),
)


def render(cfg: LadderConfig) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# {cfg.name} — ladder report, {stamp}",
        "",
        "| Rung | Status | Measured |",
        "|---|---|---|",
    ]
    stopped = False
    for key, label in STEPS:
        result = read_result(cfg, key)
        if stopped:
            status, detail = "SKIPPED", ""
        elif result is None:
            status, detail = "NOT RUN", ""
        else:
            status, detail = result["status"], result["line"]
            if status in ("FAIL", "PENDING"):
                # Nothing after a stop can be true yet -- a PENDING deploy means
                # verify has nothing to verify, and saying otherwise would be the
                # one dishonest thing this file could do.
                stopped = True
        lines.append(f"| {label} | {status} | {detail} |")

    lines += [
        "",
        "```",
        "Rerun one step:   python ladder.py model",
        "Rerun from step:  python ladder.py all --from supabase",
        "Open the app:     streamlit run src/app.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def run(cfg: LadderConfig) -> dict:
    text = render(cfg)
    (cfg.root / "REPORT.md").write_text(text, encoding="utf-8")
    passed = sum(
        1 for key, _ in STEPS
        if (read_result(cfg, key) or {}).get("status") == "PASS"
    )
    return write_result(
        cfg, "report", "PASS", f"{passed} of {len(STEPS)} rungs PASS",
        {"passed": passed, "of": len(STEPS)},
    )
