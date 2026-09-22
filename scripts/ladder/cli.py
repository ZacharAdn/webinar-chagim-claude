"""The runner. Eight steps, fixed order, one line of output each.

`ladder.py all` climbs one rung at a time and stops at every gate that costs a
decision: it opens the app on rung 1, asks which model on rung 2, asks rules or
agent on rung 3, and asks once more before anything reaches the cloud. The
gates live in gates.py; what they ask is the whole point of the skill.

`--auto` runs the old way, unattended, and never asks -- CI, a rehearsal, and
the fallback when a demo has to keep moving. A non-interactive stdin is treated
the same way, so nothing here can hang a background job.

`--local-only` skips 4 through 7 and still writes a report. Step 6 stops with
the form values printed, because a browser is a person's job; it resumes with
`--from verify`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# The app modules carry st.cache_* decorators; imported outside `streamlit run`
# they log "No runtime found" to stderr, which is noise in a one-line runner.
os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
try:
    import streamlit.logger as _st_logger

    _st_logger.set_log_level("error")
except Exception:  # noqa: BLE001 - quieter output is a nicety, never a failure
    pass

from . import gates, report as report_mod
from .context import LadderConfig, LadderError, write_result
from .steps import (
    app_step,
    deploy_step,
    feedback_step,
    github_step,
    model_step,
    supabase_step,
    verify_step,
)
from .steps.prepare import prepare

ORDER = ("prepare", "model", "app", "supabase", "github", "deploy", "verify",
         "feedback", "report")
CLOUD = ("supabase", "github", "deploy", "verify", "feedback")

REGISTRY = {
    "prepare": prepare,
    "model": model_step.run,
    "app": app_step.run,
    "supabase": supabase_step.run,
    "github": github_step.run,
    "deploy": deploy_step.run,
    "verify": verify_step.run,
    "feedback": feedback_step.run,
    "report": report_mod.run,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ladder.py", description="Build the four rungs from ladder.toml."
    )
    parser.add_argument("step", help="one of: " + ", ".join(ORDER) + ", all")
    parser.add_argument("--root", default=".", help="project directory")
    parser.add_argument("--from", dest="start", help="with 'all': resume at this step")
    parser.add_argument("--local-only", action="store_true",
                        help="skip the four cloud steps")
    parser.add_argument("--auto", action="store_true",
                        help="with 'all': never stop at a gate")
    parser.add_argument("--record", help="with 'deploy': the deployed app URL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.step not in REGISTRY and args.step != "all":
        print(f"unknown step '{args.step}' -- expected one of: "
              + ", ".join((*ORDER, "all")), file=sys.stderr)
        return 2

    root = Path(args.root).resolve()
    gated = args.step == "all" and not args.auto and gates.interactive()
    skip_cloud = args.local_only

    try:
        cfg = LadderConfig.load(root)
        if args.step == "deploy" and args.record:
            print(deploy_step.record(cfg, args.record)["line"])
            return 0

        for name in _plan(args) if args.step == "all" else [args.step]:
            if gated and name in _GATES:
                verdict = _GATES[name](cfg)
                cfg = LadderConfig.load(root)      # a gate may have edited ladder.toml
                if verdict is False:
                    if name not in CLOUD:
                        print(f"stopped before {name}. "
                              f"Resume with: python ladder.py all --from {name}")
                        return 0
                    skip_cloud = True

            if skip_cloud and name in CLOUD:
                reason = "--local-only" if args.local_only else "not going to production"
                write_result(cfg, name, "SKIPPED", reason, {})
                print(f"{name}: SKIPPED ({reason})")
                continue

            payload = REGISTRY[name](cfg)
            print(f"{name}: {payload['status']} — {payload['line']}")
            if payload["status"] == "PENDING":
                report_mod.run(cfg)
                return 0
    except LadderError as exc:
        print(str(exc).replace("\n", " "), file=sys.stderr)
        return 1
    return 0


def _gate_model(cfg: LadderConfig) -> bool:
    """Rung 1 is on the screen before rung 2 is even proposed."""
    gates.open_app(cfg)
    if not gates.after_prepare(cfg):
        return False
    gates.choose_model(cfg)
    return True


def _gate_app(cfg: LadderConfig) -> bool:
    gates.choose_recommender(cfg)
    return True


_GATES = {
    "model": _gate_model,
    "app": _gate_app,
    "supabase": gates.before_cloud,
}


def _plan(args) -> list[str]:
    steps = list(ORDER)
    if args.start:
        if args.start not in steps:
            raise LadderError(f"--from '{args.start}' is not a step")
        steps = steps[steps.index(args.start):]
    return steps
