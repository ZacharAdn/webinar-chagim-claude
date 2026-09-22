"""The gates between the rungs.

`ladder.py all` is not a pipeline you launch and watch. It stops before every
rung that costs a decision, says what it is about to do, and waits for a person:

    rung 1 -> the app opens in a browser, and nothing else happens until you say so
    rung 2 -> which model, out of the ones that fit this table
    rung 3 -> rules only, or rules plus a language model (which needs a key)
    rung 4 -> the cloud steps, which create real accounts and cost real money

`--auto` skips every gate and is the old behaviour exactly: use it in CI, in a
rehearsal, and as the fallback when a demo has to keep moving.

A gate never asks when stdin is not a terminal -- a piped or scripted run
answers the default and moves on, so nothing here can hang a background job.
"""

from __future__ import annotations

import getpass
import os
import re
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

from .context import LadderConfig

MODELS = {
    "1": ("logreg", "Logistic regression — one coefficient per column, and you "
                    "can read the direction of each out loud"),
    "2": ("tree", "Decision tree — the path to a decision is a chain of "
                  "if/else, at the price of overfitting"),
}


def interactive() -> bool:
    """A gate only asks a person. Anything else takes the default."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _ask(prompt: str, default: str = "") -> str:
    if not interactive():
        return default
    try:
        answer = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return answer or default


def _confirm(question: str) -> bool:
    return _ask(f"{question} [Y/n] ", "y").lower() not in ("n", "no")


def _free_port(preferred: int = 8501) -> int:
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# --------------------------------------------------------------------------
# Rung 1 -- the table is clean, so look at it before anything is modelled
# --------------------------------------------------------------------------
def open_app(cfg: LadderConfig) -> subprocess.Popen | None:
    """Start Streamlit on a free port and open a browser at it."""
    app = cfg.root / "src" / "app.py"
    if not app.exists():
        return None
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(app),
         "--server.port", str(port), "--server.headless", "true"],
        cwd=cfg.root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://localhost:{port}"
    print(f"    the app is live at {url} (pid {process.pid})")
    if interactive():
        webbrowser.open(url)
    return process


def after_prepare(cfg: LadderConfig) -> bool:
    print("\n— rung 1 is on the screen: the table, the NULLs, the base rate.")
    return _confirm("   Train a model on it now?")


# --------------------------------------------------------------------------
# Rung 2 -- which model, and it is written back into ladder.toml
# --------------------------------------------------------------------------
def choose_model(cfg: LadderConfig) -> str:
    print("\n— rung 2. Two model families fit a yes/no column on a table this size:")
    for key, (_, description) in MODELS.items():
        print(f"    {key}. {description}")
    current = cfg.estimator
    answer = _ask(f"   Which one? [1/2, Enter keeps {current}] ", "")
    chosen = MODELS.get(answer, (current, ""))[0]
    if chosen != current:
        set_estimator(cfg, chosen)
        print(f"    ladder.toml: estimator = \"{chosen}\"")
    return chosen


def set_estimator(cfg: LadderConfig, estimator: str) -> None:
    """ladder.toml is the whole configuration surface -- so write it there."""
    path = cfg.root / "ladder.toml"
    text = path.read_text(encoding="utf-8")
    if re.search(r"^estimator\s*=", text, re.M):
        text = re.sub(r"^estimator\s*=.*$", f'estimator = "{estimator}"', text, flags=re.M)
    else:
        text = re.sub(r"^\[model\]$", f'[model]\nestimator = "{estimator}"', text,
                      count=1, flags=re.M)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# Rung 3 -- the rules are the baseline; a model is the option, and needs a key
# --------------------------------------------------------------------------
def choose_recommender(cfg: LadderConfig) -> str:
    print("\n— rung 3. A probability is not an action. Two ways to turn it into one:")
    print("    1. The rules in ladder.toml — the bands. No key, no latency, "
          "and it is the baseline any agent has to beat")
    print("    2. The rules plus Claude as a second opinion — needs "
          "ANTHROPIC_API_KEY, and the app shows both side by side")
    if _ask("   Which one? [1/2, Enter keeps the rules] ", "1") != "2":
        return "rules"
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("    ANTHROPIC_API_KEY is already in the environment.")
        return "claude"
    key = getpass.getpass("   Paste ANTHROPIC_API_KEY (hidden, Enter to skip): ").strip()
    if not key:
        print("    no key — staying on the rules.")
        return "rules"
    store_secret(cfg, "ANTHROPIC_API_KEY", key)
    os.environ["ANTHROPIC_API_KEY"] = key
    print("    written to .streamlit/secrets.toml (gitignored) and this session.")
    return "claude"


def store_secret(cfg: LadderConfig, name: str, value: str) -> None:
    path = cfg.root / ".streamlit" / "secrets.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    line = f'{name} = "{value}"'
    if re.search(rf"^{name}\s*=", text, re.M):
        text = re.sub(rf"^{name}\s*=.*$", line, text, flags=re.M)
    else:
        text = (text.rstrip("\n") + "\n" + line + "\n").lstrip("\n")
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


# --------------------------------------------------------------------------
# Rung 4 -- everything past here creates accounts and costs money
# --------------------------------------------------------------------------
def before_cloud(cfg: LadderConfig) -> bool:
    print(f"\n— rung 4 creates real things: a Supabase project ({cfg.region}), a "
          f"{cfg.github_visibility} GitHub repo, and a deploy at "
          f"{cfg.name}.streamlit.app.")
    print("    Local work is done and REPORT.md can be written without any of it.")
    return _confirm("   Go to production?")
