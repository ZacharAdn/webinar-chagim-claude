"""Step 6 -- the form on share.streamlit.io, filled by the agent.

Streamlit Community Cloud has no API for creating an app, so this is the one
step a script cannot finish. It does everything a script can: refuse a private
repo, read the secrets off disk, and print the exact five values to type. The
agent drives the browser and then calls record() with the URL it landed on.
"""

from __future__ import annotations

from ..context import LadderConfig, LadderError, read_result, write_result

MAIN_FILE = "src/app.py"
BRANCH = "main"


def form_values(cfg: LadderConfig) -> dict:
    github = read_result(cfg, "github")
    if not github:
        raise LadderError("step 5 (github) has not run -- no repo to deploy from")
    visibility = github["data"]["visibility"]
    if visibility != "public":
        raise LadderError(
            f"repo is {visibility}; Streamlit's GitHub grant sees public repos only"
        )
    secrets_file = cfg.root / ".streamlit" / "secrets.toml"
    secrets = secrets_file.read_text(encoding="utf-8") if secrets_file.exists() else ""
    return {
        "repository": github["data"]["slug"],
        "branch": BRANCH,
        "main_file_path": MAIN_FILE,
        "app_url": f"https://{cfg.name}.streamlit.app",
        "secrets": secrets,
    }


def render_form(cfg: LadderConfig) -> str:
    values = form_values(cfg)
    body = "\n".join(
        f"  {key:<16} {value}"
        for key, value in values.items()
        if key != "secrets"
    )
    secrets = values["secrets"] or "(none -- run `python ladder.py supabase` first)"
    # The terminal is usually on a shared screen. Print the shape, never the values:
    # the browser step reads .streamlit/secrets.toml itself and pastes it.
    return (
        "Step 6 needs a browser. On https://share.streamlit.io, 'Create app' -> "
        "'Deploy a public app from GitHub', fill:\n"
        f"{body}\n"
        "  advanced        Python 3.14, and paste .streamlit/secrets.toml into Secrets:\n"
        + "\n".join(f"    {mask(line)}" for line in secrets.splitlines())
        + f"\n\nThen: python ladder.py deploy --record https://{cfg.name}.streamlit.app"
    )


def mask(line: str) -> str:
    """`KEY = "abcdef..."` -> `KEY = "abc…(40)"`. Section headers pass through."""
    if "=" not in line:
        return line
    key, _, value = line.partition("=")
    value = value.strip().strip('"')
    if not value or key.strip() == "SUPABASE_URL":
        return line
    return f'{key.rstrip()} = "{value[:3]}…({len(value)})"'


def run(cfg: LadderConfig) -> dict:
    print(render_form(cfg))
    return write_result(
        cfg, "deploy", "PENDING", "waiting for the browser step",
        {"form": {k: v for k, v in form_values(cfg).items() if k != "secrets"}},
    )


def record(cfg: LadderConfig, url: str) -> dict:
    if not url.startswith("https://") or ".streamlit.app" not in url:
        raise LadderError(f"'{url}' is not a streamlit.app URL")
    return write_result(cfg, "deploy", "PASS", url, {"url": url.rstrip("/")})
