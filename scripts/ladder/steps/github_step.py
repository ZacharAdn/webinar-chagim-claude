"""Step 5 -- the repo, at the visibility the deploy step needs.

Streamlit Community Cloud's GitHub grant covers public repositories only, which
is why github_visibility defaults to public in ladder.toml and why step 6 checks
this step's answer before it opens a browser.
"""

from __future__ import annotations

import subprocess

from ..context import LadderConfig, LadderError, write_result


def _run(command: list[str], cwd=None) -> str:
    return subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=True, timeout=300
    ).stdout.strip()


def _git(*args: str, cwd=None) -> str:
    return _run(["git", *args], cwd=cwd)


def _gh(*args: str, cwd=None) -> str:
    return _run(["gh", *args], cwd=cwd)


def owner() -> str:
    return _gh("api", "user", "-q", ".login")


def repo_exists(slug: str) -> bool:
    try:
        _gh("repo", "view", slug, "--json", "name")
        return True
    except subprocess.CalledProcessError:
        return False


def run(cfg: LadderConfig) -> dict:
    visibility = cfg.github_visibility.lower()
    if visibility not in ("public", "private"):
        raise LadderError(f"github_visibility must be public or private, got '{visibility}'")

    try:
        slug = f"{owner()}/{cfg.name}"
        _git("add", "-A", cwd=cfg.root)
        try:
            _git("commit", "-m", f"ladder: {cfg.name} built from ladder.toml", cwd=cfg.root)
        except subprocess.CalledProcessError:
            pass  # nothing to commit is not a failure
        if not repo_exists(slug):
            _gh(
                "repo", "create", slug, f"--{visibility}",
                "--source", str(cfg.root), "--remote", "origin", "--push",
                cwd=cfg.root,
            )
        else:
            try:
                _git("remote", "add", "origin",
                     f"https://github.com/{slug}.git", cwd=cfg.root)
            except subprocess.CalledProcessError:
                pass  # the remote is already there
            _git("push", "-u", "origin", "main", cwd=cfg.root)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        raise LadderError(
            f"git/gh failed: {detail[-1] if detail else exc}"
        ) from exc

    url = f"https://github.com/{slug}"
    return write_result(
        cfg, "github", "PASS", f"{slug} ({visibility})",
        {"slug": slug, "url": url, "visibility": visibility},
    )
