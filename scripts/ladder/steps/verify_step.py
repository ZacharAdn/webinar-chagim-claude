"""Step 7 -- prove it from outside, the way a stranger would.

Nothing in this file imports the app. It talks to PostgREST over HTTPS with the
publishable key and to the deployed URL with a plain opener, because the claim
being tested is "someone who was not in the room can read this", and importing
the app would quietly test the wrong thing.

Three facts get recorded: how many rows the live table holds, that a write lands
and comes back, and that the deployed URL answers. A bare 303 from streamlit.app
is the app waking up, not a failure -- classify() is where that is decided.
"""

from __future__ import annotations

import json
import time
import tomllib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.cookiejar import CookieJar

from ..context import LadderConfig, LadderError, read_result, write_result
from ..schema import PREDICTIONS_TABLE, RECORDS_TABLE

TIMEOUT = 30


def parse_count(content_range: str | None) -> int:
    """PostgREST answers '0-24/7043'. The number after the slash is the truth."""
    if not content_range or "/" not in content_range:
        return 0
    total = content_range.rsplit("/", 1)[1].strip()
    return int(total) if total.isdigit() else 0


def lower_keys(headers) -> dict:
    """HTTP header names are case-insensitive; a plain dict() of them is not.

    urllib hands back 'Content-Range' and a lookup for 'content-range' misses it
    silently -- which reads as a table with zero rows, not as an error.
    """
    return {str(key).lower(): value for key, value in dict(headers).items()}


def classify(status: int) -> str:
    if status == 200:
        return "live"
    if status in (301, 302, 303, 307, 308, 503):
        return "not-yet"
    return "dead"


def secrets(cfg: LadderConfig) -> tuple[str, str]:
    path = cfg.root / ".streamlit" / "secrets.toml"
    if not path.exists():
        raise LadderError(f"{path} is missing -- run: python ladder.py supabase")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    block = raw.get("connections", {}).get("supabase", {})
    url, key = block.get("SUPABASE_URL", ""), block.get("SUPABASE_KEY", "")
    if not url or not key:
        raise LadderError(f"{path} has no SUPABASE_URL/SUPABASE_KEY")
    if not key.startswith("sb_publishable_"):
        raise LadderError("SUPABASE_KEY is not a publishable key -- refusing to use it")
    return url.rstrip("/"), key


def rest(url: str, key: str, path: str, method: str = "GET",
         body: list | dict | None = None,
         headers: dict | None = None) -> tuple[int, dict, object]:
    request = urllib.request.Request(
        f"{url}/rest/v1/{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = response.read().decode() or "null"
            return response.status, lower_keys(response.headers), json.loads(payload)
    except urllib.error.HTTPError as exc:
        raise LadderError(
            f"{method} {path} returned {exc.code}: {exc.read().decode()[:120]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise LadderError(f"{method} {path} could not connect: {exc.reason}") from exc


def app_status(app_url: str) -> str:
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar())
    )
    try:
        with opener.open(app_url, timeout=TIMEOUT) as response:
            return classify(response.status)
    except urllib.error.HTTPError as exc:
        return classify(exc.code)
    except urllib.error.URLError:
        return "not-yet"


def run(cfg: LadderConfig, attempts: int = 20, pause: float = 15.0) -> dict:
    url, key = secrets(cfg)

    deploy = read_result(cfg, "deploy")
    app_url = (deploy or {}).get("data", {}).get("url", "")
    state = "skipped"
    if app_url:
        for attempt in range(1, attempts + 1):
            state = app_status(app_url)
            if state == "live":
                break
            print(f"  {app_url} is {state} ({attempt}/{attempts})")
            time.sleep(pause)
        if state != "live":
            raise LadderError(f"{app_url} never answered 200 (last: {state})")

    _, headers, _ = rest(
        url, key, f"{RECORDS_TABLE}?select=*&limit=1",
        headers={"Prefer": "count=exact", "Range": "0-0"},
    )
    rows = parse_count(headers.get("content-range"))
    if rows == 0:
        raise LadderError(f"the live {RECORDS_TABLE} table reads 0 rows")

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _, _, inserted = rest(
        url, key, PREDICTIONS_TABLE, method="POST",
        body=[{
            "record_id": "ladder-verify",
            "probability": 0.5,
            "recommended_action": "verify step, not a real recommendation",
            "model_version": "ladder-verify",
            "created_at": stamp,
        }],
        headers={"Prefer": "return=representation"},
    )
    new_id = inserted[0]["id"]
    _, _, back = rest(url, key, f"{PREDICTIONS_TABLE}?id=eq.{new_id}&select=*")
    if not back or back[0]["record_id"] != "ladder-verify":
        raise LadderError(f"wrote prediction id {new_id} but could not read it back")

    line = f"live read {rows:,} · write id {new_id} at {stamp[11:19]} · read back OK"
    if app_url:
        line += " · app 200"
    return write_result(
        cfg, "verify", "PASS", line,
        {"rows": rows, "prediction_id": new_id, "at": stamp,
         "app_url": app_url, "app_status": state},
    )
