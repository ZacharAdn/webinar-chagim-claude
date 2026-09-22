#!/usr/bin/env python3
"""Kept so `python scripts/prepare_data.py` does what it did in webinar-aug."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ladder.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["prepare", "--root", str(Path(__file__).resolve().parents[1])]))
