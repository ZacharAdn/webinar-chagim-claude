#!/usr/bin/env python3
"""The runner's front door. Everything real lives in scripts/ladder/.

    python ladder.py all              one rung at a time, stopping at each gate
    python ladder.py all --auto       the same, unattended: never ask
    python ladder.py all --local-only skip Supabase, GitHub, deploy and verify
    python ladder.py model            rerun one step
    python ladder.py deploy --record https://<name>.streamlit.app
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from ladder.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
