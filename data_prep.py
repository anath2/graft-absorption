"""Fetches GRAFT's code, the only external input.

GRAFT's RL code (Ward et al. 2026, MIT) is cloned, pinned, into
``data/raw/GRAFT``. Its toy environments generate their own prompts from the
small jsonl files it ships (``rl/data/``), so there is no separate dataset to
download. ``gpu.py`` mounts ``rl/`` into the Modal image.

    uv run python data_prep.py
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"

GRAFT_URL = "https://github.com/jnward/GRAFT.git"
# The commit our results use (2026-10-13, "add arXiv link to README").
GRAFT_COMMIT = "b0767d1c40e5a03d78fc5ff4b6abc92829168386"
GRAFT_DIR = RAW_DIR / "GRAFT"
GRAFT_RL_DIR = GRAFT_DIR / "rl"


def fetch_graft() -> Path:
    """Clone GRAFT at ``GRAFT_COMMIT`` (idempotent) and return its ``rl/`` path."""
    if not GRAFT_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", GRAFT_URL, str(GRAFT_DIR)], check=True)
    head = subprocess.run(
        ["git", "-C", str(GRAFT_DIR), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != GRAFT_COMMIT:
        subprocess.run(["git", "-C", str(GRAFT_DIR), "checkout", "-q", GRAFT_COMMIT], check=True)
    return GRAFT_RL_DIR


if __name__ == "__main__":
    print(fetch_graft())
