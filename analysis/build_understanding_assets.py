#!/usr/bin/env python3
"""Build all understanding benchmark publish assets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "score_understanding.py",
    "plot_understanding_success_by_round.py",
    "plot_understanding_field_accuracy.py",
]


def main() -> int:
    for script in SCRIPTS:
        cmd = [sys.executable, script]
        print(f"[RUN] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=ANALYSIS_DIR, check=True)

    print("Understanding benchmark assets built successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
