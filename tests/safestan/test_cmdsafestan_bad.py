#!/usr/bin/env python3
"""Smoke test: cmdsafestan should reject an invalid SafeStan model."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from shutil import which


def main() -> int:
    if not Path("makefile").exists():
        print("Run this test from the cmdstan/ root.", file=sys.stderr)
        return 2

    if which("cmdsafestan") is None:
        print(
            "cmdsafestan is not on PATH. Run `uv sync` and execute this test via `uv run`.",
            file=sys.stderr,
        )
        return 2

    model = Path("tests/safestan/models/bad_target_edit.stan")
    if not model.exists():
        print(f"Missing test model: {model}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.setdefault("STANC3", "safestan")

    cmd = [
        "cmdsafestan",
        "--target",
        "hpp",
        "--sstan-protect",
        "y",
        str(model),
    ]
    run = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    sys.stdout.write(run.stdout)
    sys.stderr.write(run.stderr)

    if run.returncode == 0:
        print("Expected failure, but compilation succeeded.", file=sys.stderr)
        return 1

    combined = run.stdout + run.stderr
    if "SStan violation:" not in combined:
        print(
            "Expected an SStan violation diagnostic in output.",
            file=sys.stderr,
        )
        return 1

    print("PASS: cmdsafestan rejected the invalid model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
