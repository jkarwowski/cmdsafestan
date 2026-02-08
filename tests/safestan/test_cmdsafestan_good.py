#!/usr/bin/env python3
"""Smoke test: cmdsafestan should compile a valid SafeStan model."""

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

    model = Path("tests/safestan/models/good_bernoulli.stan")
    generated_hpp = model.with_suffix(".hpp")
    if not model.exists():
        print(f"Missing test model: {model}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.setdefault("STANC3", "safestan")

    if generated_hpp.exists():
        generated_hpp.unlink()

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

    if run.returncode != 0:
        print(f"Expected success, got return code {run.returncode}", file=sys.stderr)
        return 1

    if not generated_hpp.exists():
        print(f"Expected generated header at {generated_hpp}", file=sys.stderr)
        return 1

    generated_hpp.unlink()

    print("PASS: cmdsafestan compiled the valid model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
