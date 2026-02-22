#!/usr/bin/env python3
"""Smoke test: plain mode should compile without SafeStan enforcement."""

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
    generated_hpp = model.with_suffix(".hpp")
    if not model.exists():
        print(f"Missing test model: {model}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.setdefault("STANC3", "safestan")
    env["STANCFLAGS"] = "--warn-pedantic --sstanc --sstan-protect y --sstan-protect=z"

    if generated_hpp.exists():
        generated_hpp.unlink()

    cmd = [
        "cmdsafestan",
        "--mode",
        "plain",
        "--target",
        "hpp",
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
        print(f"Expected plain mode success, got return code {run.returncode}", file=sys.stderr)
        return 1

    if not generated_hpp.exists():
        print(f"Expected generated header at {generated_hpp}", file=sys.stderr)
        return 1

    if "SStan violation:" in (run.stdout + run.stderr):
        print("Did not expect SStan violation output in plain mode.", file=sys.stderr)
        return 1

    log_output = run.stdout + run.stderr
    if "--sstanc" in log_output or "--sstan-protect" in log_output:
        print("Did not expect SafeStan flags to remain in plain mode STANCFLAGS.", file=sys.stderr)
        return 1
    if "--warn-pedantic" not in log_output:
        print("Expected non-SafeStan STANCFLAGS entries to be preserved.", file=sys.stderr)
        return 1

    generated_hpp.unlink()
    print("PASS: cmdsafestan plain mode compiled the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
