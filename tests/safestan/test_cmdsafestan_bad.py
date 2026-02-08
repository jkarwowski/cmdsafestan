#!/usr/bin/env python3
"""Smoke test: cmdsafestan should reject an invalid SafeStan model."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    test_dir = Path(__file__).resolve().parent
    cmdstan_home = test_dir.parent.parent
    cmdsafestan = cmdstan_home / "cmdsafestan"
    model_template = test_dir / "models" / "bad_target_edit.stan"

    if not cmdsafestan.exists():
        print(f"Missing cmdsafestan command: {cmdsafestan}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    if "STANC3" not in env:
        default_stanc3 = cmdstan_home.parent / "stanc3"
        if default_stanc3.exists():
            env["STANC3"] = str(default_stanc3.resolve())
        else:
            print(
                "STANC3 is not set and ../stanc3 was not found. "
                "Set STANC3 to your SafeStan stanc3 checkout.",
                file=sys.stderr,
            )
            return 2

    with tempfile.TemporaryDirectory(prefix="cmdsafestan-bad-") as tmp:
        tmp_model = Path(tmp) / model_template.name
        shutil.copy2(model_template, tmp_model)
        cmd = [str(cmdsafestan), "--sstan-protect", "y", str(tmp_model)]
        run = subprocess.run(
            cmd,
            cwd=cmdstan_home,
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
