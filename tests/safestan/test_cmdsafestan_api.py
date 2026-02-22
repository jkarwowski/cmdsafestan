#!/usr/bin/env python3
"""Smoke tests for the Python evaluate_model_string helper."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from cmdsafestan_api import evaluate_model_string

SAFE_MODEL = """
data {
  int<lower=0, upper=1> y;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(1, 1);
  y ~ bernoulli(theta);
}
"""

UNSAFE_MODEL = """
data {
  int<lower=0, upper=1> y;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(1, 1);
  target += 1;
  y ~ bernoulli(theta);
}
"""


def main() -> int:
    if not Path("makefile").exists():
        print("Run this test from the cmdstan/ root.", file=sys.stderr)
        return 2

    safe_result = evaluate_model_string(
        SAFE_MODEL,
        {"y": 1},
        protect=["y"],
        cmdstan_root=".",
    )
    if not safe_result.safe:
        print("Expected safe model to pass SafeStan compile checks.", file=sys.stderr)
        print(safe_result.compile_output, file=sys.stderr)
        return 1
    if safe_result.runtime_ready:
        if safe_result.run_returncode != 0:
            print("Expected safe model run to succeed.", file=sys.stderr)
            print(safe_result.run_output, file=sys.stderr)
            return 1
        if safe_result.log_likelihood is None or not math.isfinite(
            safe_result.log_likelihood
        ):
            print("Expected finite lp__ value for safe model run.", file=sys.stderr)
            return 1
    else:
        if safe_result.log_likelihood is not None:
            print("Expected no lp__ when runtime is unavailable.", file=sys.stderr)
            return 1
        if "Runtime unavailable:" not in safe_result.run_output:
            print("Expected explicit runtime-unavailable message.", file=sys.stderr)
            return 1

    unsafe_result = evaluate_model_string(
        UNSAFE_MODEL,
        {"y": 1},
        protect=["y"],
        cmdstan_root=".",
    )
    if unsafe_result.safe:
        print("Expected unsafe model to fail SafeStan compile checks.", file=sys.stderr)
        return 1
    if unsafe_result.violation is None or "SStan violation:" not in unsafe_result.violation:
        print("Expected explicit SStan violation diagnostic.", file=sys.stderr)
        print(unsafe_result.compile_output, file=sys.stderr)
        return 1
    if unsafe_result.log_likelihood is not None:
        print("Unsafe model should not produce lp__.", file=sys.stderr)
        return 1

    print("PASS: evaluate_model_string reports safe/unsafe status and lp__.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
