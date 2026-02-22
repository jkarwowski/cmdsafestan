#!/usr/bin/env python3
"""Benchmark SafeStan Python API latency split by init and model evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from cmdsafestan.api import SafeStanResult, evaluate_model_string, init

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


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "total_seconds": 0.0,
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
        }

    return {
        "count": len(values),
        "total_seconds": sum(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def _run_and_time(
    model_name: str,
    model_code: str,
    *,
    data: dict[str, Any],
    protect: str,
    runtime: Any,
    seed: int,
    jobs: int | None,
    stream_output: bool,
) -> tuple[float, SafeStanResult]:
    start = time.perf_counter()
    result = evaluate_model_string(
        model_code,
        data,
        protect=protect,
        runtime=runtime,
        seed=seed,
        jobs=jobs,
        stream_output=stream_output,
    )
    elapsed = time.perf_counter() - start

    if model_name == "safe" and not result.safe:
        raise RuntimeError("Safe model unexpectedly failed SafeStan checks.")
    if model_name == "unsafe" and result.safe:
        raise RuntimeError("Unsafe model unexpectedly passed SafeStan checks.")

    return elapsed, result


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.cmdstan_root).resolve()
    if not (root / "makefile").exists():
        raise FileNotFoundError(f"makefile not found under {root}")

    data = {"y": 1}
    init_times: list[float] = []
    safe_times: list[float] = []
    unsafe_times: list[float] = []
    bootstrap_seconds: float | None = None

    if args.bootstrap_first:
        start = time.perf_counter()
        init(
            cmdstan_root=root,
            stanc3=args.stanc3,
            runtime_root=args.runtime_root,
            tmp_root=args.tmp_root,
            bootstrap=True,
            build_runtime=True,
            jobs=args.jobs,
            stream_output=args.stream_output,
        )
        bootstrap_seconds = time.perf_counter() - start

    for i in range(args.runs):
        start = time.perf_counter()
        runtime = init(
            cmdstan_root=root,
            stanc3=args.stanc3,
            runtime_root=args.runtime_root,
            tmp_root=args.tmp_root,
            bootstrap=False,
        )
        init_times.append(time.perf_counter() - start)

        safe_elapsed, safe_result = _run_and_time(
            "safe",
            SAFE_MODEL,
            data=data,
            protect=args.protect,
            runtime=runtime,
            seed=args.seed + i,
            jobs=args.jobs,
            stream_output=args.stream_output,
        )
        safe_times.append(safe_elapsed)

        unsafe_elapsed, unsafe_result = _run_and_time(
            "unsafe",
            UNSAFE_MODEL,
            data=data,
            protect=args.protect,
            runtime=runtime,
            seed=args.seed + i,
            jobs=args.jobs,
            stream_output=args.stream_output,
        )
        unsafe_times.append(unsafe_elapsed)

        if args.verbose:
            print(
                f"run={i + 1}/{args.runs} "
                f"init={init_times[-1]:.3f}s "
                f"safe={safe_elapsed:.3f}s(lp={safe_result.log_likelihood}) "
                f"unsafe={unsafe_elapsed:.3f}s(violation={unsafe_result.violation})"
            )

    benchmark = {
        "settings": {
            "runs": args.runs,
            "cmdstan_root": str(root),
            "stanc3": args.stanc3,
            "runtime_root": args.runtime_root,
            "tmp_root": args.tmp_root,
            "jobs": args.jobs,
            "protect": args.protect,
            "bootstrap_first": args.bootstrap_first,
        },
        "bootstrap_init": {
            "seconds": bootstrap_seconds,
        },
        "per_init": _summary(init_times),
        "per_model": {
            "safe": _summary(safe_times),
            "unsafe": _summary(unsafe_times),
        },
    }

    return benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cmdsafestan API latency split by init and per-model eval."
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs.")
    parser.add_argument(
        "--cmdstan-root",
        default=".",
        help="CmdStan/cmdsafestan root containing makefile.",
    )
    parser.add_argument("--stanc3", default="safestan", help="SafeStan checkout path.")
    parser.add_argument(
        "--runtime-root",
        default="safestan/stan",
        help="Runtime root containing src/stan and lib/stan_math.",
    )
    parser.add_argument(
        "--tmp-root",
        default=".cmdsafestan-tmp",
        help="Temp root used by evaluate_model_string.",
    )
    parser.add_argument(
        "--protect",
        default="y",
        help="Protected data variable names (comma-separated).",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Base random seed.")
    parser.add_argument("--jobs", type=int, default=None, help="Pass -jN to make.")
    parser.add_argument(
        "--bootstrap-first",
        action="store_true",
        help="Run one bootstrap init (sync stanc + make build) before timed runs.",
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Stream build/run output during benchmark.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-run timings.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only final JSON summary.",
    )

    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.jobs is not None and args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    benchmark = run_benchmark(args)

    if not args.json_only:
        print("Benchmark summary")
        print(f"runs: {benchmark['settings']['runs']}")
        if benchmark["bootstrap_init"]["seconds"] is not None:
            print(f"bootstrap init: {benchmark['bootstrap_init']['seconds']:.3f}s")

        per_init = benchmark["per_init"]
        print(
            "per init: "
            f"mean={per_init['mean_seconds']:.3f}s "
            f"median={per_init['median_seconds']:.3f}s "
            f"min={per_init['min_seconds']:.3f}s "
            f"max={per_init['max_seconds']:.3f}s"
        )

        for model_name in ("safe", "unsafe"):
            stats = benchmark["per_model"][model_name]
            print(
                f"per model [{model_name}]: "
                f"mean={stats['mean_seconds']:.3f}s "
                f"median={stats['median_seconds']:.3f}s "
                f"min={stats['min_seconds']:.3f}s "
                f"max={stats['max_seconds']:.3f}s"
            )

        print()

    print(json.dumps(benchmark, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
