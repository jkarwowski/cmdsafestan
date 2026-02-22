#!/usr/bin/env python3
"""Benchmark SafeStan Python API latency with warmup/cleanup and multiprocessing."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cmdsafestan.api import SafeStanResult, evaluate_model_string, init

_STAGE_ORDER = (
    "write_inputs",
    "compile_hpp",
    "compile_exe",
    "run_sample",
    "total",
)

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


def _append_stage_samples(samples: dict[str, list[float]], timings: dict[str, float]) -> None:
    for stage_name, seconds in timings.items():
        samples.setdefault(stage_name, []).append(seconds)


def _summarize_stage_samples(
    samples: dict[str, list[float]]
) -> dict[str, dict[str, float | int]]:
    ordered_names = [stage for stage in _STAGE_ORDER if stage in samples]
    extra_names = sorted(name for name in samples if name not in _STAGE_ORDER)
    return {name: _summary(samples[name]) for name in ordered_names + extra_names}


def _format_stage_timings(timings_seconds: dict[str, float]) -> str:
    ordered_names = [stage for stage in _STAGE_ORDER if stage in timings_seconds]
    extra_names = sorted(name for name in timings_seconds if name not in _STAGE_ORDER)
    parts = [
        f"{stage}={timings_seconds[stage]:.3f}s"
        for stage in ordered_names + extra_names
    ]
    return " ".join(parts)


def _resolve_under_root(root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _cleanup_model_eval_cache(root: Path, runtime_root: str, tmp_root: str) -> None:
    runtime = _resolve_under_root(root, runtime_root)
    tmp = _resolve_under_root(root, tmp_root)

    if tmp.exists():
        shutil.rmtree(tmp)

    for pattern in ("src/cmdstan/main*.o", "src/cmdstan/main*.d"):
        for path in root.glob(pattern):
            path.unlink(missing_ok=True)

    gch_dir = runtime / "src" / "stan" / "model" / "model_header.hpp.gch"
    if gch_dir.exists():
        shutil.rmtree(gch_dir)


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

    if model_name == "safe":
        if not result.safe:
            raise RuntimeError("Safe model unexpectedly failed SafeStan checks.")
        if result.runtime_ready and result.run_returncode != 0:
            raise RuntimeError(
                "Safe model build succeeded but run failed.\n" + result.run_output
            )
        if result.runtime_ready and result.log_likelihood is None:
            raise RuntimeError("Safe model run succeeded but lp__ was not parsed.")
    elif model_name == "unsafe":
        if result.safe:
            raise RuntimeError("Unsafe model unexpectedly passed SafeStan checks.")

    return elapsed, result


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    init_start = time.perf_counter()
    runtime = init(
        cmdstan_root=task["cmdstan_root"],
        stanc3=task["stanc3"],
        runtime_root=task["runtime_root"],
        tmp_root=task["tmp_root"],
        bootstrap=False,
    )
    init_seconds = time.perf_counter() - init_start

    safe_elapsed, safe_result = _run_and_time(
        "safe",
        SAFE_MODEL,
        data={"y": 1},
        protect=task["protect"],
        runtime=runtime,
        seed=task["seed"],
        jobs=task["jobs"],
        stream_output=task["stream_output"],
    )

    result: dict[str, Any] = {
        "run": task["run"] + 1,
        "init_seconds": init_seconds,
        "safe": {
            "elapsed_seconds": safe_elapsed,
            "log_likelihood": safe_result.log_likelihood,
            "timings_seconds": safe_result.timings_seconds,
        },
    }

    if not task["safe_only"]:
        unsafe_elapsed, unsafe_result = _run_and_time(
            "unsafe",
            UNSAFE_MODEL,
            data={"y": 1},
            protect=task["protect"],
            runtime=runtime,
            seed=task["seed"],
            jobs=task["jobs"],
            stream_output=task["stream_output"],
        )
        result["unsafe"] = {
            "elapsed_seconds": unsafe_elapsed,
            "violation": unsafe_result.violation,
            "timings_seconds": unsafe_result.timings_seconds,
        }

    return result


def _run_tasks(
    *,
    runs: int,
    nproc: int,
    task_base: dict[str, Any],
) -> list[dict[str, Any]]:
    if runs <= 0:
        return []

    if nproc == 1:
        return [_run_single({**task_base, "run": i}) for i in range(runs)]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=nproc) as pool:
        future_map = {
            pool.submit(_run_single, {**task_base, "run": i}): i for i in range(runs)
        }
        for future in as_completed(future_map):
            results.append(future.result())

    results.sort(key=lambda item: item["run"])
    return results


def _summarize_runs(
    run_results: list[dict[str, Any]],
    *,
    include_unsafe: bool,
) -> dict[str, Any]:
    init_times: list[float] = [item["init_seconds"] for item in run_results]

    safe_times: list[float] = [item["safe"]["elapsed_seconds"] for item in run_results]
    safe_stage_samples: dict[str, list[float]] = {}
    for item in run_results:
        _append_stage_samples(safe_stage_samples, item["safe"]["timings_seconds"])

    summary: dict[str, Any] = {
        "per_init": _summary(init_times),
        "per_model": {
            "safe": _summary(safe_times),
        },
        "per_stage": {
            "safe": _summarize_stage_samples(safe_stage_samples),
        },
    }

    if include_unsafe:
        unsafe_times: list[float] = [
            item["unsafe"]["elapsed_seconds"] for item in run_results
        ]
        unsafe_stage_samples: dict[str, list[float]] = {}
        for item in run_results:
            _append_stage_samples(unsafe_stage_samples, item["unsafe"]["timings_seconds"])

        summary["per_model"]["unsafe"] = _summary(unsafe_times)
        summary["per_stage"]["unsafe"] = _summarize_stage_samples(unsafe_stage_samples)

    return summary


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.cmdstan_root).resolve()
    if not (root / "makefile").exists():
        raise FileNotFoundError(f"makefile not found under {root}")

    if args.clean_eval_cache:
        _cleanup_model_eval_cache(root, args.runtime_root, args.tmp_root)

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

    task_base = {
        "cmdstan_root": str(root),
        "stanc3": args.stanc3,
        "runtime_root": args.runtime_root,
        "tmp_root": args.tmp_root,
        "protect": args.protect,
        "jobs": args.jobs,
        "stream_output": args.stream_output,
        "safe_only": args.safe_only,
    }

    warmup_wall_seconds = 0.0
    warmup_results: list[dict[str, Any]] = []
    if args.warmup_runs > 0:
        warmup_start = time.perf_counter()
        warmup_results = _run_tasks(
            runs=args.warmup_runs,
            nproc=1,
            task_base={**task_base, "seed": args.seed},
        )
        warmup_wall_seconds = time.perf_counter() - warmup_start

    measured_start = time.perf_counter()
    run_results = _run_tasks(
        runs=args.runs,
        nproc=args.nproc,
        task_base={**task_base, "seed": args.seed + args.warmup_runs},
    )
    measurement_wall_seconds = time.perf_counter() - measured_start

    measured_summary = _summarize_runs(run_results, include_unsafe=not args.safe_only)
    warmup_summary = _summarize_runs(warmup_results, include_unsafe=not args.safe_only)

    benchmark = {
        "settings": {
            "runs": args.runs,
            "nproc": args.nproc,
            "cmdstan_root": str(root),
            "stanc3": args.stanc3,
            "runtime_root": args.runtime_root,
            "tmp_root": args.tmp_root,
            "jobs": args.jobs,
            "jobs_auto_set": args.jobs_auto_set,
            "protect": args.protect,
            "bootstrap_first": args.bootstrap_first,
            "clean_eval_cache": args.clean_eval_cache,
            "warmup_runs": args.warmup_runs,
            "safe_only": args.safe_only,
        },
        "bootstrap_init": {
            "seconds": bootstrap_seconds,
        },
        "warmup": {
            "runs": args.warmup_runs,
            "wall_seconds": warmup_wall_seconds,
            **warmup_summary,
        },
        "measurement": {
            "wall_seconds": measurement_wall_seconds,
            "runs_per_second": (
                (args.runs / measurement_wall_seconds) if measurement_wall_seconds > 0 else 0.0
            ),
            **measured_summary,
        },
        "runs_detail": run_results,
    }

    return benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark cmdsafestan API latency with optional warmup/cache-clean and "
            "multiprocessing."
        )
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of measured runs.")
    parser.add_argument("--nproc", type=int, default=1, help="Number of worker processes.")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Number of warmup runs before measured runs.",
    )
    parser.add_argument(
        "--clean-eval-cache",
        action="store_true",
        help="Remove model-eval cache (tmp dirs/main.o/main.d/PCH) before benchmark.",
    )
    parser.add_argument(
        "--safe-only",
        action="store_true",
        help="Benchmark only safe models (skip unsafe compile-fail path).",
    )
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
        help="Run one bootstrap init (sync stanc + make build) before benchmark.",
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
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.nproc < 1:
        raise ValueError("--nproc must be >= 1")
    if args.jobs is not None and args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    args.jobs_auto_set = False
    if args.nproc > 1 and args.jobs is None:
        args.jobs = 1
        args.jobs_auto_set = True

    return args


def _print_model_summary(benchmark: dict[str, Any], model_name: str) -> None:
    stats = benchmark["measurement"]["per_model"][model_name]
    print(
        f"per model [{model_name}]: "
        f"mean={stats['mean_seconds']:.3f}s "
        f"median={stats['median_seconds']:.3f}s "
        f"min={stats['min_seconds']:.3f}s "
        f"max={stats['max_seconds']:.3f}s"
    )
    stage_stats = benchmark["measurement"]["per_stage"][model_name]
    if stage_stats:
        stage_parts = [
            f"{stage_name}={stage_data['mean_seconds']:.3f}s"
            for stage_name, stage_data in stage_stats.items()
        ]
        print(f"  stage mean [{model_name}]: {' '.join(stage_parts)}")


def main() -> int:
    args = parse_args()
    benchmark = run_benchmark(args)

    if args.verbose:
        for item in benchmark["runs_detail"]:
            safe = item["safe"]
            line = (
                f"run={item['run']}/{args.runs} "
                f"init={item['init_seconds']:.3f}s "
                f"safe={safe['elapsed_seconds']:.3f}s(lp={safe['log_likelihood']})"
            )
            if not args.safe_only:
                unsafe = item["unsafe"]
                line += (
                    f" unsafe={unsafe['elapsed_seconds']:.3f}s"
                    f"(violation={unsafe['violation']})"
                )
            print(line)
            print(f"  safe stages: {_format_stage_timings(safe['timings_seconds'])}")
            if not args.safe_only:
                print(
                    "  unsafe stages: "
                    + _format_stage_timings(item["unsafe"]["timings_seconds"])
                )

    if not args.json_only:
        print("Benchmark summary")
        print(f"runs: {benchmark['settings']['runs']}")
        print(f"nproc: {benchmark['settings']['nproc']}")
        print(f"jobs: {benchmark['settings']['jobs']}")
        if benchmark["settings"]["jobs_auto_set"]:
            print("jobs auto-set to 1 because nproc > 1")
        if benchmark["bootstrap_init"]["seconds"] is not None:
            print(f"bootstrap init: {benchmark['bootstrap_init']['seconds']:.3f}s")
        if benchmark["settings"]["clean_eval_cache"]:
            print("clean eval cache: enabled")
        if benchmark["warmup"]["runs"] > 0:
            print(
                f"warmup: runs={benchmark['warmup']['runs']} "
                f"wall={benchmark['warmup']['wall_seconds']:.3f}s"
            )
        print(
            "measurement wall: "
            f"{benchmark['measurement']['wall_seconds']:.3f}s "
            f"({benchmark['measurement']['runs_per_second']:.3f} runs/s)"
        )

        per_init = benchmark["measurement"]["per_init"]
        print(
            "per init: "
            f"mean={per_init['mean_seconds']:.3f}s "
            f"median={per_init['median_seconds']:.3f}s "
            f"min={per_init['min_seconds']:.3f}s "
            f"max={per_init['max_seconds']:.3f}s"
        )

        _print_model_summary(benchmark, "safe")
        if not args.safe_only:
            _print_model_summary(benchmark, "unsafe")
        print()

    print(json.dumps(benchmark, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
