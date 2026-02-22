#!/usr/bin/env python3
"""Python API for SafeStan compile/run checks with one-time bootstrap support."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class SafeStanResult:
    """Result for compile + optional run of a model string."""

    safe: bool
    log_likelihood: float | None
    compile_returncode: int
    compile_output: str
    violation: str | None
    runtime_ready: bool
    run_returncode: int | None
    run_output: str
    timings_seconds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SafeStanRuntime:
    """Resolved runtime config returned by init()."""

    cmdstan_root: Path
    stanc3: str
    runtime_root: Path
    tmp_root: Path
    no_stanc_sync: bool


def _normalize_protect(protect: str | Sequence[str]) -> str:
    if isinstance(protect, str):
        value = protect.strip()
    else:
        parts = [part.strip() for part in protect if part and part.strip()]
        value = ",".join(parts)
    if not value:
        raise ValueError("protect must be a non-empty string or sequence of strings")
    return value


def _extract_violation(output: str) -> str | None:
    marker = "SStan violation:"
    idx = output.find(marker)
    if idx == -1:
        return None
    line = output[idx:].splitlines()[0].strip()
    return line


def _extract_lp(csv_path: Path) -> float | None:
    header: list[str] | None = None
    with csv_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if header is None:
                header = parts
                continue
            if "lp__" not in header:
                return None
            lp_idx = header.index("lp__")
            try:
                return float(parts[lp_idx])
            except (IndexError, ValueError):
                return None
    return None


def _resolve_root(cmdstan_root: str | Path) -> Path:
    root = Path(cmdstan_root).resolve()
    if not (root / "makefile").exists():
        raise FileNotFoundError(f"makefile not found under {root}")
    return root


def _resolve_under_root(path_value: str | Path, *, root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _resolve_runtime(
    *,
    cmdstan_root: str | Path,
    stanc3: str,
    runtime_root: str | Path,
    tmp_root: str | Path | None,
    no_stanc_sync: bool,
) -> SafeStanRuntime:
    root = _resolve_root(cmdstan_root)
    runtime = _resolve_under_root(runtime_root, root=root)
    resolved_tmp_root = root if tmp_root is None else _resolve_under_root(tmp_root, root=root)
    return SafeStanRuntime(
        cmdstan_root=root,
        stanc3=stanc3,
        runtime_root=runtime,
        tmp_root=resolved_tmp_root,
        no_stanc_sync=no_stanc_sync,
    )


def _missing_runtime_paths(runtime: SafeStanRuntime) -> list[str]:
    math_make = runtime.runtime_root / "lib" / "stan_math" / "make" / "compiler_flags"
    stan_header = runtime.runtime_root / "src" / "stan" / "callbacks" / "writer.hpp"
    rapidjson_dir = runtime.runtime_root / "lib" / "rapidjson_1.1.0"
    missing: list[str] = []
    if not math_make.exists():
        missing.append(str(math_make))
    if not stan_header.exists():
        missing.append(str(stan_header))
    if not rapidjson_dir.exists():
        missing.append(str(rapidjson_dir))
    return missing


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stream_output: bool,
) -> tuple[int, str]:
    if not stream_output:
        run = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        return run.returncode, run.stdout + run.stderr

    print("[cmdsafestan_api] " + " ".join(shlex.quote(part) for part in command))
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    merged_output: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        merged_output.append(line)
    return proc.wait(), "".join(merged_output)


def _sync_stanc_binary(
    runtime: SafeStanRuntime,
    *,
    env: dict[str, str],
    jobs: int | None,
    stream_output: bool,
) -> None:
    stanc_suffix = ".exe" if platform.system() == "Windows" else ""
    stanc_target = runtime.cmdstan_root / "bin" / f"stanc{stanc_suffix}"
    stanc_checkout = _resolve_under_root(runtime.stanc3, root=runtime.cmdstan_root)
    local_stanc = stanc_checkout / "_build" / "default" / "src" / "stanc" / "stanc.exe"

    if local_stanc.is_file():
        stanc_target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = stanc_target.with_name(f"{stanc_target.name}.tmp.{os.getpid()}")
        shutil.copy2(local_stanc, tmp_target)
        if platform.system() != "Windows":
            os.chmod(tmp_target, os.stat(tmp_target).st_mode | 0o111)
        os.replace(tmp_target, stanc_target)
        if stream_output:
            print(f"[cmdsafestan_api] synced {stanc_target} from {local_stanc}")
        return

    if stanc_target.is_file():
        if stream_output:
            print(f"[cmdsafestan_api] using existing {stanc_target}")
        return

    bootstrap_cmd = ["make"]
    if jobs is not None:
        bootstrap_cmd.append(f"-j{jobs}")
    bootstrap_cmd.extend(["-B", f"bin/stanc{stanc_suffix}"])
    rc, output = _run_command(
        bootstrap_cmd,
        cwd=runtime.cmdstan_root,
        env=env,
        stream_output=stream_output,
    )
    if rc != 0:
        raise RuntimeError("Failed to bootstrap bin/stanc.\n" + output)


def _cmdsafestan_command(
    *,
    runtime: SafeStanRuntime,
    model_path: Path,
    protect_value: str,
    target: str | None,
    no_stanc_sync: bool,
    jobs: int | None,
) -> list[str]:
    command = [sys.executable, "-m", "cmdsafestan.cli"]
    if target is not None:
        command.extend(["--target", target])
    if jobs is not None:
        command.extend(["--jobs", str(jobs)])
    if no_stanc_sync:
        command.append("--no-stanc-sync")
    command.extend(
        [
            "--stanc3",
            runtime.stanc3,
            "--sstan-protect",
            protect_value,
            str(model_path),
        ]
    )
    return command


def init(
    *,
    cmdstan_root: str | Path = ".",
    stanc3: str = "safestan",
    runtime_root: str | Path = "safestan/stan",
    tmp_root: str | Path | None = ".cmdsafestan-tmp",
    bootstrap: bool = True,
    build_runtime: bool = True,
    jobs: int | None = None,
    stream_output: bool = False,
) -> SafeStanRuntime:
    """Initialize runtime once so later evaluate_model_string calls can skip global sync."""

    if jobs is not None and jobs < 1:
        raise ValueError("jobs must be >= 1")

    runtime = _resolve_runtime(
        cmdstan_root=cmdstan_root,
        stanc3=stanc3,
        runtime_root=runtime_root,
        tmp_root=tmp_root,
        no_stanc_sync=False,
    )
    runtime.tmp_root.mkdir(parents=True, exist_ok=True)
    if not bootstrap:
        return runtime

    env = os.environ.copy()
    env["STANC3"] = runtime.stanc3
    _sync_stanc_binary(runtime, env=env, jobs=jobs, stream_output=stream_output)

    if build_runtime:
        build_cmd = ["make"]
        if jobs is not None:
            build_cmd.append(f"-j{jobs}")
        build_cmd.append("build")
        build_rc, build_output = _run_command(
            build_cmd,
            cwd=runtime.cmdstan_root,
            env=env,
            stream_output=stream_output,
        )
        if build_rc != 0:
            raise RuntimeError("SafeStan runtime bootstrap failed.\n" + build_output)

    # After bootstrap, workers can avoid rewriting shared bin/stanc each call.
    return SafeStanRuntime(
        cmdstan_root=runtime.cmdstan_root,
        stanc3=runtime.stanc3,
        runtime_root=runtime.runtime_root,
        tmp_root=runtime.tmp_root,
        no_stanc_sync=True,
    )


def evaluate_model_string(
    model_code: str,
    data: dict[str, Any],
    *,
    protect: str | Sequence[str],
    runtime: SafeStanRuntime | None = None,
    cmdstan_root: str | Path = ".",
    stanc3: str = "safestan",
    runtime_root: str | Path = "safestan/stan",
    tmp_root: str | Path | None = None,
    seed: int = 12345,
    stream_output: bool = False,
    jobs: int | None = None,
    no_stanc_sync: bool | None = None,
) -> SafeStanResult:
    """Compile and run a model string, returning safety + lp__ summary."""

    if jobs is not None and jobs < 1:
        raise ValueError("jobs must be >= 1")

    protect_value = _normalize_protect(protect)
    if runtime is None:
        runtime = _resolve_runtime(
            cmdstan_root=cmdstan_root,
            stanc3=stanc3,
            runtime_root=runtime_root,
            tmp_root=tmp_root,
            no_stanc_sync=False,
        )
    runtime.tmp_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["STANC3"] = runtime.stanc3

    use_no_stanc_sync = runtime.no_stanc_sync if no_stanc_sync is None else no_stanc_sync
    missing_runtime = _missing_runtime_paths(runtime)
    runtime_ready = not missing_runtime
    total_start = time.perf_counter()
    timings_seconds: dict[str, float] = {}

    with tempfile.TemporaryDirectory(prefix="cmdsafestan-api-", dir=runtime.tmp_root) as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "model.stan"
        data_path = tmp_path / "data.json"
        output_csv = tmp_path / "output.csv"
        model_exe = tmp_path / f"model{'.exe' if platform.system() == 'Windows' else ''}"

        write_start = time.perf_counter()
        model_path.write_text(model_code, encoding="utf-8")
        data_path.write_text(json.dumps(data), encoding="utf-8")
        timings_seconds["write_inputs"] = time.perf_counter() - write_start

        compile_cmd = _cmdsafestan_command(
            runtime=runtime,
            model_path=model_path,
            protect_value=protect_value,
            target="hpp",
            no_stanc_sync=use_no_stanc_sync,
            jobs=jobs,
        )
        compile_hpp_start = time.perf_counter()
        compile_rc, compile_output = _run_command(
            compile_cmd,
            cwd=runtime.cmdstan_root,
            env=env,
            stream_output=stream_output,
        )
        timings_seconds["compile_hpp"] = time.perf_counter() - compile_hpp_start
        violation = _extract_violation(compile_output)
        if compile_rc != 0:
            timings_seconds["total"] = time.perf_counter() - total_start
            return SafeStanResult(
                safe=False,
                log_likelihood=None,
                compile_returncode=compile_rc,
                compile_output=compile_output,
                violation=violation,
                runtime_ready=runtime_ready,
                run_returncode=None,
                run_output="",
                timings_seconds=timings_seconds,
            )

        if not runtime_ready:
            timings_seconds["total"] = time.perf_counter() - total_start
            return SafeStanResult(
                safe=True,
                log_likelihood=None,
                compile_returncode=compile_rc,
                compile_output=compile_output,
                violation=violation,
                runtime_ready=False,
                run_returncode=None,
                run_output=(
                    "Runtime unavailable: missing required runtime paths: "
                    + ", ".join(missing_runtime)
                ),
                timings_seconds=timings_seconds,
            )

        exe_build_cmd = _cmdsafestan_command(
            runtime=runtime,
            model_path=model_path,
            protect_value=protect_value,
            target=None,
            no_stanc_sync=use_no_stanc_sync,
            jobs=jobs,
        )
        compile_exe_start = time.perf_counter()
        exe_build_rc, exe_build_output = _run_command(
            exe_build_cmd,
            cwd=runtime.cmdstan_root,
            env=env,
            stream_output=stream_output,
        )
        timings_seconds["compile_exe"] = time.perf_counter() - compile_exe_start
        if exe_build_rc != 0:
            timings_seconds["total"] = time.perf_counter() - total_start
            return SafeStanResult(
                safe=True,
                log_likelihood=None,
                compile_returncode=compile_rc,
                compile_output=compile_output,
                violation=violation,
                runtime_ready=True,
                run_returncode=exe_build_rc,
                run_output=exe_build_output,
                timings_seconds=timings_seconds,
            )

        run_cmd = [
            str(model_exe),
            "sample",
            "num_warmup=0",
            "num_samples=1",
            "adapt",
            "engaged=0",
            "random",
            f"seed={seed}",
            "data",
            f"file={data_path}",
            "output",
            f"file={output_csv}",
            "refresh=0",
        ]
        run_start = time.perf_counter()
        run_rc, run_output = _run_command(
            run_cmd,
            cwd=runtime.cmdstan_root,
            env=env,
            stream_output=stream_output,
        )
        timings_seconds["run_sample"] = time.perf_counter() - run_start
        lp_value = _extract_lp(output_csv) if run_rc == 0 else None
        timings_seconds["total"] = time.perf_counter() - total_start

        return SafeStanResult(
            safe=True,
            log_likelihood=lp_value,
            compile_returncode=compile_rc,
            compile_output=compile_output,
            violation=violation,
            runtime_ready=True,
            run_returncode=run_rc,
            run_output=run_output,
            timings_seconds=timings_seconds,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile/run model from files and print SafeStan result as JSON."
    )
    parser.add_argument("--model-file", required=True, help="Path to .stan model file.")
    parser.add_argument("--data-file", required=True, help="Path to JSON data file.")
    parser.add_argument(
        "--protect",
        required=True,
        help="Comma-separated protected data variables, e.g. y or y,x.",
    )
    parser.add_argument(
        "--cmdstan-root",
        default=".",
        help="CmdStan root directory (default: current directory).",
    )
    parser.add_argument(
        "--stanc3",
        default="safestan",
        help="Local SafeStan stanc3 path (default: safestan).",
    )
    parser.add_argument(
        "--runtime-root",
        default="safestan/stan",
        help="Runtime root containing src/stan and lib/{stan_math,rapidjson_1.1.0}.",
    )
    parser.add_argument(
        "--tmp-root",
        default=None,
        help=(
            "Directory for per-call temporary model build dirs. "
            "Defaults to cmdstan-root unless --bootstrap-init is used."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Pass -jN to make via cmdsafestan wrapper.",
    )
    parser.add_argument(
        "--bootstrap-init",
        action="store_true",
        help=(
            "Run one-time init() (sync stanc + make build) before evaluating model."
        ),
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Stream compile/run diagnostics to stdout while executing.",
    )
    args = parser.parse_args(argv)

    model_text = Path(args.model_file).read_text(encoding="utf-8")
    data = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    runtime: SafeStanRuntime | None = None
    if args.bootstrap_init:
        runtime = init(
            cmdstan_root=args.cmdstan_root,
            stanc3=args.stanc3,
            runtime_root=args.runtime_root,
            tmp_root=args.tmp_root,
            jobs=args.jobs,
            stream_output=args.stream_output,
        )

    result = evaluate_model_string(
        model_text,
        data,
        protect=args.protect,
        runtime=runtime,
        cmdstan_root=args.cmdstan_root,
        stanc3=args.stanc3,
        runtime_root=args.runtime_root,
        tmp_root=args.tmp_root,
        jobs=args.jobs,
        stream_output=args.stream_output,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
