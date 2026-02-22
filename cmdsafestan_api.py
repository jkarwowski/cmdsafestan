#!/usr/bin/env python3
"""Minimal Python API for SafeStan compile/run checks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
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


def evaluate_model_string(
    model_code: str,
    data: dict[str, Any],
    *,
    protect: str | Sequence[str],
    cmdstan_root: str | Path = ".",
    stanc3: str = "safestan",
    runtime_root: str | Path = "safestan/stan",
    seed: int = 12345,
    stream_output: bool = False,
) -> SafeStanResult:
    """Compile and run a model string, returning safety + lp__ summary."""

    root = Path(cmdstan_root).resolve()
    if not (root / "makefile").exists():
        raise FileNotFoundError(f"makefile not found under {root}")

    protect_value = _normalize_protect(protect)
    env = os.environ.copy()
    env.setdefault("STANC3", stanc3)
    runtime = root / Path(runtime_root)
    math_make = runtime / "lib" / "stan_math" / "make" / "compiler_flags"
    stan_header = runtime / "src" / "stan" / "callbacks" / "writer.hpp"
    rapidjson_dir = runtime / "lib" / "rapidjson_1.1.0"
    missing_runtime: list[str] = []
    if not math_make.exists():
        missing_runtime.append(str(math_make))
    if not stan_header.exists():
        missing_runtime.append(str(stan_header))
    if not rapidjson_dir.exists():
        missing_runtime.append(str(rapidjson_dir))
    runtime_ready = not missing_runtime

    with tempfile.TemporaryDirectory(prefix="cmdsafestan-api-", dir=root) as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "model.stan"
        data_path = tmp_path / "data.json"
        output_csv = tmp_path / "output.csv"
        model_exe = tmp_path / f"model{'.exe' if platform.system() == 'Windows' else ''}"

        model_path.write_text(model_code, encoding="utf-8")
        data_path.write_text(json.dumps(data), encoding="utf-8")

        compile_cmd = [
            sys.executable,
            "-m",
            "cmdsafestan_cli",
            "--target",
            "hpp",
            "--stanc3",
            stanc3,
            "--sstan-protect",
            protect_value,
            str(model_path),
        ]
        compile_rc, compile_output = _run_command(
            compile_cmd,
            cwd=root,
            env=env,
            stream_output=stream_output,
        )
        violation = _extract_violation(compile_output)
        if compile_rc != 0:
            return SafeStanResult(
                safe=False,
                log_likelihood=None,
                compile_returncode=compile_rc,
                compile_output=compile_output,
                violation=violation,
                runtime_ready=runtime_ready,
                run_returncode=None,
                run_output="",
            )

        if not runtime_ready:
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
            )

        exe_build_cmd = [
            sys.executable,
            "-m",
            "cmdsafestan_cli",
            "--stanc3",
            stanc3,
            "--sstan-protect",
            protect_value,
            str(model_path),
        ]
        exe_build_rc, exe_build_output = _run_command(
            exe_build_cmd,
            cwd=root,
            env=env,
            stream_output=stream_output,
        )
        if exe_build_rc != 0:
            return SafeStanResult(
                safe=True,
                log_likelihood=None,
                compile_returncode=compile_rc,
                compile_output=compile_output,
                violation=violation,
                runtime_ready=True,
                run_returncode=exe_build_rc,
                run_output=exe_build_output,
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
        run_rc, run_output = _run_command(
            run_cmd,
            cwd=root,
            env=env,
            stream_output=stream_output,
        )
        lp_value = _extract_lp(output_csv) if run_rc == 0 else None

        return SafeStanResult(
            safe=True,
            log_likelihood=lp_value,
            compile_returncode=compile_rc,
            compile_output=compile_output,
            violation=violation,
            runtime_ready=True,
            run_returncode=run_rc,
            run_output=run_output,
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
        "--stream-output",
        action="store_true",
        help="Stream compile/run diagnostics to stdout while executing.",
    )
    args = parser.parse_args(argv)

    model_text = Path(args.model_file).read_text(encoding="utf-8")
    data = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    result = evaluate_model_string(
        model_text,
        data,
        protect=args.protect,
        cmdstan_root=args.cmdstan_root,
        stanc3=args.stanc3,
        runtime_root=args.runtime_root,
        stream_output=args.stream_output,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0 if result.safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
