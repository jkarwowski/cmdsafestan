#!/usr/bin/env python3
"""Minimal Python API for SafeStan compile/run checks."""

from __future__ import annotations

import argparse
import json
import os
import platform
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


def evaluate_model_string(
    model_code: str,
    data: dict[str, Any],
    *,
    protect: str | Sequence[str],
    cmdstan_root: str | Path = ".",
    stanc3: str = "safestan",
    seed: int = 12345,
) -> SafeStanResult:
    """Compile and run a model string, returning safety + lp__ summary."""

    root = Path(cmdstan_root).resolve()
    if not (root / "makefile").exists():
        raise FileNotFoundError(f"makefile not found under {root}")

    protect_value = _normalize_protect(protect)
    env = os.environ.copy()
    env.setdefault("STANC3", stanc3)
    math_make = root / stanc3 / "lib" / "stan_math" / "make" / "compiler_flags"
    runtime_ready = math_make.exists()

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
        compile_run = subprocess.run(
            compile_cmd,
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        compile_output = compile_run.stdout + compile_run.stderr
        violation = _extract_violation(compile_output)
        if compile_run.returncode != 0:
            return SafeStanResult(
                safe=False,
                log_likelihood=None,
                compile_returncode=compile_run.returncode,
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
                compile_returncode=compile_run.returncode,
                compile_output=compile_output,
                violation=violation,
                runtime_ready=False,
                run_returncode=None,
                run_output=(
                    f"Runtime unavailable: expected {math_make} for executable build."
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
        exe_build_run = subprocess.run(
            exe_build_cmd,
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        exe_build_output = exe_build_run.stdout + exe_build_run.stderr
        if exe_build_run.returncode != 0:
            return SafeStanResult(
                safe=True,
                log_likelihood=None,
                compile_returncode=compile_run.returncode,
                compile_output=compile_output,
                violation=violation,
                runtime_ready=True,
                run_returncode=exe_build_run.returncode,
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
        run_proc = subprocess.run(
            run_cmd,
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        run_output = run_proc.stdout + run_proc.stderr
        lp_value = _extract_lp(output_csv) if run_proc.returncode == 0 else None

        return SafeStanResult(
            safe=True,
            log_likelihood=lp_value,
            compile_returncode=compile_run.returncode,
            compile_output=compile_output,
            violation=violation,
            runtime_ready=True,
            run_returncode=run_proc.returncode,
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
    args = parser.parse_args(argv)

    model_text = Path(args.model_file).read_text(encoding="utf-8")
    data = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    result = evaluate_model_string(
        model_text,
        data,
        protect=args.protect,
        cmdstan_root=args.cmdstan_root,
        stanc3=args.stanc3,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0 if result.safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
