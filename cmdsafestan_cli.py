#!/usr/bin/env python3
"""CmdStan-style SafeStan wrapper command."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call CmdStan make with SafeStan flags enabled "
            "(--sstanc and --sstan-protect)."
        )
    )
    parser.add_argument(
        "model",
        help=(
            "Model path. Accepts a .stan file (recommended) or a CmdStan make "
            "target stem."
        ),
    )
    parser.add_argument(
        "--sstan-protect",
        required=True,
        help="Comma-separated top-level data variables to protect.",
    )
    parser.add_argument(
        "--target",
        default="exe",
        choices=("exe", "hpp"),
        help="Build an executable (default, CmdStan-like) or only generate .hpp.",
    )
    parser.add_argument(
        "--stanc3",
        default=os.environ.get("STANC3", "safestan"),
        help=(
            "STANC3 path to local SafeStan compiler checkout "
            '(defaults to $STANC3 or "safestan").'
        ),
    )
    parser.add_argument(
        "--stancflag",
        action="append",
        default=[],
        help="Extra stanc flag. Repeat to pass multiple flags.",
    )
    parser.add_argument(
        "--make-arg",
        action="append",
        default=[],
        help="Extra argument passed to make. Repeat for multiple values.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Pass -jN to make.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved make command and exit.",
    )
    return parser.parse_args(argv)


def make_target(model: str, target: str) -> str:
    exe_suffix = ".exe" if platform.system() == "Windows" else ""
    if target == "hpp":
        if model.endswith(".stan"):
            return model[:-5] + ".hpp"
        if model.endswith(".hpp"):
            return model
        return model + ".hpp"

    if model.endswith(".stan"):
        return model[:-5] + exe_suffix
    if exe_suffix and model.endswith(exe_suffix):
        return model
    return model + exe_suffix


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not os.path.exists("makefile"):
        print(
            "cmdsafestan: run this command from the CmdStan root (makefile not found).",
            file=sys.stderr,
        )
        return 2

    if args.jobs is not None and args.jobs < 1:
        print("cmdsafestan: --jobs must be >= 1", file=sys.stderr)
        return 2

    env = os.environ.copy()
    flags: list[str] = []
    existing = env.get("STANCFLAGS", "").strip()
    if existing:
        flags.append(existing)
    flags.append("--sstanc")
    flags.append(f"--sstan-protect={args.sstan_protect}")
    flags.extend([f.strip() for f in args.stancflag if f and f.strip()])
    env["STANCFLAGS"] = " ".join(flags)
    if args.stanc3:
        env["STANC3"] = args.stanc3

    command = ["make"]
    if args.jobs is not None:
        command.append(f"-j{args.jobs}")
    command.extend(args.make_arg)
    command.append(make_target(args.model, args.target))

    if "STANC3" in env:
        print(f"[cmdsafestan] STANC3={env['STANC3']}")
    print(f"[cmdsafestan] STANCFLAGS={env['STANCFLAGS']}")
    print(f"[cmdsafestan] {' '.join(shlex.quote(part) for part in command)}")

    if args.dry_run:
        return 0

    result = subprocess.run(command, env=env, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
