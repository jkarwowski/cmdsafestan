# CmdStan for SafeStan

This directory keeps the standard CmdStan workflow, with a SafeStan-oriented
entrypoint layered on top.

For upstream CmdStan background (interfaces, licensing, general installation
notes), see:

- `README_old.md`

## What Is Added Here

- `cmdsafestan`: a thin wrapper around `make` that always passes:
  - `--sstanc`
  - `--sstan-protect=<vars>`
- local `stanc3` integration via `STANC3=...` (for your SafeStan-enabled
  `stanc3` checkout)
- two Python smoke tests for one valid and one invalid SafeStan model

## Local SafeStan stanc3

CmdStan already supports using a local `stanc3` source tree. In this fork,
that is the intended path.

In `make/local`:

```make
STANC3=../stanc3
```

Or per-command:

```bash
STANC3=../stanc3 make path/to/model.hpp
```

## Using `cmdsafestan`

Translate model to C++ header (`.hpp`, default):

```bash
./cmdsafestan --sstan-protect y path/to/model.stan
```

Build full model executable:

```bash
./cmdsafestan --target exe --sstan-protect y path/to/model.stan
```

Use an explicit local SafeStan compiler checkout:

```bash
./cmdsafestan --stanc3 ../stanc3 --sstan-protect y path/to/model.stan
```

Add extra stanc flags:

```bash
./cmdsafestan --sstan-protect y --stancflag=--warn-pedantic path/to/model.stan
```

## Python Smoke Tests (uv-managed)

This directory now includes `pyproject.toml` so tests can be run through `uv`.

From `cmdstan/`:

```bash
uv run --project . python tests/safestan/test_cmdsafestan_good.py
uv run --project . python tests/safestan/test_cmdsafestan_bad.py
```

Both tests call `cmdsafestan` and expect:

- `good`: success and generated `.hpp`
- `bad`: non-zero exit and `SStan violation:` diagnostic
