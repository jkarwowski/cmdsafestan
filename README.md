# CmdStan for SafeStan

This directory keeps the standard CmdStan workflow, with a SafeStan-oriented
entrypoint layered on top.

For upstream CmdStan background (interfaces, licensing, general installation
notes), see:

- `README_old.md`

## What Is Added Here

- `cmdsafestan`: a CmdStan-style CLI wrapper around `make` that always passes:
  - `--sstanc`
  - `--sstan-protect=<vars>`
- `safestan` is the only supported module path in this fork (`stan` submodule is not used)
- local compiler integration via `STANC3=...` (for your SafeStan-enabled
  `stanc3` checkout)
- two Python smoke tests for one valid and one invalid SafeStan model
- optional `safestan` submodule pointing at `jkarwowski/safestanc3` (`master`)

## Local SafeStan stanc3

CmdStan already supports using a local `stanc3` source tree. In this fork,
that is the intended path.

In `make/local`:

```make
STANC3=safestan
```

Or initialize the bundled `safestan` submodule:

```bash
git submodule update --init --remote safestan
```

then in `make/local`:

```make
STANC3=safestan
```

Or per-command:

```bash
STANC3=safestan make path/to/model.hpp
```

If `dune` is not directly on your `PATH`, local compiler builds can use:

```bash
opam exec -- dune build @install
```

## Using `cmdsafestan`

Install with uv (editable/project install):

```bash
uv sync
```

(`uv.lock` pins this project as `source = { editable = "." }`.)

Then run like normal CmdStan from `cmdstan/`:

(`cmdsafestan` defaults to `STANC3=safestan`; override with `--stanc3` or
environment variable `STANC3`.)

Build model executable (default target):

```bash
uv run cmdsafestan --sstan-protect y path/to/model.stan
```

Translate only to C++ header (`.hpp`):

```bash
uv run cmdsafestan --target hpp --sstan-protect y path/to/model.stan
```

Use an explicit local SafeStan compiler checkout:

```bash
uv run cmdsafestan --stanc3 safestan --sstan-protect y path/to/model.stan
```

Add extra stanc flags:

```bash
uv run cmdsafestan --sstan-protect y --stancflag=--warn-pedantic path/to/model.stan
```

## Python Smoke Tests (uv-managed)

From `cmdstan/`:

```bash
uv sync
uv run --project . python tests/safestan/test_cmdsafestan_good.py
uv run --project . python tests/safestan/test_cmdsafestan_bad.py
```

Both tests call `cmdsafestan` and expect:

- `good`: success and generated `.hpp`
- `bad`: non-zero exit and `SStan violation:` diagnostic
