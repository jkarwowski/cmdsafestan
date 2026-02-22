# cmdsafestan

SafeStan-first wrapper for CmdStan.

## Setup

```bash
git submodule update --init --recursive safestan
git -C safestan submodule update --init --recursive stan
uv sync
```

If `safestan/_build/default/src/stanc/stanc.exe` does not exist yet:

```bash
cd safestan
opam exec -- dune build @install
cd ..
```

## Build and run

Build executable:

```bash
uv run cmdsafestan --sstan-protect y tests/safestan/models/good_bernoulli.stan
```

Run executable:

```bash
printf '{"y": 1}\n' > /tmp/good_data.json
tests/safestan/models/good_bernoulli sample \
  num_warmup=200 num_samples=200 \
  data file=/tmp/good_data.json \
  output file=/tmp/good_output.csv
```

Build `.hpp` only:

```bash
uv run cmdsafestan --target hpp --sstan-protect y tests/safestan/models/good_bernoulli.stan
```

## Python API (self-contained safe/unsafe example)

```python
from cmdsafestan_api import evaluate_model_string, init

SAFE_MODEL = """
data { int<lower=0, upper=1> y; }
parameters { real<lower=0, upper=1> theta; }
model {
  theta ~ beta(1, 1);
  y ~ bernoulli(theta);
}
"""

UNSAFE_MODEL = """
data { int<lower=0, upper=1> y; }
parameters { real<lower=0, upper=1> theta; }
model {
  theta ~ beta(1, 1);
  target += 1;
  y ~ bernoulli(theta);
}
"""

data = {"y": 1}

# Run once before multiprocessing workers start.
# This syncs bin/stanc and builds shared runtime deps once.
runtime = init(
    cmdstan_root=".",
    bootstrap=True,
    build_runtime=True,
    jobs=4,
    stream_output=True,
)

safe_result = evaluate_model_string(
    SAFE_MODEL,
    data,
    protect=["y"],
    runtime=runtime,
    stream_output=True,
)
print("safe?", safe_result.safe)                # True
print("runtime ready?", safe_result.runtime_ready)
print("lp__", safe_result.log_likelihood)       # float when runtime available

unsafe_result = evaluate_model_string(
    UNSAFE_MODEL,
    data,
    protect=["y"],
    runtime=runtime,
)
print("safe?", unsafe_result.safe)              # False
print("violation:", unsafe_result.violation)    # "SStan violation: ..."
```

Behavior notes:

- `evaluate_model_string` uses a unique temporary directory per call for model/data/output files, so concurrent workers do not share per-model artifacts.
- `init(..., bootstrap=True)` makes later calls faster by reusing shared built dependencies and skipping repeated global `bin/stanc` sync.
- Default temp root after `init` is `.cmdsafestan-tmp/` under repo root (override with `tmp_root=...` in `init`).

See runnable scripts in `examples/python/example.py` and `examples/python/example_good.py`.
