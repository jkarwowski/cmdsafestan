# cmdsafestan

Small SafeStan-first wrapper for CmdStan.

## Setup

```bash
git submodule update --init --recursive safestan
uv sync
```

Initialize SafeStan runtime submodules (Stan + Math):

```bash
git -C safestan submodule update --init --recursive stan
```

If `safestan/_build/default/src/stanc/stanc.exe` does not exist yet, build compiler once:

```bash
cd safestan
opam exec -- dune build @install
cd ..
```

For executable builds and `lp__` runs, runtime C++ headers/libs are also
required (not just the `stanc3` compiler). In practice this means:

- `safestan/stan/lib/stan_math/`
- `safestan/stan/src/stan/`
- `safestan/stan/lib/rapidjson_1.1.0/`

If these are missing, use `.hpp` mode (safety checks still work).

## Build

Build model executable:

```bash
uv run cmdsafestan --sstan-protect y tests/safestan/models/good_bernoulli.stan
```

Build `.hpp` only:

```bash
uv run cmdsafestan --target hpp --sstan-protect y tests/safestan/models/good_bernoulli.stan
```

## Run

```bash
printf '{"y": 1}\n' > /tmp/good_data.json
tests/safestan/models/good_bernoulli sample \
  num_warmup=200 num_samples=200 \
  data file=/tmp/good_data.json \
  output file=/tmp/good_output.csv
```

## Python: model string -> safety + log-likelihood

Use `evaluate_model_string` for the shortest path from Python string + Python dict data.

```python
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

data = {"y": 1}

safe_result = evaluate_model_string(
    SAFE_MODEL,
    data,
    protect=["y"],
    cmdstan_root=".",
    stream_output=True,  # show build/run diagnostics live
)
print("safe?", safe_result.safe)
print("runtime ready?", safe_result.runtime_ready)
print("lp__", safe_result.log_likelihood)

unsafe_result = evaluate_model_string(
    UNSAFE_MODEL,
    data,
    protect=["y"],
    cmdstan_root=".",
)
print("safe?", unsafe_result.safe)
print("violation:", unsafe_result.violation)
```

Notes:

- `safe_result.safe == True` means compilation passed SafeStan checks.
- `runtime_ready` tells you whether executable run support is available.
- `log_likelihood` is the run's `lp__` from one sample (quick scalar score) when runtime is available, otherwise `None`.
- For unsafe models, `safe == False`, `log_likelihood == None`, and `violation` contains `SStan violation: ...`.
- First call can be slow because it may compile runtime dependencies (SUNDIALS/TBB + model). Use `stream_output=True` to see progress.
