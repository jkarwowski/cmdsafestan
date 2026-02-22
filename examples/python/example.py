from cmdsafestan.api import evaluate_model_string, init

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

# One-time setup (sync stanc + build runtime dependencies).
# Do this once before launching multiprocessing workers.
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
print("safe?", safe_result.safe)
print("runtime ready?", safe_result.runtime_ready)
print("lp__", safe_result.log_likelihood)

unsafe_result = evaluate_model_string(
    UNSAFE_MODEL,
    data,
    protect=["y"],
    runtime=runtime,
)
print("safe?", unsafe_result.safe)
print("violation:", unsafe_result.violation)
