from cmdsafestan.api import evaluate_model_string, init

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

# Fast path after bootstrap has already run.
runtime = init(
    cmdstan_root=".",
    bootstrap=False,
)

unsafe_result = evaluate_model_string(
    UNSAFE_MODEL,
    data,
    protect=["y"],
    runtime=runtime,
)
print("safe?", unsafe_result.safe)
print("violation:", unsafe_result.violation)
