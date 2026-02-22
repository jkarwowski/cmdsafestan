from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

from cmdsafestan.api import evaluate_model_string, init

GOOD_MODEL_1 = """
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

GOOD_MODEL_2 = """
data {
  int<lower=0, upper=1> y;
}
parameters {
  real alpha;
}
model {
  alpha ~ normal(0, 1);
  y ~ bernoulli_logit(alpha);
}
"""

BAD_MODEL = """
data {
  int<lower=0, upper=1> y;
}
parameters {
  real alpha;
}
model {
  alpha ~ normal(0, 1);
  target += 1;
  y ~ bernoulli_logit(alpha);
}
"""


def _evaluate_task(name: str, model_code: str, seed: int) -> dict[str, object]:
    runtime = init(
        cmdstan_root=".",
        bootstrap=False,
    )
    result = evaluate_model_string(
        model_code,
        {"y": 1},
        protect=["y"],
        runtime=runtime,
        jobs=1,
        seed=seed,
    )
    return {
        "name": name,
        "safe": result.safe,
        "runtime_ready": result.runtime_ready,
        "run_returncode": result.run_returncode,
        "log_likelihood": result.log_likelihood,
        "violation": result.violation,
        "timings_seconds": result.timings_seconds,
    }


def main() -> None:
    print("Warmup: building shared model-eval artifacts once...")
    warm_runtime = init(
        cmdstan_root=".",
        bootstrap=False,
    )
    _ = evaluate_model_string(
        GOOD_MODEL_1,
        {"y": 1},
        protect=["y"],
        runtime=warm_runtime,
        jobs=1,
        seed=12345,
    )

    tasks = [
        ("good-1", GOOD_MODEL_1, 12346),
        ("good-2", GOOD_MODEL_2, 12347),
        ("bad-1", BAD_MODEL, 12348),
    ]

    print("Running 3 models in parallel (nproc=3)...")
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(_evaluate_task, name, model_code, seed)
            for name, model_code, seed in tasks
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: str(item["name"]))
    for item in results:
        print(
            f"{item['name']}: "
            f"safe={item['safe']} "
            f"runtime_ready={item['runtime_ready']} "
            f"run_returncode={item['run_returncode']} "
            f"lp__={item['log_likelihood']} "
            f"violation={item['violation']}"
        )
        print(f"  timings={item['timings_seconds']}")


if __name__ == "__main__":
    main()
