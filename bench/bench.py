"""End-to-end benchmarks against upstream LightGBM on the same data."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import lightgbm as upstream  # noqa: E402
import mojo_lightgbm as mlgb  # noqa: E402


def timeit(function, repeat=2):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def regression_data(n, d, seed=0):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.normal(size=(n, d)))
    weights = rng.normal(size=d)
    y = np.ascontiguousarray(
        x @ weights + 0.5 * np.sin(2.0 * x[:, 0]) + rng.normal(scale=0.1, size=n)
    )
    return x, y


def main():
    train_x, train_y = regression_data(5_000, 12)
    predict_x, _ = regression_data(100_000, 12, seed=9)
    params = {
        "n_estimators": 20,
        "num_leaves": 15,
        "max_bin": 63,
        "learning_rate": 0.08,
        "min_child_samples": 20,
        "verbosity": -1,
    }
    upstream_params = dict(params)
    upstream_params["n_jobs"] = 1

    mojo_model = mlgb.LGBMRegressor(**params).fit(train_x, train_y)
    upstream_model = upstream.LGBMRegressor(**upstream_params).fit(train_x, train_y)
    cases = [
        (
            "LGBMRegressor.fit (5k x 12, 20 trees)",
            lambda: mlgb.LGBMRegressor(**params).fit(train_x, train_y),
            lambda: upstream.LGBMRegressor(**upstream_params).fit(train_x, train_y),
        ),
        (
            "Booster.predict (100k x 12, 20 trees)",
            lambda: mojo_model.predict(predict_x),
            lambda: upstream_model.predict(predict_x),
        ),
    ]

    binary_y = (train_y > np.median(train_y)).astype(np.int64)
    cases.append(
        (
            "LGBMClassifier.fit (5k x 12, 20 trees)",
            lambda: mlgb.LGBMClassifier(**params).fit(train_x, binary_y),
            lambda: upstream.LGBMClassifier(**upstream_params).fit(train_x, binary_y),
        )
    )

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| Benchmark | mojo-lightgbm | upstream LightGBM (1 thread) | Upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases:
        ours()
        theirs()
        mojo_seconds = timeit(ours)
        upstream_seconds = timeit(theirs)
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {upstream_seconds / mojo_seconds:.2f}x |"
        )


if __name__ == "__main__":
    main()
