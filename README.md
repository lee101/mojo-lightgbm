# mojo-lightgbm

`mojo-lightgbm` is a standalone Mojo implementation of the compute-heavy core
of LightGBM-style gradient boosting. It builds trees leaf-wise, bins continuous
features into quantile histograms, evaluates Newton split gain from gradient and
Hessian sums, and traverses complete ensembles in a compiled Mojo kernel.

The Python package is named `mojo_lightgbm`, so it can be imported beside the
real `lightgbm` package for parity testing. Its covered classes and functions
keep the corresponding LightGBM names and common signatures.

## Covered subset

- `Dataset`, `Booster`, and `train`
- `LGBMRegressor` with the L2 regression objective
- `LGBMClassifier` for binary log-loss classification
- Leaf-wise (`best-first`) GBDT growth with `num_leaves` and `max_depth`
- Quantile binning with `max_bin`
- `min_child_samples`, `min_child_weight`, `min_split_gain`, L1 and L2 leaf
  regularization, and learning-rate shrinkage
- Raw scores, probabilities, leaf indices, split/gain feature importance,
  `get_leaf_output()`, and a LightGBM-shaped `dump_model()`

This is not a complete LightGBM replacement. Multiclass, categorical and sparse
features, ranking, sample weights, row/feature sampling, GOSS, DART, random
forests, validation callbacks, early stopping, continued training, SHAP values,
and LightGBM text-model serialization are not covered. Unsupported modes raise
`NotImplementedError`. NaNs are supported with a fixed default-left policy;
upstream LightGBM can instead learn the missing-value direction per split.

## Install

The Pixi environment includes the pinned Mojo nightly, NumPy, pytest, and real
upstream LightGBM:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-lightgbm.so`.

## Usage

```python
import numpy as np
from mojo_lightgbm import LGBMRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 6))
y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] ** 2

model = LGBMRegressor(
    n_estimators=40,
    num_leaves=15,
    max_bin=63,
    learning_rate=0.08,
).fit(X, y)

print(model.predict(X[:5]))
print(model.feature_importances_)
```

Run it inside the environment with `pixi run python example.py`.

The functional API uses the upstream names as well:

```python
from mojo_lightgbm import Dataset, train

booster = train(
    {"objective": "regression", "num_leaves": 15, "max_bin": 63},
    Dataset(X, label=y),
    num_boost_round=40,
)
predictions = booster.predict(X)
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. These inputs stay below the large-histogram parallel threshold,
so both implementations use one CPU thread. Times are the best of two runs
after warm-up. Ratios above 1 mean Mojo is faster.

| Benchmark | mojo-lightgbm | upstream LightGBM (1 thread) | Upstream / Mojo |
|---|---:|---:|---:|
| LGBMRegressor.fit (5k x 12, 20 trees) | 38.34 ms | 25.53 ms | 0.67x |
| Booster.predict (100k x 12, 20 trees) | 80.85 ms | 97.85 ms | 1.21x |
| LGBMClassifier.fit (5k x 12, 20 trees) | 39.41 ms | 29.94 ms | 0.76x |

Upstream LightGBM remains faster at training. Its mature C++ trainer has years
of optimization and performs tree orchestration in C++. This port fuses stable
partitioning with construction of one child histogram, obtains its sibling by
SIMD subtraction from the parent histogram, and scans both child split
candidates through one FFI call. Histogram and partition workspaces are reused
across boosting rounds. The flat Mojo prediction kernel wins on this ensemble
and batch size.

These are real measurements from the included benchmark, not projected
results. Different CPUs, tree shapes, thread counts, and dataset sizes will
change them.

## How it works

Python computes all finite-feature quantiles in one batched operation and keeps
the estimator state and leaf-priority queue. Mojo binary-searches those cuts to
produce a row-major `int64` bin matrix. For each split, Mojo builds the smaller
child's contiguous `[feature, bin]` gradient, Hessian, and count histograms and
derives the sibling histogram from the parent. Another kernel scans prefix sums
and returns the highest-gain valid split. Large histogram builds are divided
into independent row chunks only above a size threshold; smaller builds remain
serial to avoid thread-launch overhead. A stable partition kernel produces
contiguous child row-index arrays.

Trees are flattened into structure-of-arrays buffers: feature index,
floating-point threshold, left and right child index, leaf value, and leaf
index. Prediction crosses the FFI once for the entire batch and ensemble.

All arrays are C-contiguous NumPy buffers. The ctypes boundary passes their
addresses as 64-bit integers; the single Mojo compilation unit reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside non-parametric
`@export` functions. Python owns every allocation, and Mojo neither retains nor
frees caller memory.

No GPU path is included. Histogram construction and tree traversal are
low-arithmetic-intensity, irregular memory-access kernels, while the vectorized
gradient kernels are too small a share of training to repay device transfer and
launch overhead. None of the measured training targets justified GPU execution.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The tests compare kernel results to direct NumPy references and estimator
behavior to the installed upstream `lightgbm` package on identical data.
