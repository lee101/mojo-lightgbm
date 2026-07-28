import numpy as np
import pytest

from mojo_lightgbm._lib import addr, f64, i64, lib
from mojo_lightgbm.engine import (
    _PARALLEL_HISTOGRAM_MIN_CELLS,
    _best_candidate,
    _histograms,
    _partition,
    _quantize,
)


def test_quantize_matches_searchsorted():
    x = np.ascontiguousarray(
        [[-2.0, 4.0], [0.0, 1.0], [0.5, np.nan], [3.0, 8.0]],
        dtype=np.float64,
    )
    cuts = np.zeros((2, 3), dtype=np.float64)
    cuts[0] = [-1.0, 0.5, 2.0]
    cuts[1] = [2.0, 5.0, 7.0]
    ncuts = np.array([3, 3], dtype=np.int64)
    result = _quantize(x, cuts, ncuts, 4)
    expected = np.column_stack(
        [
            np.searchsorted(cuts[0], x[:, 0], side="left"),
            np.searchsorted(cuts[1], x[:, 1], side="left"),
        ]
    ).astype(np.int64)
    expected[2, 1] = 0
    assert np.array_equal(result, expected)


def test_histograms_match_numpy_reference():
    rng = np.random.default_rng(3)
    bins = np.ascontiguousarray(rng.integers(0, 8, size=(80, 5)), dtype=np.int64)
    indices = np.ascontiguousarray(rng.choice(80, 43, replace=False), dtype=np.int64)
    grad = np.ascontiguousarray(rng.normal(size=80))
    hess = np.ascontiguousarray(rng.uniform(0.05, 1.0, size=80))
    hg = np.empty((5, 8), dtype=np.float64)
    hh = np.empty_like(hg)
    hc = np.empty((5, 8), dtype=np.int64)
    lib().mlgb_histograms(
        addr(bins), addr(indices), addr(grad), addr(hess), addr(hg), addr(hh),
        addr(hc), len(indices), 5, 8,
    )
    for feature in range(5):
        expected_g = np.bincount(
            bins[indices, feature], weights=grad[indices], minlength=8
        )
        expected_h = np.bincount(
            bins[indices, feature], weights=hess[indices], minlength=8
        )
        expected_c = np.bincount(bins[indices, feature], minlength=8)
        assert np.allclose(hg[feature], expected_g)
        assert np.allclose(hh[feature], expected_h)
        assert np.array_equal(hc[feature], expected_c)


def test_parallel_histograms_match_numpy_reference():
    rng = np.random.default_rng(13)
    n_features = 16
    n_rows = _PARALLEL_HISTOGRAM_MIN_CELLS // n_features
    bins = np.ascontiguousarray(
        rng.integers(0, 31, size=(n_rows, n_features)), dtype=np.int64
    )
    indices = np.arange(n_rows, dtype=np.int64)
    grad = np.ascontiguousarray(rng.normal(size=n_rows))
    hess = np.ascontiguousarray(rng.uniform(0.05, 1.0, size=n_rows))
    histograms = _histograms(indices, bins, grad, hess, 31)
    actual_grad, actual_hess, actual_count = histograms.owners
    for feature in range(n_features):
        expected_grad = np.bincount(
            bins[:, feature], weights=grad, minlength=31
        )
        expected_hess = np.bincount(
            bins[:, feature], weights=hess, minlength=31
        )
        expected_count = np.bincount(bins[:, feature], minlength=31)
        assert np.allclose(actual_grad[feature], expected_grad)
        assert np.allclose(actual_hess[feature], expected_hess)
        assert np.array_equal(actual_count[feature], expected_count)


def test_simd_histogram_subtraction_and_totals_handle_tail():
    rng = np.random.default_rng(21)
    parent_grad = np.ascontiguousarray(rng.normal(size=(5, 7)))
    parent_hess = np.ascontiguousarray(rng.uniform(1.0, 2.0, size=(5, 7)))
    parent_count = np.ascontiguousarray(
        rng.integers(10, 20, size=(5, 7)), dtype=np.int64
    )
    child_grad = np.ascontiguousarray(rng.normal(size=(5, 7)))
    child_hess = np.ascontiguousarray(rng.uniform(0.0, 1.0, size=(5, 7)))
    child_count = np.ascontiguousarray(
        rng.integers(0, 10, size=(5, 7)), dtype=np.int64
    )
    expected = (
        parent_grad - child_grad,
        parent_hess - child_hess,
        parent_count - child_count,
    )
    lib().mlgb_histogram_subtract_inplace(
        addr(parent_grad), addr(parent_hess), addr(parent_count),
        addr(child_grad), addr(child_hess), addr(child_count),
        parent_grad.size,
    )
    assert np.allclose(parent_grad, expected[0])
    assert np.allclose(parent_hess, expected[1])
    assert np.array_equal(parent_count, expected[2])
    result = np.empty(2, dtype=np.float64)
    lib().mlgb_histogram_totals(
        addr(parent_grad), addr(parent_hess), addr(result), 7
    )
    assert result[0] == pytest.approx(np.sum(parent_grad[0]))
    assert result[1] == pytest.approx(np.sum(parent_hess[0]))


def test_best_split_matches_reference():
    bins = np.ascontiguousarray(
        [[0, 2], [0, 1], [1, 2], [1, 0], [2, 0], [2, 1]], dtype=np.int64
    )
    grad = np.ascontiguousarray([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    hess = np.ones(6, dtype=np.float64)
    indices = np.arange(6, dtype=np.int64)
    ncuts = np.array([2, 2], dtype=np.int64)
    actual = _best_candidate(
        indices, bins, grad, hess, ncuts, 3, 1, 0.0, 0.0, 0.0, 1.0
    )
    candidates = []
    parent = grad.sum() ** 2 / (hess.sum() + 1.0)
    for feature in range(2):
        for threshold in range(2):
            mask = bins[:, feature] <= threshold
            gl, hl = grad[mask].sum(), hess[mask].sum()
            gr, hr = grad[~mask].sum(), hess[~mask].sum()
            gain = gl * gl / (hl + 1.0) + gr * gr / (hr + 1.0) - parent
            candidates.append((gain, feature, threshold))
    expected = max(candidates)
    assert actual[0] == pytest.approx(expected[0])
    chosen = next(
        gain
        for gain, feature, threshold in candidates
        if (feature, threshold) == actual[1:]
    )
    assert chosen == pytest.approx(expected[0])


def test_partition_preserves_all_indices():
    bins = np.ascontiguousarray([[0, 2], [2, 1], [1, 0], [3, 3]], dtype=np.int64)
    indices = np.array([3, 0, 2, 1], dtype=np.int64)
    left, right = _partition(indices, bins, feature=0, threshold_bin=1)
    assert np.array_equal(left, [0, 2])
    assert np.array_equal(right, [3, 1])
    assert np.array_equal(np.sort(np.r_[left, right]), np.arange(4))


@pytest.mark.parametrize("binary", [False, True])
def test_gradient_kernels(binary):
    raw = np.ascontiguousarray(np.linspace(-2.0, 1.5, 11))
    label = np.ascontiguousarray(np.arange(11) % 2, dtype=np.float64)
    grad = np.empty(11)
    hess = np.empty(11)
    function = lib().mlgb_binary_gradients if binary else lib().mlgb_regression_gradients
    function(addr(raw), addr(label), addr(grad), addr(hess), len(raw))
    if binary:
        probability = 1.0 / (1.0 + np.exp(-raw))
        assert np.allclose(grad, probability - label)
        assert np.allclose(hess, probability * (1.0 - probability))
    else:
        assert np.allclose(grad, raw - label)
        assert np.array_equal(hess, np.ones(len(raw)))


def test_ffi_helpers_keep_matching_numpy_buffers_zero_copy():
    floats = np.arange(8, dtype=np.float64)
    integers = np.arange(8, dtype=np.int64)
    assert f64(floats) is floats
    assert i64(integers) is integers


def test_ffi_helpers_reject_lossy_or_invalid_buffers():
    with pytest.raises(TypeError, match="complex"):
        f64(np.array([1.0 + 2.0j]))
    with pytest.raises(ValueError, match="integral"):
        i64(np.array([1.5]))
    with pytest.raises(TypeError, match="C-contiguous"):
        addr(np.arange(12, dtype=np.float64).reshape(3, 4).T)
