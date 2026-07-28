from __future__ import annotations

import heapq
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ._lib import addr, f64, i64, lib
from .basic import Booster, Dataset, Tree

_PARALLEL_HISTOGRAM_MIN_CELLS = 262_144
_HISTOGRAM_EXECUTOR = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))


def _integer_parameter(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _call_kernel(arguments):
    function, args = arguments
    function(*args)


def _cuts(x: np.ndarray, max_bin: int):
    n_features = x.shape[1]
    cut_matrix = np.zeros((n_features, max_bin - 1), dtype=np.float64)
    ncuts = np.zeros(n_features, dtype=np.int64)
    probabilities = np.arange(1, max_bin, dtype=np.float64) / max_bin
    if np.all(np.isfinite(x)):
        quantiles = np.quantile(x, probabilities, axis=0, method="linear")
        maxima = np.max(x, axis=0)
        for feature in range(n_features):
            cuts = np.unique(quantiles[:, feature])
            cuts = cuts[cuts < maxima[feature]]
            ncuts[feature] = len(cuts)
            cut_matrix[feature, : len(cuts)] = cuts
        return cut_matrix, ncuts
    for feature in range(n_features):
        column = x[:, feature]
        finite = column[np.isfinite(column)]
        if not len(finite):
            continue
        cuts = np.unique(np.quantile(finite, probabilities, method="linear"))
        cuts = cuts[cuts < np.max(finite)]
        ncuts[feature] = len(cuts)
        cut_matrix[feature, : len(cuts)] = cuts
    return np.ascontiguousarray(cut_matrix), ncuts


def _quantize(x, cuts, ncuts, max_bin):
    bins = np.empty(x.shape, dtype=np.int64)
    lib().mlgb_quantize(
        addr(x), addr(cuts), addr(ncuts), addr(bins),
        len(x), x.shape[1], max_bin,
    )
    return bins


def _leaf_value(g, h, reg_alpha, reg_lambda, learning_rate):
    shrunk = np.sign(g) * max(abs(g) - reg_alpha, 0.0)
    return -learning_rate * shrunk / (h + reg_lambda)


@dataclass(slots=True)
class _Histograms:
    grad_addr: int
    hess_addr: int
    count_addr: int
    n_features: int
    max_bin: int
    owners: tuple[np.ndarray, np.ndarray, np.ndarray]


def _histogram_workspace(n_slots, n_features, max_bin):
    owners = (
        np.empty((n_slots, n_features, max_bin), dtype=np.float64),
        np.empty((n_slots, n_features, max_bin), dtype=np.float64),
        np.empty((n_slots, n_features, max_bin), dtype=np.int64),
    )
    bases = tuple(addr(array) for array in owners)
    stride = n_features * max_bin * 8
    return [
        _Histograms(
            bases[0] + slot * stride,
            bases[1] + slot * stride,
            bases[2] + slot * stride,
            n_features,
            max_bin,
            owners,
        )
        for slot in range(n_slots)
    ]


def _histograms(
    indices, bins, grad, hess, max_bin, input_addresses=None, target=None
):
    n_features = bins.shape[1]
    if target is None:
        owners = (
            np.empty((n_features, max_bin), dtype=np.float64),
            np.empty((n_features, max_bin), dtype=np.float64),
            np.empty((n_features, max_bin), dtype=np.int64),
        )
        target = _Histograms(
            addr(owners[0]), addr(owners[1]), addr(owners[2]),
            n_features, max_bin, owners,
        )
    if input_addresses is None:
        bins_addr, grad_addr, hess_addr = addr(bins), addr(grad), addr(hess)
    else:
        bins_addr, grad_addr, hess_addr = input_addresses
    indices_addr = addr(indices)
    if (
        len(indices) * n_features >= _PARALLEL_HISTOGRAM_MIN_CELLS
        and n_features >= 4
    ):
        cells = len(indices) * n_features
        n_workers = min(
            8, max(2, cells // _PARALLEL_HISTOGRAM_MIN_CELLS + 1)
        )
        partials = (
            np.empty((n_workers, n_features, max_bin), dtype=np.float64),
            np.empty((n_workers, n_features, max_bin), dtype=np.float64),
            np.empty((n_workers, n_features, max_bin), dtype=np.int64),
        )
        partial_addresses = tuple(addr(array) for array in partials)
        histogram_function = lib().mlgb_histograms
        chunk_size = (len(indices) + n_workers - 1) // n_workers
        tasks = []
        for worker in range(n_workers):
            start = worker * chunk_size
            stop = min(start + chunk_size, len(indices))
            if start >= stop:
                break
            offset = worker * n_features * max_bin * 8
            tasks.append(
                (
                    histogram_function,
                    (
                        bins_addr, indices_addr + start * 8, grad_addr, hess_addr,
                        partial_addresses[0] + offset,
                        partial_addresses[1] + offset,
                        partial_addresses[2] + offset,
                        stop - start, n_features, max_bin,
                    ),
                )
            )
        tuple(
            _HISTOGRAM_EXECUTOR.map(_call_kernel, tasks)
        )
        lib().mlgb_histogram_reduce(
            partial_addresses[0], partial_addresses[1], partial_addresses[2],
            target.grad_addr, target.hess_addr, target.count_addr,
            len(tasks), n_features * max_bin,
        )
    else:
        lib().mlgb_histograms(
            bins_addr, indices_addr, grad_addr, hess_addr,
            target.grad_addr, target.hess_addr, target.count_addr,
            len(indices), n_features, max_bin,
        )
    return target


def _subtract_histograms(parent, child):
    lib().mlgb_histogram_subtract_inplace(
        parent.grad_addr, parent.hess_addr, parent.count_addr,
        child.grad_addr, child.hess_addr, child.count_addr,
        parent.n_features * parent.max_bin,
    )
    return parent


def _histogram_totals(histograms, result):
    lib().mlgb_histogram_totals(
        histograms.grad_addr, histograms.hess_addr, addr(result),
        histograms.max_bin,
    )
    return float(result[0]), float(result[1])


@dataclass(slots=True)
class _Candidate:
    gain: float
    feature: int
    threshold_bin: int
    histograms: _Histograms
    total_grad: float
    total_hess: float
    total_count: int
    left_grad: float
    left_hess: float
    left_count: int


def _candidate_from_histograms(
    histograms,
    indices,
    ncuts,
    max_bin,
    min_child_samples,
    min_child_weight,
    min_split_gain,
    reg_alpha,
    reg_lambda,
    total_grad,
    total_hess,
    result,
    ncuts_address=None,
    result_address=None,
):
    n_features = histograms.n_features
    lib().mlgb_best_split(
        histograms.grad_addr, histograms.hess_addr, histograms.count_addr,
        addr(ncuts) if ncuts_address is None else ncuts_address,
        addr(result) if result_address is None else result_address,
        n_features, max_bin, total_grad, total_hess, len(indices), min_child_samples,
        min_child_weight, min_split_gain, reg_alpha, reg_lambda,
    )
    if result[0] < 0:
        return None
    return _Candidate(
        float(result[2]),
        int(result[0]),
        int(result[1]),
        histograms,
        total_grad,
        total_hess,
        len(indices),
        float(result[3]),
        float(result[4]),
        int(result[5]),
    )


def _best_candidate(
    indices,
    bins,
    grad,
    hess,
    ncuts,
    max_bin,
    min_child_samples,
    min_child_weight,
    min_split_gain,
    reg_alpha,
    reg_lambda,
):
    histograms = _histograms(indices, bins, grad, hess, max_bin)
    result = np.empty(6, dtype=np.float64)
    total_grad, total_hess = _histogram_totals(histograms, result)
    candidate = _candidate_from_histograms(
        histograms, indices, ncuts, max_bin, min_child_samples,
        min_child_weight, min_split_gain, reg_alpha, reg_lambda,
        total_grad, total_hess, result,
    )
    if candidate is None:
        return None
    return candidate.gain, candidate.feature, candidate.threshold_bin


def _partition(
    indices,
    bins,
    feature,
    threshold_bin,
    bins_address=None,
    joined=None,
    joined_address=None,
    scratch=None,
    scratch_address=None,
):
    if joined is None:
        joined = np.empty(len(indices), dtype=np.int64)
    if scratch is None:
        scratch = np.empty(len(indices), dtype=np.int64)
    nleft = lib().mlgb_partition(
        addr(bins) if bins_address is None else bins_address,
        addr(indices), addr(joined) if joined_address is None else joined_address,
        addr(scratch) if scratch_address is None else scratch_address,
        len(indices),
        bins.shape[1], feature, threshold_bin,
    )
    return joined[:nleft], joined[nleft:]


def _fit_tree(x, bins, cuts, ncuts, grad, hess, params):
    max_bin = params["max_bin"]
    num_leaves = params["num_leaves"]
    max_depth = params["max_depth"]
    learning_rate = params["learning_rate"]
    reg_alpha = params["reg_alpha"]
    reg_lambda = params["reg_lambda"]
    min_child_samples = params["min_child_samples"]
    min_child_weight = params["min_child_weight"]
    min_split_gain = params["min_split_gain"]

    features = [-1]
    thresholds = [0.0]
    left = [0]
    right = [0]
    values = [0.0]
    leaf_ids = [0]
    gains = [0.0]
    node_indices = [np.arange(len(x), dtype=np.int64)]
    depths = [0]
    split_result = np.empty(6, dtype=np.float64)
    input_addresses = (addr(bins), addr(grad), addr(hess))
    ncuts_address = addr(ncuts)
    result_address = addr(split_result)
    workspace_leaves = min(
        num_leaves, max(1, len(x) // max(min_child_samples, 1))
    )
    histogram_workspace = _histogram_workspace(
        workspace_leaves, bins.shape[1], max_bin
    )
    next_histogram = 1
    if (workspace_leaves - 1) * len(x) <= 2_000_000:
        partition_workspace = np.empty(
            (workspace_leaves - 1, len(x)), dtype=np.int64
        )
        partition_scratch = np.empty_like(partition_workspace)
        partition_base = addr(partition_workspace)
        scratch_base = addr(partition_scratch)
    else:
        partition_workspace = None
        partition_scratch = None
        partition_base = 0
        scratch_base = 0
    partition_slot = 0
    root_histograms = _histograms(
        node_indices[0], bins, grad, hess, max_bin, input_addresses,
        histogram_workspace[0],
    )
    root_grad, root_hess = _histogram_totals(root_histograms, split_result)
    values[0] = _leaf_value(
        root_grad, root_hess, reg_alpha, reg_lambda, learning_rate
    )

    queue = []
    root_candidate = _candidate_from_histograms(
        root_histograms, node_indices[0], ncuts, max_bin, min_child_samples,
        min_child_weight, min_split_gain, reg_alpha, reg_lambda,
        root_grad, root_hess, split_result, ncuts_address, result_address,
    )
    if root_candidate is not None:
        heapq.heappush(queue, (-root_candidate.gain, 0, root_candidate))

    leaves = 1
    while queue and leaves < num_leaves:
        _, node, candidate = heapq.heappop(queue)
        gain = candidate.gain
        feature = candidate.feature
        threshold_bin = candidate.threshold_bin
        indices = node_indices[node]
        if partition_workspace is None:
            left_indices, right_indices = _partition(
                indices, bins, feature, threshold_bin, input_addresses[0]
            )
        else:
            joined = partition_workspace[partition_slot, : len(indices)]
            scratch = partition_scratch[partition_slot, : len(indices)]
            joined_address = partition_base + partition_slot * len(x) * 8
            scratch_address = scratch_base + partition_slot * len(x) * 8
            left_indices, right_indices = _partition(
                indices, bins, feature, threshold_bin, input_addresses[0],
                joined, joined_address, scratch, scratch_address,
            )
            partition_slot += 1
        if not len(left_indices) or not len(right_indices):
            continue
        left_node = len(features)
        right_node = left_node + 1
        features[node] = feature
        thresholds[node] = float(cuts[feature, threshold_bin])
        left[node] = left_node
        right[node] = right_node
        gains[node] = gain
        leaf_ids[node] = -1
        child_indices_list = (left_indices, right_indices)
        child_stats = (
            (
                candidate.left_grad,
                candidate.left_hess,
                candidate.left_count,
            ),
            (
                candidate.total_grad - candidate.left_grad,
                candidate.total_hess - candidate.left_hess,
                candidate.total_count - candidate.left_count,
            ),
        )
        for child_indices, (child_grad, child_hess, _) in zip(
            child_indices_list, child_stats
        ):
            features.append(-1)
            thresholds.append(0.0)
            left.append(0)
            right.append(0)
            values.append(
                _leaf_value(
                    child_grad, child_hess, reg_alpha, reg_lambda, learning_rate
                )
            )
            leaf_ids.append(-1)
            gains.append(0.0)
            node_indices.append(child_indices)
            depths.append(depths[node] + 1)
        leaves += 1

        eligible = []
        for side, child in enumerate((left_node, right_node)):
            child_grad, child_hess, child_count = child_stats[side]
            if (
                (max_depth <= 0 or depths[child] < max_depth)
                and child_count >= 2 * min_child_samples
                and child_hess >= 2.0 * min_child_weight
            ):
                eligible.append(side)

        child_histograms = {}
        if len(eligible) == 2:
            smaller = 0 if len(left_indices) <= len(right_indices) else 1
            child_histograms[smaller] = _histograms(
                child_indices_list[smaller], bins, grad, hess, max_bin,
                input_addresses, histogram_workspace[next_histogram],
            )
            next_histogram += 1
            other = 1 - smaller
            child_histograms[other] = _subtract_histograms(
                candidate.histograms, child_histograms[smaller]
            )
        elif eligible:
            side = eligible[0]
            child_histograms[side] = _histograms(
                child_indices_list[side], bins, grad, hess, max_bin,
                input_addresses, histogram_workspace[next_histogram],
            )
            next_histogram += 1

        for side in eligible:
            child = left_node + side
            child_grad, child_hess, _ = child_stats[side]
            child_candidate = _candidate_from_histograms(
                child_histograms[side], child_indices_list[side], ncuts, max_bin,
                min_child_samples, min_child_weight, min_split_gain,
                reg_alpha, reg_lambda, child_grad, child_hess, split_result,
                ncuts_address, result_address,
            )
            if child_candidate is not None:
                heapq.heappush(
                    queue, (-child_candidate.gain, child, child_candidate)
                )

    dense_leaf_id = 0
    for node, feature in enumerate(features):
        if feature < 0:
            leaf_ids[node] = dense_leaf_id
            dense_leaf_id += 1
    return Tree(
        i64(features), f64(thresholds), i64(left), i64(right), f64(values),
        i64(leaf_ids), f64(gains),
    )


def train(
    params,
    train_set,
    num_boost_round=100,
    valid_sets=None,
    valid_names=None,
    feval=None,
    init_model=None,
    keep_training_booster=False,
    callbacks=None,
):
    if not isinstance(train_set, Dataset):
        raise TypeError("train_set must be a Dataset")
    if train_set.label is None:
        raise ValueError("train_set must have labels")
    if init_model is not None:
        raise NotImplementedError("continued training is not covered")
    if callbacks:
        raise NotImplementedError("callbacks and early stopping are not covered")
    if valid_sets or feval is not None:
        raise NotImplementedError("validation sets and custom evaluation are not covered")
    merged = dict(train_set.params)
    merged.update({} if params is None else params)
    objective = merged.get("objective", "regression")
    if objective in ("regression", "regression_l2", "l2", "mse"):
        objective = "regression"
    elif objective in ("binary", "binary_logloss"):
        objective = "binary"
    else:
        raise NotImplementedError("covered objectives are regression_l2 and binary")
    if merged.get("boosting_type", merged.get("boosting", "gbdt")) not in ("gbdt", None):
        raise NotImplementedError("only gbdt boosting is covered")
    if train_set.weight is not None:
        raise NotImplementedError("sample weights are not covered")
    if train_set.group is not None:
        raise NotImplementedError("ranking groups are not covered")
    if train_set.init_score is not None:
        raise NotImplementedError("initial scores are not covered")

    x = f64(train_set.data)
    y = f64(train_set.label)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("data must be 2D and label must be a matching 1D array")
    if not len(x) or not x.shape[1]:
        raise ValueError("data must contain at least one row and one feature")
    if np.any(np.isinf(x)):
        raise ValueError("features may contain NaN but not infinity")
    if not np.all(np.isfinite(y)):
        raise ValueError("labels must be finite")
    if objective == "binary" and not np.array_equal(np.unique(y), [0.0, 1.0]):
        raise ValueError("binary labels must contain both 0 and 1")
    if not isinstance(num_boost_round, (int, np.integer)) or num_boost_round < 0:
        raise ValueError("num_boost_round must be a non-negative integer")
    cfg = {
        "max_bin": _integer_parameter("max_bin", merged.get("max_bin", 255)),
        "num_leaves": _integer_parameter(
            "num_leaves", merged.get("num_leaves", 31)
        ),
        "max_depth": _integer_parameter(
            "max_depth", merged.get("max_depth", -1)
        ),
        "learning_rate": float(merged.get("learning_rate", 0.1)),
        "reg_alpha": float(merged.get("reg_alpha", merged.get("lambda_l1", 0.0))),
        "reg_lambda": float(merged.get("reg_lambda", merged.get("lambda_l2", 0.0))),
        "min_child_samples": _integer_parameter(
            "min_child_samples",
            merged.get("min_child_samples", merged.get("min_data_in_leaf", 20)),
        ),
        "min_child_weight": float(merged.get("min_child_weight", merged.get("min_sum_hessian_in_leaf", 1e-3))),
        "min_split_gain": float(merged.get("min_split_gain", merged.get("min_gain_to_split", 0.0))),
    }
    if cfg["max_bin"] < 2 or cfg["num_leaves"] < 2:
        raise ValueError("max_bin and num_leaves must be at least 2")
    if cfg["min_child_samples"] < 1:
        raise ValueError("min_child_samples must be at least 1")
    if (
        not np.isfinite(cfg["learning_rate"])
        or cfg["learning_rate"] <= 0.0
        or not np.isfinite(cfg["min_child_weight"])
        or cfg["min_child_weight"] < 0.0
        or not np.isfinite(cfg["min_split_gain"])
        or cfg["min_split_gain"] < 0.0
        or not np.isfinite(cfg["reg_alpha"])
        or cfg["reg_alpha"] < 0.0
        or not np.isfinite(cfg["reg_lambda"])
        or cfg["reg_lambda"] < 0.0
    ):
        raise ValueError("learning rate and regularization parameters are invalid")

    booster = Booster(merged, train_set)
    booster.objective = objective
    booster.n_features = x.shape[1]
    if train_set.feature_name == "auto":
        booster.feature_names = [f"Column_{i}" for i in range(x.shape[1])]
    else:
        booster.feature_names = list(train_set.feature_name)
        if len(booster.feature_names) != x.shape[1]:
            raise ValueError("feature_name must contain one name per feature")
    if objective == "regression":
        booster.base_score = float(np.mean(y))
    else:
        positive = float(np.clip(np.mean(y), 1e-15, 1.0 - 1e-15))
        booster.base_score = float(np.log(positive / (1.0 - positive)))
    raw = np.full(len(x), booster.base_score, dtype=np.float64)
    grad = np.empty(len(x), dtype=np.float64)
    hess = np.empty(len(x), dtype=np.float64)
    cuts, ncuts = _cuts(x, cfg["max_bin"])
    bins = _quantize(x, cuts, ncuts, cfg["max_bin"])
    gradient_kernel = (
        lib().mlgb_regression_gradients
        if objective == "regression"
        else lib().mlgb_binary_gradients
    )

    for _ in range(num_boost_round):
        gradient_kernel(addr(raw), addr(y), addr(grad), addr(hess), len(x))
        tree = _fit_tree(x, bins, cuts, ncuts, grad, hess, cfg)
        booster.trees.append(tree)
        booster._flat_cache = None
        lib().mlgb_predict_add(
            addr(x), addr(tree.features), addr(tree.thresholds),
            addr(tree.left), addr(tree.right), addr(tree.values), addr(raw),
            len(x), x.shape[1],
        )
    booster.current_iteration_number = len(booster.trees)
    booster.best_iteration = len(booster.trees)
    return booster
