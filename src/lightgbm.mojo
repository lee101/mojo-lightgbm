"""Compute kernels for histogram-based leaf-wise gradient boosting."""

from std.math import exp
from std.sys import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def soft_threshold(value: Float64, l1: Float64) -> Float64:
    if value > l1:
        return value - l1
    if value < -l1:
        return value + l1
    return 0.0


@export("mlgb_quantize")
def mlgb_quantize(
    x_addr: Int,
    cuts_addr: Int,
    ncuts_addr: Int,
    bins_addr: Int,
    n_rows: Int,
    n_features: Int,
    max_bin: Int,
) abi("C"):
    var x = fp(x_addr)
    var cuts = fp(cuts_addr)
    var ncuts = ip(ncuts_addr)
    var bins = ip(bins_addr)
    for row in range(n_rows):
        for feature in range(n_features):
            var value = x[row * n_features + feature]
            if value != value:
                bins[row * n_features + feature] = 0
                continue
            var lo = 0
            var hi = Int(ncuts[feature])
            var base = feature * (max_bin - 1)
            while lo < hi:
                var mid = lo + (hi - lo) // 2
                if value <= cuts[base + mid]:
                    hi = mid
                else:
                    lo = mid + 1
            bins[row * n_features + feature] = Int64(lo)


@export("mlgb_histograms")
def mlgb_histograms(
    bins_addr: Int,
    indices_addr: Int,
    grad_addr: Int,
    hess_addr: Int,
    hist_grad_addr: Int,
    hist_hess_addr: Int,
    hist_count_addr: Int,
    n_indices: Int,
    n_features: Int,
    max_bin: Int,
) abi("C"):
    var bins = ip(bins_addr)
    var indices = ip(indices_addr)
    var grad = fp(grad_addr)
    var hess = fp(hess_addr)
    var hist_grad = fp(hist_grad_addr)
    var hist_hess = fp(hist_hess_addr)
    var hist_count = ip(hist_count_addr)
    var hist_size = n_features * max_bin
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= hist_size:
        hist_grad.store(i, SIMD[DType.float64, W](0.0))
        hist_hess.store(i, SIMD[DType.float64, W](0.0))
        hist_count.store(i, SIMD[DType.int64, W](0))
        i += W
    while i < hist_size:
        hist_grad[i] = 0.0
        hist_hess[i] = 0.0
        hist_count[i] = 0
        i += 1

    for pos in range(n_indices):
        var row = Int(indices[pos])
        var g = grad[row]
        var h = hess[row]
        for feature in range(n_features):
            var bin_id = Int(bins[row * n_features + feature])
            var slot = feature * max_bin + bin_id
            hist_grad[slot] += g
            hist_hess[slot] += h
            hist_count[slot] += 1


@export("mlgb_histogram_reduce")
def mlgb_histogram_reduce(
    partial_grad_addr: Int,
    partial_hess_addr: Int,
    partial_count_addr: Int,
    hist_grad_addr: Int,
    hist_hess_addr: Int,
    hist_count_addr: Int,
    n_partials: Int,
    n: Int,
) abi("C"):
    var partial_grad = fp(partial_grad_addr)
    var partial_hess = fp(partial_hess_addr)
    var partial_count = ip(partial_count_addr)
    var hist_grad = fp(hist_grad_addr)
    var hist_hess = fp(hist_hess_addr)
    var hist_count = ip(hist_count_addr)
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        var grad_vector = SIMD[DType.float64, W](0.0)
        var hess_vector = SIMD[DType.float64, W](0.0)
        var count_vector = SIMD[DType.int64, W](0)
        for partial in range(n_partials):
            var offset = partial * n + i
            grad_vector += partial_grad.load[width=W](offset)
            hess_vector += partial_hess.load[width=W](offset)
            count_vector += partial_count.load[width=W](offset)
        hist_grad.store(i, grad_vector)
        hist_hess.store(i, hess_vector)
        hist_count.store(i, count_vector)
        i += W
    while i < n:
        var grad_sum = 0.0
        var hess_sum = 0.0
        var count_sum: Int64 = 0
        for partial in range(n_partials):
            var offset = partial * n + i
            grad_sum += partial_grad[offset]
            hess_sum += partial_hess[offset]
            count_sum += partial_count[offset]
        hist_grad[i] = grad_sum
        hist_hess[i] = hess_sum
        hist_count[i] = count_sum
        i += 1


@export("mlgb_histogram_subtract_inplace")
def mlgb_histogram_subtract_inplace(
    parent_grad_addr: Int,
    parent_hess_addr: Int,
    parent_count_addr: Int,
    child_grad_addr: Int,
    child_hess_addr: Int,
    child_count_addr: Int,
    n: Int,
) abi("C"):
    var parent_grad = fp(parent_grad_addr)
    var parent_hess = fp(parent_hess_addr)
    var parent_count = ip(parent_count_addr)
    var child_grad = fp(child_grad_addr)
    var child_hess = fp(child_hess_addr)
    var child_count = ip(child_count_addr)
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        parent_grad.store(
            i,
            parent_grad.load[width=W](i) - child_grad.load[width=W](i),
        )
        parent_hess.store(
            i,
            parent_hess.load[width=W](i) - child_hess.load[width=W](i),
        )
        parent_count.store(
            i,
            parent_count.load[width=W](i) - child_count.load[width=W](i),
        )
        i += W
    while i < n:
        parent_grad[i] -= child_grad[i]
        parent_hess[i] -= child_hess[i]
        parent_count[i] -= child_count[i]
        i += 1


@export("mlgb_histogram_totals")
def mlgb_histogram_totals(
    hist_grad_addr: Int,
    hist_hess_addr: Int,
    result_addr: Int,
    max_bin: Int,
) abi("C"):
    var hist_grad = fp(hist_grad_addr)
    var hist_hess = fp(hist_hess_addr)
    var result = fp(result_addr)
    comptime W = simd_width_of[DType.float64]()
    var grad_vector = SIMD[DType.float64, W](0.0)
    var hess_vector = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= max_bin:
        grad_vector += hist_grad.load[width=W](i)
        hess_vector += hist_hess.load[width=W](i)
        i += W
    var total_grad = Float64(grad_vector.reduce_add())
    var total_hess = Float64(hess_vector.reduce_add())
    while i < max_bin:
        total_grad += hist_grad[i]
        total_hess += hist_hess[i]
        i += 1
    result[0] = total_grad
    result[1] = total_hess


@export("mlgb_best_split")
def mlgb_best_split(
    hist_grad_addr: Int,
    hist_hess_addr: Int,
    hist_count_addr: Int,
    ncuts_addr: Int,
    result_addr: Int,
    n_features: Int,
    max_bin: Int,
    total_grad: Float64,
    total_hess: Float64,
    total_count: Int,
    min_child_samples: Int,
    min_child_weight: Float64,
    min_split_gain: Float64,
    reg_alpha: Float64,
    reg_lambda: Float64,
) abi("C"):
    var hist_grad = fp(hist_grad_addr)
    var hist_hess = fp(hist_hess_addr)
    var hist_count = ip(hist_count_addr)
    var ncuts = ip(ncuts_addr)
    var result = fp(result_addr)
    var parent_g = soft_threshold(total_grad, reg_alpha)
    var parent_score = parent_g * parent_g / (total_hess + reg_lambda)
    var best_gain = -1.0
    var best_feature = -1
    var best_bin = -1
    var best_left_grad = 0.0
    var best_left_hess = 0.0
    var best_left_count = 0
    for feature in range(n_features):
        var left_grad = 0.0
        var left_hess = 0.0
        var left_count = 0
        var limit = Int(ncuts[feature])
        for bin_id in range(limit):
            var slot = feature * max_bin + bin_id
            left_grad += hist_grad[slot]
            left_hess += hist_hess[slot]
            left_count += Int(hist_count[slot])
            var right_count = total_count - left_count
            var right_hess = total_hess - left_hess
            if (
                left_count < min_child_samples
                or right_count < min_child_samples
                or left_hess < min_child_weight
                or right_hess < min_child_weight
            ):
                continue
            var right_grad = total_grad - left_grad
            var lg = soft_threshold(left_grad, reg_alpha)
            var rg = soft_threshold(right_grad, reg_alpha)
            var gain = (
                lg * lg / (left_hess + reg_lambda)
                + rg * rg / (right_hess + reg_lambda)
                - parent_score
            )
            if gain > best_gain and gain > min_split_gain:
                best_gain = gain
                best_feature = feature
                best_bin = bin_id
                best_left_grad = left_grad
                best_left_hess = left_hess
                best_left_count = left_count
    result[0] = Float64(best_feature)
    result[1] = Float64(best_bin)
    result[2] = best_gain
    result[3] = best_left_grad
    result[4] = best_left_hess
    result[5] = Float64(best_left_count)


@export("mlgb_best_split_pair")
def mlgb_best_split_pair(
    hist0_grad_addr: Int,
    hist0_hess_addr: Int,
    hist0_count_addr: Int,
    hist1_grad_addr: Int,
    hist1_hess_addr: Int,
    hist1_count_addr: Int,
    ncuts_addr: Int,
    result_addr: Int,
    n_features: Int,
    max_bin: Int,
    total0_grad: Float64,
    total0_hess: Float64,
    total0_count: Int,
    total1_grad: Float64,
    total1_hess: Float64,
    total1_count: Int,
    min_child_samples: Int,
    min_child_weight: Float64,
    min_split_gain: Float64,
    reg_alpha: Float64,
    reg_lambda: Float64,
) abi("C"):
    mlgb_best_split(
        hist0_grad_addr, hist0_hess_addr, hist0_count_addr, ncuts_addr,
        result_addr, n_features, max_bin, total0_grad, total0_hess,
        total0_count, min_child_samples, min_child_weight, min_split_gain,
        reg_alpha, reg_lambda,
    )
    mlgb_best_split(
        hist1_grad_addr, hist1_hess_addr, hist1_count_addr, ncuts_addr,
        result_addr + 6 * 8, n_features, max_bin, total1_grad, total1_hess,
        total1_count, min_child_samples, min_child_weight, min_split_gain,
        reg_alpha, reg_lambda,
    )


@export("mlgb_partition")
def mlgb_partition(
    bins_addr: Int,
    indices_addr: Int,
    joined_addr: Int,
    scratch_addr: Int,
    n_indices: Int,
    n_features: Int,
    feature: Int,
    threshold_bin: Int,
) abi("C") -> Int:
    var bins = ip(bins_addr)
    var indices = ip(indices_addr)
    var joined = ip(joined_addr)
    var scratch = ip(scratch_addr)
    var nleft = 0
    var nright = 0
    for pos in range(n_indices):
        var row = Int(indices[pos])
        if Int(bins[row * n_features + feature]) <= threshold_bin:
            joined[nleft] = Int64(row)
            nleft += 1
        else:
            scratch[nright] = Int64(row)
            nright += 1
    comptime W = simd_width_of[DType.int64]()
    var i = 0
    while i + W <= nright:
        joined.store(nleft + i, scratch.load[width=W](i))
        i += W
    while i < nright:
        joined[nleft + i] = scratch[i]
        i += 1
    return nleft


@export("mlgb_partition_histogram_left")
def mlgb_partition_histogram_left(
    bins_addr: Int,
    indices_addr: Int,
    grad_addr: Int,
    hess_addr: Int,
    joined_addr: Int,
    scratch_addr: Int,
    hist_grad_addr: Int,
    hist_hess_addr: Int,
    hist_count_addr: Int,
    n_indices: Int,
    n_features: Int,
    max_bin: Int,
    feature: Int,
    threshold_bin: Int,
) abi("C") -> Int:
    var bins = ip(bins_addr)
    var indices = ip(indices_addr)
    var grad = fp(grad_addr)
    var hess = fp(hess_addr)
    var joined = ip(joined_addr)
    var scratch = ip(scratch_addr)
    var hist_grad = fp(hist_grad_addr)
    var hist_hess = fp(hist_hess_addr)
    var hist_count = ip(hist_count_addr)
    var hist_size = n_features * max_bin
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= hist_size:
        hist_grad.store(i, SIMD[DType.float64, W](0.0))
        hist_hess.store(i, SIMD[DType.float64, W](0.0))
        hist_count.store(i, SIMD[DType.int64, W](0))
        i += W
    while i < hist_size:
        hist_grad[i] = 0.0
        hist_hess[i] = 0.0
        hist_count[i] = 0
        i += 1

    var nleft = 0
    var nright = 0
    for pos in range(n_indices):
        var row = Int(indices[pos])
        if Int(bins[row * n_features + feature]) <= threshold_bin:
            joined[nleft] = Int64(row)
            nleft += 1
            var g = grad[row]
            var h = hess[row]
            for histogram_feature in range(n_features):
                var bin_id = Int(
                    bins[row * n_features + histogram_feature]
                )
                var slot = histogram_feature * max_bin + bin_id
                hist_grad[slot] += g
                hist_hess[slot] += h
                hist_count[slot] += 1
        else:
            scratch[nright] = Int64(row)
            nright += 1
    i = 0
    while i + W <= nright:
        joined.store(nleft + i, scratch.load[width=W](i))
        i += W
    while i < nright:
        joined[nleft + i] = scratch[i]
        i += 1
    return nleft


@export("mlgb_regression_gradients")
def mlgb_regression_gradients(
    raw_addr: Int, label_addr: Int, grad_addr: Int, hess_addr: Int, n: Int
) abi("C"):
    var raw = fp(raw_addr)
    var label = fp(label_addr)
    var grad = fp(grad_addr)
    var hess = fp(hess_addr)
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        grad.store(
            i,
            raw.load[width=W](i) - label.load[width=W](i),
        )
        hess.store(i, SIMD[DType.float64, W](1.0))
        i += W
    while i < n:
        grad[i] = raw[i] - label[i]
        hess[i] = 1.0
        i += 1


@export("mlgb_binary_gradients")
def mlgb_binary_gradients(
    raw_addr: Int, label_addr: Int, grad_addr: Int, hess_addr: Int, n: Int
) abi("C"):
    var raw = fp(raw_addr)
    var label = fp(label_addr)
    var grad = fp(grad_addr)
    var hess = fp(hess_addr)
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        var z = raw.load[width=W](i)
        var clipped = min(max(z, -709.0), 709.0)
        var probability = 1.0 / (1.0 + exp(-clipped))
        grad.store(i, probability - label.load[width=W](i))
        hess.store(i, probability * (1.0 - probability))
        i += W
    while i < n:
        var z = raw[i]
        var probability: Float64
        if z >= 0.0:
            probability = 1.0 / (1.0 + exp(-z))
        else:
            var ez = exp(z)
            probability = ez / (1.0 + ez)
        grad[i] = probability - label[i]
        hess[i] = probability * (1.0 - probability)
        i += 1


@export("mlgb_predict")
def mlgb_predict(
    x_addr: Int,
    tree_offsets_addr: Int,
    features_addr: Int,
    thresholds_addr: Int,
    left_addr: Int,
    right_addr: Int,
    values_addr: Int,
    dst_addr: Int,
    n_rows: Int,
    n_features: Int,
    n_trees: Int,
    base_score: Float64,
) abi("C"):
    var x = fp(x_addr)
    var tree_offsets = ip(tree_offsets_addr)
    var features = ip(features_addr)
    var thresholds = fp(thresholds_addr)
    var left = ip(left_addr)
    var right = ip(right_addr)
    var values = fp(values_addr)
    var dst = fp(dst_addr)
    for row in range(n_rows):
        var prediction = base_score
        for tree in range(n_trees):
            var node = Int(tree_offsets[tree])
            while Int(features[node]) >= 0:
                var feature = Int(features[node])
                var value = x[row * n_features + feature]
                if value != value or value <= thresholds[node]:
                    node = Int(left[node])
                else:
                    node = Int(right[node])
            prediction += values[node]
        dst[row] = prediction


@export("mlgb_predict_add")
def mlgb_predict_add(
    x_addr: Int,
    features_addr: Int,
    thresholds_addr: Int,
    left_addr: Int,
    right_addr: Int,
    values_addr: Int,
    dst_addr: Int,
    n_rows: Int,
    n_features: Int,
) abi("C"):
    var x = fp(x_addr)
    var features = ip(features_addr)
    var thresholds = fp(thresholds_addr)
    var left = ip(left_addr)
    var right = ip(right_addr)
    var values = fp(values_addr)
    var dst = fp(dst_addr)
    for row in range(n_rows):
        var node = 0
        while Int(features[node]) >= 0:
            var feature = Int(features[node])
            var value = x[row * n_features + feature]
            if value != value or value <= thresholds[node]:
                node = Int(left[node])
            else:
                node = Int(right[node])
        dst[row] += values[node]


@export("mlgb_predict_leaf")
def mlgb_predict_leaf(
    x_addr: Int,
    tree_offsets_addr: Int,
    features_addr: Int,
    thresholds_addr: Int,
    left_addr: Int,
    right_addr: Int,
    leaf_ids_addr: Int,
    dst_addr: Int,
    n_rows: Int,
    n_features: Int,
    n_trees: Int,
) abi("C"):
    var x = fp(x_addr)
    var tree_offsets = ip(tree_offsets_addr)
    var features = ip(features_addr)
    var thresholds = fp(thresholds_addr)
    var left = ip(left_addr)
    var right = ip(right_addr)
    var leaf_ids = ip(leaf_ids_addr)
    var dst = ip(dst_addr)
    for row in range(n_rows):
        for tree in range(n_trees):
            var node = Int(tree_offsets[tree])
            while Int(features[node]) >= 0:
                var feature = Int(features[node])
                var value = x[row * n_features + feature]
                if value != value or value <= thresholds[node]:
                    node = Int(left[node])
                else:
                    node = Int(right[node])
            dst[row * n_trees + tree] = leaf_ids[node]
