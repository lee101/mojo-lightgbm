from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._lib import addr, f64, i64, lib


class LightGBMError(RuntimeError):
    pass


class Dataset:
    def __init__(
        self,
        data,
        label=None,
        reference=None,
        weight=None,
        group=None,
        init_score=None,
        feature_name="auto",
        categorical_feature="auto",
        params=None,
        free_raw_data=True,
        position=None,
    ):
        if categorical_feature not in ("auto", None, []):
            raise NotImplementedError("categorical features are not covered")
        self.data = data
        self.label = label
        self.reference = reference
        self.weight = weight
        self.group = group
        self.init_score = init_score
        self.feature_name = feature_name
        self.params = {} if params is None else dict(params)
        self.free_raw_data = free_raw_data
        self.position = position

    def construct(self):
        return self

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label

    def get_weight(self):
        return self.weight

    def num_data(self):
        return len(self.data)

    def num_feature(self):
        return np.asarray(self.data).shape[1]


@dataclass
class Tree:
    features: np.ndarray
    thresholds: np.ndarray
    left: np.ndarray
    right: np.ndarray
    values: np.ndarray
    leaf_ids: np.ndarray
    split_gains: np.ndarray


class Booster:
    def __init__(self, params=None, train_set=None, model_file=None, model_str=None):
        if model_file is not None or model_str is not None:
            raise NotImplementedError("model deserialization is not covered")
        self.params = {} if params is None else dict(params)
        self.train_set = train_set
        self.trees: list[Tree] = []
        self.base_score = 0.0
        self.objective = "regression"
        self.n_features = 0
        self.feature_names: list[str] = []
        self.best_iteration = 0
        self.current_iteration_number = 0
        self._flat_cache: tuple[np.ndarray, ...] | None = None

    def current_iteration(self):
        return self.current_iteration_number

    def num_trees(self):
        return len(self.trees)

    def num_feature(self):
        return self.n_features

    def feature_name(self):
        return list(self.feature_names)

    def _flatten(self, num_iteration=None):
        count = len(self.trees) if num_iteration in (None, 0) else min(num_iteration, len(self.trees))
        if self._flat_cache is not None and count == len(self.trees):
            return self._flat_cache
        offsets = []
        features = []
        thresholds = []
        left = []
        right = []
        values = []
        leaf_ids = []
        offset = 0
        for tree in self.trees[:count]:
            offsets.append(offset)
            features.extend(tree.features)
            thresholds.extend(tree.thresholds)
            left.extend(tree.left + offset)
            right.extend(tree.right + offset)
            values.extend(tree.values)
            leaf_ids.extend(tree.leaf_ids)
            offset += len(tree.features)
        packed = (
            i64(offsets),
            i64(features),
            f64(thresholds),
            i64(left),
            i64(right),
            f64(values),
            i64(leaf_ids),
        )
        if count == len(self.trees):
            self._flat_cache = packed
        return packed

    def predict(
        self,
        data,
        start_iteration=0,
        num_iteration=None,
        raw_score=False,
        pred_leaf=False,
        pred_contrib=False,
        data_has_header=False,
        validate_features=False,
        **kwargs,
    ):
        if start_iteration not in (0, None):
            raise NotImplementedError("start_iteration other than zero is not covered")
        if pred_contrib:
            raise NotImplementedError("SHAP contributions are not covered")
        x = f64(data)
        if x.ndim == 1:
            x = np.ascontiguousarray(x.reshape(1, -1))
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected a 2D array with {self.n_features} features")
        if np.any(np.isinf(x)):
            raise ValueError("features may contain NaN but not infinity")
        packed = self._flatten(num_iteration)
        n_trees = len(packed[0])
        if pred_leaf:
            result = np.empty((len(x), n_trees), dtype=np.int64)
            if n_trees:
                lib().mlgb_predict_leaf(
                    addr(x), *(addr(a) for a in packed[:5]), addr(packed[6]),
                    addr(result), len(x), self.n_features, n_trees,
                )
            return result
        result = np.empty(len(x), dtype=np.float64)
        if n_trees:
            lib().mlgb_predict(
                addr(x), *(addr(a) for a in packed[:6]), addr(result),
                len(x), self.n_features, n_trees, self.base_score,
            )
        else:
            result.fill(self.base_score)
        if raw_score or self.objective == "regression":
            return result
        return 1.0 / (1.0 + np.exp(-np.clip(result, -709.0, 709.0)))

    def feature_importance(self, importance_type="split", iteration=None):
        result = np.zeros(self.n_features, dtype=np.float64)
        for tree in self.trees[: iteration or len(self.trees)]:
            for feature, gain in zip(tree.features, tree.split_gains):
                if feature >= 0:
                    result[feature] += 1.0 if importance_type == "split" else gain
        if importance_type == "split":
            return result.astype(np.int32)
        if importance_type != "gain":
            raise ValueError("importance_type must be 'split' or 'gain'")
        return result

    def get_leaf_output(self, tree_id, leaf_id):
        tree = self.trees[tree_id]
        matches = np.flatnonzero(tree.leaf_ids == leaf_id)
        if not len(matches):
            raise LightGBMError("leaf index is out of range")
        return float(tree.values[matches[0]])

    def dump_model(self, num_iteration=None, **kwargs):
        trees = []
        for tree in self.trees[: num_iteration or len(self.trees)]:
            def node(index):
                feature = int(tree.features[index])
                if feature < 0:
                    return {
                        "leaf_index": int(tree.leaf_ids[index]),
                        "leaf_value": float(tree.values[index]),
                    }
                return {
                    "split_feature": feature,
                    "threshold": float(tree.thresholds[index]),
                    "split_gain": float(tree.split_gains[index]),
                    "left_child": node(int(tree.left[index])),
                    "right_child": node(int(tree.right[index])),
                }
            trees.append({"num_leaves": int(np.sum(tree.features < 0)), "tree_structure": node(0)})
        return {
            "name": "tree",
            "version": "mojo-lightgbm",
            "num_tree_per_iteration": 1,
            "num_trees": len(trees),
            "feature_names": self.feature_name(),
            "tree_info": trees,
        }

    def model_to_string(self, *args, **kwargs):
        raise NotImplementedError("LightGBM text model serialization is not covered")

    def save_model(self, *args, **kwargs):
        raise NotImplementedError("LightGBM text model serialization is not covered")
