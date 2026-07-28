from __future__ import annotations

import numpy as np

from .basic import Dataset
from .engine import train


class LGBMModel:
    def __init__(
        self,
        *,
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=1e-3,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=None,
        importance_type="split",
        **kwargs,
    ):
        if class_weight is not None or subsample != 1.0 or colsample_bytree != 1.0:
            raise NotImplementedError("class weights and row/feature sampling are not covered")
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.objective = objective
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_type = importance_type
        self.kwargs = dict(kwargs)
        self._Booster = None

    def get_params(self, deep=True):
        result = {
            "boosting_type": self.boosting_type,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample_for_bin": self.subsample_for_bin,
            "objective": self.objective,
            "min_split_gain": self.min_split_gain,
            "min_child_weight": self.min_child_weight,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "importance_type": self.importance_type,
        }
        result.update(self.kwargs)
        return result

    def set_params(self, **params):
        for name, value in params.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                self.kwargs[name] = value
        return self

    def _fit(self, x, y, objective, feature_name="auto", categorical_feature="auto"):
        params = self.get_params()
        params["objective"] = objective
        params.update(self.kwargs)
        self._Booster = train(
            params,
            Dataset(
                x,
                label=y,
                feature_name=feature_name,
                categorical_feature=categorical_feature,
            ),
            self.n_estimators,
        )
        self.booster_ = self._Booster
        self.n_features_in_ = np.asarray(x).shape[1]
        self.n_estimators_ = self._Booster.num_trees()
        self.feature_name_ = self._Booster.feature_name()
        self.feature_importances_ = self._Booster.feature_importance(self.importance_type)
        return self

    def predict(self, x, raw_score=False, pred_leaf=False, **kwargs):
        if self._Booster is None:
            raise ValueError("estimator is not fitted")
        return self._Booster.predict(x, raw_score=raw_score, pred_leaf=pred_leaf, **kwargs)


class LGBMRegressor(LGBMModel):
    def fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_init_score=None,
        eval_metric=None,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ):
        if any(value is not None for value in (sample_weight, init_score, callbacks, init_model)):
            raise NotImplementedError("weights, init scores, callbacks, and continued training are not covered")
        if eval_set is not None:
            raise NotImplementedError("validation sets and evaluation metrics are not covered")
        return self._fit(
            X, y, self.objective or "regression", feature_name, categorical_feature
        )

    def score(self, X, y):
        y = np.asarray(y)
        residual = np.sum((y - self.predict(X)) ** 2)
        total = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - residual / total)


class LGBMClassifier(LGBMModel):
    def fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_class_weight=None,
        eval_init_score=None,
        eval_metric=None,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ):
        if any(value is not None for value in (sample_weight, init_score, callbacks, init_model)):
            raise NotImplementedError("weights, init scores, callbacks, and continued training are not covered")
        if eval_set is not None:
            raise NotImplementedError("validation sets and evaluation metrics are not covered")
        classes = np.unique(y)
        if len(classes) != 2:
            raise NotImplementedError("only binary classification is covered")
        self.classes_ = classes
        encoded = (np.asarray(y) == classes[1]).astype(np.float64)
        self._fit(
            X,
            encoded,
            self.objective or "binary",
            feature_name,
            categorical_feature,
        )
        self.n_classes_ = 2
        return self

    def predict_proba(self, X, raw_score=False, pred_leaf=False, **kwargs):
        positive = LGBMModel.predict(
            self, X, raw_score=raw_score, pred_leaf=pred_leaf, **kwargs
        )
        if raw_score or pred_leaf:
            return positive
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X, raw_score=False, pred_leaf=False, **kwargs):
        result = super().predict(X, raw_score=raw_score, pred_leaf=pred_leaf, **kwargs)
        if raw_score or pred_leaf:
            return result
        return self.classes_[(result >= 0.5).astype(np.int64)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
