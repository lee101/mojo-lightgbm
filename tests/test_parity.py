import numpy as np
import pytest

import lightgbm as upstream
import mojo_lightgbm as mlgb


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1000, 8))
    y = (
        3.0 * x[:, 0]
        - 2.0 * x[:, 1]
        + x[:, 2] * x[:, 3]
        + rng.normal(scale=0.2, size=len(x))
    )
    return x, y


@pytest.fixture(scope="module")
def fit_params():
    return {
        "n_estimators": 50,
        "num_leaves": 15,
        "max_bin": 63,
        "learning_rate": 0.08,
        "min_child_samples": 10,
        "verbosity": -1,
    }


def test_regressor_behavioral_parity(regression_data, fit_params):
    x, y = regression_data
    ours = mlgb.LGBMRegressor(**fit_params).fit(x, y)
    theirs = upstream.LGBMRegressor(**fit_params).fit(x, y)
    ours_prediction = ours.predict(x)
    their_prediction = theirs.predict(x)
    assert ours.score(x, y) > 0.96
    assert abs(ours.score(x, y) - theirs.score(x, y)) < 0.02
    assert np.corrcoef(ours_prediction, their_prediction)[0, 1] > 0.99
    assert ours.n_estimators_ == fit_params["n_estimators"]
    assert ours.n_features_in_ == x.shape[1]


def test_classifier_behavioral_parity(regression_data, fit_params):
    x, continuous = regression_data
    y = np.where(continuous > np.median(continuous), "positive", "negative")
    ours = mlgb.LGBMClassifier(**fit_params).fit(x, y)
    theirs = upstream.LGBMClassifier(**fit_params).fit(x, y)
    ours_prediction = ours.predict(x)
    their_prediction = theirs.predict(x)
    ours_probability = ours.predict_proba(x)
    their_probability = theirs.predict_proba(x)
    assert ours.score(x, y) > 0.97
    assert np.mean(ours_prediction == their_prediction) > 0.98
    assert np.corrcoef(ours_probability[:, 1], their_probability[:, 1])[0, 1] > 0.99
    assert np.allclose(ours_probability.sum(axis=1), 1.0)
    assert np.array_equal(ours.classes_, theirs.classes_)


def test_dataset_train_and_booster_api(regression_data):
    x, y = regression_data
    dataset = mlgb.Dataset(x, label=y, feature_name=[f"f{i}" for i in range(x.shape[1])])
    booster = mlgb.train(
        {
            "objective": "regression_l2",
            "num_leaves": 7,
            "max_bin": 31,
            "min_data_in_leaf": 10,
        },
        dataset,
        num_boost_round=8,
    )
    assert dataset.construct() is dataset
    assert dataset.get_data() is x
    assert dataset.get_label() is y
    assert dataset.get_weight() is None
    assert dataset.num_data() == len(x)
    assert dataset.num_feature() == x.shape[1]
    assert booster.current_iteration() == 8
    assert booster.num_trees() == 8
    assert booster.num_feature() == x.shape[1]
    assert booster.feature_name() == [f"f{i}" for i in range(x.shape[1])]
    assert booster.predict(x[:7]).shape == (7,)


def test_raw_score_and_leaf_prediction(regression_data, fit_params):
    x, y = regression_data
    labels = y > np.median(y)
    model = mlgb.LGBMClassifier(**fit_params).fit(x, labels)
    raw = model.predict(x[:20], raw_score=True)
    probability = model.predict_proba(x[:20])[:, 1]
    assert np.allclose(probability, 1.0 / (1.0 + np.exp(-raw)))
    leaves = model.predict(x[:20], pred_leaf=True)
    assert leaves.shape == (20, fit_params["n_estimators"])
    assert np.all(leaves >= 0)
    for tree in range(leaves.shape[1]):
        assert np.max(leaves[:, tree]) < fit_params["num_leaves"]


def test_leafwise_tree_limits_and_dump(regression_data):
    x, y = regression_data
    model = mlgb.LGBMRegressor(
        n_estimators=3,
        num_leaves=9,
        max_depth=2,
        max_bin=31,
        min_child_samples=5,
    ).fit(x, y)
    dumped = model.booster_.dump_model()
    assert dumped["num_trees"] == 3
    for info in dumped["tree_info"]:
        assert info["num_leaves"] <= 4
    split = model.booster_.feature_importance("split")
    gain = model.booster_.feature_importance("gain")
    assert split.dtype == np.int32
    assert split.sum() > 0
    assert np.all(gain >= 0.0)
    first_leaf = model.predict(x[:1], pred_leaf=True)[0, 0]
    assert np.isfinite(model.booster_.get_leaf_output(0, first_leaf))


def test_constant_target_is_stable():
    x = np.arange(120, dtype=np.float64).reshape(40, 3)
    y = np.full(40, 7.5)
    model = mlgb.LGBMRegressor(n_estimators=5, min_child_samples=5).fit(x, y)
    assert np.allclose(model.predict(x), y)


def test_zero_max_depth_means_unlimited_and_regularization_is_applied(regression_data):
    x, y = regression_data
    plain = mlgb.LGBMRegressor(
        n_estimators=1, max_depth=0, num_leaves=5, min_child_samples=5
    ).fit(x, y)
    regularized = mlgb.LGBMRegressor(
        n_estimators=1,
        max_depth=0,
        num_leaves=5,
        min_child_samples=5,
        reg_alpha=10.0,
        reg_lambda=10.0,
    ).fit(x, y)
    assert plain.booster_.dump_model()["tree_info"][0]["num_leaves"] > 1
    assert not np.allclose(plain.predict(x), regularized.predict(x))


def test_sklearn_parameter_roundtrip():
    model = mlgb.LGBMRegressor(num_leaves=7, n_estimators=4, max_bin=15)
    assert model.get_params()["num_leaves"] == 7
    assert model.set_params(num_leaves=9, min_data_in_leaf=3) is model
    assert model.get_params()["num_leaves"] == 9
    assert model.get_params()["min_data_in_leaf"] == 3


def test_unsupported_modes_fail_explicitly(regression_data):
    x, y = regression_data
    with pytest.raises(NotImplementedError, match="sampling"):
        mlgb.LGBMRegressor(subsample=0.8)
    with pytest.raises(NotImplementedError, match="objectives"):
        mlgb.train({"objective": "multiclass"}, mlgb.Dataset(x, y), 2)
    with pytest.raises(NotImplementedError, match="validation"):
        mlgb.LGBMRegressor().fit(x, y, eval_set=[(x, y)])
    with pytest.raises(NotImplementedError, match="categorical"):
        mlgb.LGBMRegressor().fit(x, y, categorical_feature=[0])


@pytest.mark.parametrize(
    ("data", "label", "params", "message"),
    [
        (np.empty((0, 2)), np.empty(0), {}, "at least one row"),
        (np.ones((4, 2)), np.ones(4), {"objective": "binary"}, "0 and 1"),
        (np.array([[1.0, np.inf]]), np.array([1.0]), {}, "not infinity"),
        (np.ones((4, 2)), np.ones(4), {"learning_rate": 0.0}, "invalid"),
    ],
)
def test_invalid_training_inputs_stop_before_ffi(data, label, params, message):
    with pytest.raises(ValueError, match=message):
        mlgb.train(params, mlgb.Dataset(data, label), 1)


def test_integer_parameters_are_not_silently_narrowed():
    with pytest.raises(TypeError, match="max_bin must be an integer"):
        mlgb.train(
            {"max_bin": 31.5},
            mlgb.Dataset(np.ones((4, 2)), np.arange(4.0)),
            1,
        )
