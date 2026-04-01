"""Unit tests for the Exact explainer."""

import pickle

from conftest import compare_numpy_outputs_against_baseline

import shap

from . import common


def test_tabular_simple_case():
    import pytest

    xgboost = pytest.importorskip("xgboost")
    sk = pytest.importorskip("sklearn")

    model = xgboost.XGBClassifier(tree_method="exact", base_score=0.5)
    X, y = sk.datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, return_X_y=True)

    X_train = X[:80]
    X_test = X[80:]
    y_train = y[:80]
    y_test = y[80:]
    model.fit(X_train, y_train)
    ex = shap.explainers.ExactExplainer(model.predict_proba, X_train)
    shap_values = ex(X_test)
    y_pred = model.predict(X_test)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_interactions():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_interactions_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_auto_masker_single_value():
    # This currently fails with an MemoryError, I assume due to having a different dimension than required!
    model, data = common.basic_xgboost_scenario(1)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_auto_masker_minimal():
    model, data = common.basic_xgboost_scenario(2)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_multi_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict_proba, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, shap.maskers.Partition(data), data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_multi_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(
        shap.explainers.ExactExplainer, model.predict_proba, shap.maskers.Partition(data), data
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, shap.maskers.Independent(data), data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_multi_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(
        shap.explainers.ExactExplainer, model.predict_proba, shap.maskers.Independent(data), data
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_serialization():
    model, data = common.basic_xgboost_scenario()
    return common.test_serialization(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_serialization_no_model_or_masker():
    model, data = common.basic_xgboost_scenario()
    return common.test_serialization(
        shap.explainers.ExactExplainer,
        model.predict,
        data,
        data,
        model_saver=False,
        masker_saver=False,
        model_loader=lambda _: model.predict,
        masker_loader=lambda _: data,
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_serialization_custom_model_save():
    model, data = common.basic_xgboost_scenario()
    return common.test_serialization(
        shap.explainers.ExactExplainer, model.predict, data, data, model_saver=pickle.dump, model_loader=pickle.load
    )
