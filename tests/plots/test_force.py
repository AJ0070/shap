from contextlib import nullcontext as does_not_raise
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import param

import shap


@pytest.fixture
def data_explainer_shap_values():
    RandomForestRegressor = pytest.importorskip("sklearn.ensemble").RandomForestRegressor

    # train model
    X, y = shap.datasets.california(n_points=500)
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    return X, explainer, explainer.shap_values(X)


@pytest.mark.parametrize(
    "cmap, exp_ctx",
    [
        # Valid cmaps
        param("coolwarm", does_not_raise(), id="valid-str"),
        param(["#000000", "#ffffff"], does_not_raise(), id="valid-list[str]"),
        # Invalid cmaps
        param(
            777,
            pytest.raises(TypeError, match="Plot color map must be string or list!"),
            id="invalid-dtype1",
        ),
        param(
            [],
            pytest.raises(ValueError, match="Color map must be at least two colors"),
            id="invalid-insufficient-colors1",
        ),
        param(
            ["#8834BB"],
            pytest.raises(ValueError, match="Color map must be at least two colors"),
            id="invalid-insufficient-colors2",
        ),
        param(
            ["#883488", "#Gg8888"],
            pytest.raises(ValueError, match=r"Invalid color .+ found in cmap"),
            id="invalid-hexcolor-in-list1",
        ),
        param(
            ["#883488", "#1111119"],
            pytest.raises(ValueError, match=r"Invalid color .+ found in cmap"),
            id="invalid-hexcolor-in-list2",
        ),
    ],
)
def test_verify_valid_cmap(cmap, exp_ctx):
    from shap.plots._force import verify_valid_cmap

    with exp_ctx:
        verify_valid_cmap(cmap)


def test_random_force_plot_mpl_with_data(data_explainer_shap_values):
    """Test if force plot with matplotlib works."""
    X, explainer, shap_values = data_explainer_shap_values

    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False)
    with pytest.raises(TypeError, match="force plot now requires the base value as the first parameter"):
        shap.force_plot([1, 1], shap_values, X.iloc[0, :], show=False)
    plt.close("all")


def test_random_force_plot_mpl_text_rotation_with_data(data_explainer_shap_values):
    """Test if force plot with matplotlib works when supplied with text_rotation."""
    X, explainer, shap_values = data_explainer_shap_values

    # visualize the first prediction's explanation
    shap.force_plot(
        explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, text_rotation=30, show=False
    )
    plt.close("all")


def test_force_plot_mpl_accepts_and_returns_ax(data_explainer_shap_values):
    """Force plot should draw on the passed axis and return it when show=False."""
    X, explainer, shap_values = data_explainer_shap_values

    fig, ax = plt.subplots(figsize=(8, 2))
    returned_ax = shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        X.iloc[0, :],
        matplotlib=True,
        show=False,
        ax=ax,
    )

    assert returned_ax is ax
    plt.close(fig)


def test_force_plot_mpl_respects_existing_figure(data_explainer_shap_values):
    """When ax is omitted, force plot should use the current figure."""
    X, explainer, shap_values = data_explainer_shap_values

    fig = plt.figure(figsize=(8, 2))
    returned_ax = shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        X.iloc[0, :],
        matplotlib=True,
        show=False,
    )

    assert returned_ax.figure is fig
    plt.close(fig)


def test_force_plot_mpl_explanation_interface_returns_ax(data_explainer_shap_values):
    """Force plot should support Explanation input directly and return the passed axis."""
    X, explainer, _ = data_explainer_shap_values
    explanation = explainer(X)

    fig, ax = plt.subplots(figsize=(8, 2))
    returned_ax = shap.force_plot(
        explanation[0],
        matplotlib=True,
        show=False,
        ax=ax,
    )

    assert returned_ax is ax
    plt.close(fig)


def test_force_plot_multiple_samples_returns_array_visualizer(data_explainer_shap_values):
    """Multiple rows should produce an AdditiveForceArrayVisualizer in JS mode."""
    X, explainer, shap_values = data_explainer_shap_values

    vis = shap.force_plot(
        explainer.expected_value,
        shap_values[:5, :],
        X.iloc[:5, :],
        matplotlib=False,
    )

    from shap.plots._force import AdditiveForceArrayVisualizer

    assert isinstance(vis, AdditiveForceArrayVisualizer)
    assert "AdditiveForceArrayVisualizer" in vis.html()


def test_force_plot_multiple_samples_mpl_not_supported(data_explainer_shap_values):
    """Matplotlib mode is not implemented for stacked force plots."""
    X, explainer, shap_values = data_explainer_shap_values

    with pytest.raises(NotImplementedError, match="not yet supported"):
        shap.force_plot(
            explainer.expected_value,
            shap_values[:5, :],
            X.iloc[:5, :],
            matplotlib=True,
            show=False,
        )


def test_force_plot_rejects_list_shap_values():
    """Legacy list-shaped multi-output input should raise a helpful error."""
    with pytest.raises(TypeError, match="looks multi output"):
        shap.force_plot(0.0, [np.array([1.0, -1.0])])


def test_force_plot_base_value_shape_validation():
    """Inconsistent base_value/shap_values shapes should raise an error."""
    with pytest.raises(TypeError, match="force plot now requires the base value"):
        shap.force_plot(np.array([0.0, 1.0]), np.array([0.3]))


def test_force_save_html_roundtrip(data_explainer_shap_values):
    """save_html should serialize a visualizer into a complete HTML document."""
    X, explainer, shap_values = data_explainer_shap_values
    vis = shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=False)

    from shap.plots import _force as force_mod

    with TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "force_plot.html"
        force_mod.save_html(str(out_file), vis, full_html=True)
        html = out_file.read_text(encoding="utf-8")

    assert "<html>" in html
    assert "AdditiveForceVisualizer" in html


def test_force_save_html_requires_visualizer():
    """save_html should reject inputs that are not force visualizers."""
    from shap.plots import _force as force_mod

    with pytest.raises(TypeError, match="requires a Visualizer"):
        force_mod.save_html("dummy.html", object())


def test_force_initjs_requires_ipython(monkeypatch):
    """initjs should raise when IPython support is unavailable."""
    from shap.plots import _force as force_mod

    monkeypatch.setattr(force_mod, "have_ipython", False)
    with pytest.raises(AssertionError, match="IPython must be installed"):
        force_mod.initjs()


def test_force_helper_functions():
    from shap.plots import _force as force_mod

    generated = force_mod.id_generator(size=10)
    assert generated.startswith("i")
    assert len(generated) == 11

    assert force_mod.ensure_not_numpy(np.str_("x")) == "x"
    assert force_mod.ensure_not_numpy(np.float64(1.2)) == 1.2


@pytest.mark.mpl_image_compare(tolerance=3)
def test_force_plot_negative_sign():
    np.random.seed(0)
    base = 100
    contribution = np.r_[-np.random.rand(5)]
    names = [f"minus_{i}" for i in range(5)]

    shap.force_plot(
        base,
        contribution,
        names,
        matplotlib=True,
        show=False,
    )
    return plt.gcf()


@pytest.mark.mpl_image_compare(tolerance=3)
def test_force_plot_positive_sign():
    np.random.seed(0)
    base = 100
    contribution = np.r_[np.random.rand(5)]
    names = [f"plus_{i}" for i in range(5)]

    shap.force_plot(
        base,
        contribution,
        names,
        matplotlib=True,
        show=False,
    )
    return plt.gcf()
