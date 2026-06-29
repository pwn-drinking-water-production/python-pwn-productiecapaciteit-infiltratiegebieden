import numpy as np
import pandas as pd
import pytest
from scipy.special import k0

import productiecapaciteit.src.weerstand_pandasaccessors as accessor_module
from productiecapaciteit.src.weerstand_pandasaccessors import (  # noqa: F401
    WvpResistanceAccessor,
    WvpTransientResistanceAccessor,
)
from productiecapaciteit.src.wvp_transient_funs import (
    build_crosssection_multiwell,
    build_multiwell_geometry,
    crosssection_image_offsets,
    crosssection_observation_points,
    steady_multiwell_resistance_from_kd,
)


def _steady_coefficients(offset=-0.01, slope=0.0):
    return pd.Series({
        "offset": offset,
        "offset_datum": pd.Timestamp("2020-01-01"),
        "slope": slope,
        "temp_mean": 12.0,
        "temp_delta": 0.0,
        "time_offset": 0.0,
        "method": "Niet",
        "temp_ref": 12.0,
    })


def _transient_coefficients(
    offset=-0.01,
    slope=0.0,
    well_radius_m=0.2,
    storage_coefficient=0.2,
    leakage_resistance_d=200.0,
):
    steady = _steady_coefficients(offset=offset, slope=slope)
    return pd.Series({
        "kD_ref_m2_per_d": 100.0,
        "kD_ref_slope_m2_per_d_per_d": 0.0,
        "kD_ref_datum": steady["offset_datum"],
        "temperature_mean_degC": steady["temp_mean"],
        "temperature_delta_degC": steady["temp_delta"],
        "temperature_ref_degC": steady["temp_ref"],
        "temperature_time_offset_d": steady["time_offset"],
        "temperature_method": steady["method"],
        "well_radius_m": well_radius_m,
        "storage_coefficient": storage_coefficient,
        "leakage_resistance_d": leakage_resistance_d,
    })


def _constant_resistance_coefficients(kd=100.0, leakage_resistance_d=200.0):
    coefficients = _transient_coefficients(
        offset=-0.01,
        leakage_resistance_d=leakage_resistance_d,
        well_radius_m=0.2,
    )
    coefficients["kD_ref_m2_per_d"] = kd
    coefficients["kD_ref_slope_m2_per_d_per_d"] = 0.0
    return coefficients


def test_wvpt_requires_physical_coefficients():
    coefficients = _steady_coefficients()

    with pytest.raises(AttributeError, match="kD_ref_m2_per_d"):
        coefficients.wvpt.alpha


@pytest.mark.parametrize(
    "field",
    ["well_radius_m", "storage_coefficient", "leakage_resistance_d"],
)
def test_wvpt_rejects_nonpositive_physical_coefficients(field):
    coefficients = _transient_coefficients()
    coefficients[field] = 0.0

    with pytest.raises(ValueError, match="positive"):
        coefficients.wvpt.alpha


def test_wvpt_instantaneous_resistance_functions_are_not_available():
    index = pd.date_range("2020-01-01", periods=5, freq="30D")
    transient = _constant_resistance_coefficients(kd=100.0, leakage_resistance_d=200.0)

    with pytest.raises(NotImplementedError, match="WVPT is transient"):
        transient.wvpt.a_model(index, 1, 15.0, [])
    with pytest.raises(NotImplementedError, match="WVPT is transient"):
        transient.wvpt.a_model_reftemp(index, 1, 15.0, [])
    with pytest.raises(NotImplementedError, match="WVPT is transient"):
        transient.wvpt.resistance_model(index, 1, 15.0, [])
    with pytest.raises(NotImplementedError, match="WVPT is transient"):
        transient.wvpt.resistance_model_reftemp(index, 1, 15.0, [])
    assert not hasattr(transient.wvpt, "drawdown_model")


def test_wvpt_alpha_beta_follow_physical_definitions():
    coefficients = _transient_coefficients(
        well_radius_m=0.3,
        storage_coefficient=0.25,
        leakage_resistance_d=150.0,
    )

    assert coefficients.wvpt.alpha**2 == pytest.approx(0.3**2 * 0.25 / 4.0)
    assert coefficients.wvpt.beta**2 == pytest.approx(1.0 / (150.0 * 0.25))


def test_wvpt_kd_model_roundtrips_steady_hantush_resistance():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    kd = 100.0
    coefficients = _constant_resistance_coefficients(kd=kd)

    actual = coefficients.wvpt.kD_model(index)

    np.testing.assert_allclose(actual.to_numpy(), kd, rtol=1e-7)


def test_wvpt_dp_model_matches_steady_drawdown_for_constant_flow():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.full(index.size, 10.0)

    transient = -coefficients.wvpt.dp_model(
        index, flow_m3h, nput=1, dx_tussenputten=15.0, r_mirrorwel=[], integration_method="gauss"
    )
    steady_resistance = steady_multiwell_resistance_from_kd(
        coefficients.wvpt.kD_model(index).to_numpy(dtype=float),
        [(1.0, 1.0)],
        nput=1,
        leakage_resistance_d=coefficients.wvpt.leakage_resistance_d,
        well_radius_m=coefficients.wvpt.well_radius_m,
    )

    np.testing.assert_allclose(transient.to_numpy(), steady_resistance * flow_m3h, rtol=1e-7)


def test_wvpt_dp_steady_matches_steady_resistance_building_block():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.array([12.0, 24.0, 36.0, 48.0])

    steady = coefficients.wvpt.dp_steady(
        index, flow_m3h, nput=1, dx_tussenputten=15.0, r_mirrorwel=[]
    )

    expected_resistance = steady_multiwell_resistance_from_kd(
        coefficients.wvpt.kD_model(index).to_numpy(dtype=float),
        [(1.0, 1.0)],
        nput=1,
        leakage_resistance_d=coefficients.wvpt.leakage_resistance_d,
        well_radius_m=coefficients.wvpt.well_radius_m,
    )

    # Sign convention matches dp_model: drawdown is returned as negative meters.
    np.testing.assert_allclose(steady.to_numpy(), -(expected_resistance * flow_m3h))
    assert steady.name == "wvpt_model_dp_steady"
    assert steady.index.equals(index)


def test_wvpt_dp_steady_matches_transient_steady_limit_for_constant_flow():
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.full(index.size, 10.0)

    steady = coefficients.wvpt.dp_steady(
        index, flow_m3h, nput=1, dx_tussenputten=15.0, r_mirrorwel=[]
    )
    transient = coefficients.wvpt.dp_model(
        index,
        flow_m3h,
        nput=1,
        dx_tussenputten=15.0,
        r_mirrorwel=[],
        initial_condition="steady",
        integration_method="gauss",
    )

    np.testing.assert_allclose(steady.to_numpy(), transient.to_numpy(), rtol=1e-7)


def test_wvpt_dp_steady_respects_multiwell_geometry():
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.full(index.size, 10.0)

    steady = coefficients.wvpt.dp_steady(
        index,
        flow_m3h,
        nput=3,
        dx_tussenputten=15.0,
        r_mirrorwel=[(-1.0, 50.0)],
        target_well_index=1,
    )

    multiwell, _ = build_multiwell_geometry(
        15.0,
        [(-1.0, 50.0)],
        3,
        target_well_index=1,
        distance_scale=1.0 / coefficients.wvpt.well_radius_m,
    )
    expected_resistance = steady_multiwell_resistance_from_kd(
        coefficients.wvpt.kD_model(index).to_numpy(dtype=float),
        multiwell,
        nput=3,
        leakage_resistance_d=coefficients.wvpt.leakage_resistance_d,
        well_radius_m=coefficients.wvpt.well_radius_m,
    )

    np.testing.assert_allclose(steady.to_numpy(), -(expected_resistance * flow_m3h))


def test_wvpt_dp_steady_applies_temperature_correction():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    temp_wvp = pd.Series(20.0, index=index)
    flow_m3h = np.full(index.size, 10.0)

    kd = coefficients.wvpt.kD_model(index, temp_wvp=temp_wvp).to_numpy(dtype=float)
    expected_resistance = steady_multiwell_resistance_from_kd(
        kd,
        [(1.0, 1.0)],
        nput=1,
        leakage_resistance_d=coefficients.wvpt.leakage_resistance_d,
        well_radius_m=coefficients.wvpt.well_radius_m,
    )

    steady = coefficients.wvpt.dp_steady(
        index,
        flow_m3h,
        nput=1,
        dx_tussenputten=15.0,
        r_mirrorwel=[],
        temp_wvp=temp_wvp,
    )

    np.testing.assert_allclose(steady.to_numpy(), -(expected_resistance * flow_m3h))


def test_wvpt_dp_steady_zero_flow_gives_zero():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)

    steady = coefficients.wvpt.dp_steady(
        index,
        np.zeros(index.size),
        nput=1,
        dx_tussenputten=15.0,
        r_mirrorwel=[],
    )

    np.testing.assert_allclose(steady.to_numpy(), 0.0, rtol=0.0, atol=0.0)


def test_wvpt_zero_flow_gives_zero_dp():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)

    dp = coefficients.wvpt.dp_model(
        index,
        np.zeros(index.size),
        nput=1,
        dx_tussenputten=15.0,
        r_mirrorwel=[],
        initial_condition="zero",
    )

    np.testing.assert_allclose(dp.to_numpy(), 0.0, rtol=0.0, atol=0.0)


def test_negative_image_well_reduces_steady_resistance():
    kd = 100.0
    leakage_resistance_d = 200.0
    well_radius_m = 0.2
    no_image, _ = build_multiwell_geometry(
        15.0,
        [],
        1,
        distance_scale=1.0 / well_radius_m,
    )
    with_image, _ = build_multiwell_geometry(
        15.0,
        [(-1.0, 50.0)],
        1,
        distance_scale=1.0 / well_radius_m,
    )

    no_image_resistance = steady_multiwell_resistance_from_kd(
        kd,
        no_image,
        nput=1,
        leakage_resistance_d=leakage_resistance_d,
        well_radius_m=well_radius_m,
    )
    with_image_resistance = steady_multiwell_resistance_from_kd(
        kd,
        with_image,
        nput=1,
        leakage_resistance_d=leakage_resistance_d,
        well_radius_m=well_radius_m,
    )

    assert with_image_resistance < no_image_resistance


def test_invalid_r_mirrorwel_shape_is_rejected():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)

    with pytest.raises(ValueError, match="multiplicity"):
        coefficients.wvpt.dp_model(
            index,
            np.ones(index.size),
            nput=1,
            dx_tussenputten=15.0,
            r_mirrorwel=[50.0],
        )


def test_build_multiwell_geometry_accepts_dx_mirrorwell_keyword_alias():
    dx_keyword, dx_counts = build_multiwell_geometry(
        dx_put=15.0,
        dx_mirrorwell=[(-1.0, 50.0)],
        nput=3,
        distance_scale=5.0,
    )
    r_keyword, r_counts = build_multiwell_geometry(
        dx_put=15.0,
        r_mirrorwel=[(-1.0, 50.0)],
        nput=3,
        distance_scale=5.0,
    )

    assert dx_keyword == r_keyword
    assert dx_counts == r_counts


def test_r_mirrorwel_boundary_distance_is_doubled_to_image_well_distance():
    geometry, counts = build_multiwell_geometry(
        dx_put=15.0,
        r_mirrorwel=[(-1.0, 50.0)],
        nput=1,
        distance_scale=1.0,
    )

    assert geometry == [(1.0, 1.0), (-1.0, 100.0)]
    assert counts["self_mirrorwell_terms"] == 1


def test_wvpt_warns_when_storage_coefficient_exceeds_one():
    coefficients = _transient_coefficients(storage_coefficient=1.2)

    with pytest.warns(RuntimeWarning, match="storage_coefficient"):
        assert coefficients.wvpt.beta > 0.0


def test_wvpt_dp_model_converts_total_flow_to_per_well_m3d(monkeypatch):
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.array([12.0, 24.0, 36.0, 48.0])
    captured = {}

    def fake_objective(args, return_result=False, **pextra):
        captured["Q_obs"] = pextra["Q_obs"].copy()
        return pextra["Q_obs"]

    monkeypatch.setattr(accessor_module, "objective", fake_objective)

    actual = coefficients.wvpt.dp_model(
        index,
        flow_m3h,
        nput=3,
        dx_tussenputten=15.0,
        r_mirrorwel=[],
    )

    expected = flow_m3h / 3.0 * 24.0
    np.testing.assert_allclose(captured["Q_obs"], expected)
    np.testing.assert_allclose(actual.to_numpy(), -expected)


def test_wvpt_dp_model_reindexes_flow_series_before_numpy(monkeypatch):
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = pd.Series(
        [30.0, 10.0, 20.0],
        index=[index[2], index[0], index[1]],
    )
    captured = {}

    def fake_objective(args, return_result=False, **pextra):
        captured["Q_obs"] = pextra["Q_obs"].copy()
        return pextra["Q_obs"]

    monkeypatch.setattr(accessor_module, "objective", fake_objective)

    coefficients.wvpt.dp_model(
        index,
        flow_m3h,
        nput=2,
        dx_tussenputten=15.0,
        r_mirrorwel=[],
    )

    np.testing.assert_allclose(captured["Q_obs"], np.array([10.0, 20.0, 30.0]) / 2.0 * 24.0)


def test_wvpt_dp_model_passes_integration_options(monkeypatch):
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    captured = {}

    def fake_objective(args, return_result=False, **pextra):
        captured["integration_method"] = pextra["integration_method"]
        captured["n_gauss"] = pextra["n_gauss"]
        captured["max_gauss_step_days"] = pextra["max_gauss_step_days"]
        captured["hantush_method"] = pextra["hantush_method"]
        return np.zeros(index.size)

    monkeypatch.setattr(accessor_module, "objective", fake_objective)

    coefficients.wvpt.dp_model(
        index,
        np.ones(index.size),
        nput=1,
        dx_tussenputten=15.0,
        r_mirrorwel=[],
        integration_method="quad",
        n_gauss=48,
        max_gauss_step_days=0.25,
    )

    assert captured == {
        "integration_method": "quad",
        "n_gauss": 48,
        "max_gauss_step_days": 0.25,
        "hantush_method": "variable_kd",
    }


def test_wvpt_non_reference_temperature_applies_current_viscosity_behavior():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    reference_kd = 100.0
    coefficients = _constant_resistance_coefficients(kd=reference_kd)
    temp_wvp = pd.Series(20.0, index=index)
    flow_m3h = np.full(index.size, 10.0)

    viscosity_ratio = coefficients.wvpt.visc_ratio(
        temp_wvp,
        temp_ref=coefficients.wvpt.temp_ref,
    ).to_numpy(dtype=float)
    kd = coefficients.wvpt.kD_model(index, temp_wvp=temp_wvp)
    np.testing.assert_allclose(kd.to_numpy(), reference_kd / viscosity_ratio)
    expected_resistance = steady_multiwell_resistance_from_kd(
        kd.to_numpy(dtype=float),
        [(1.0, 1.0)],
        nput=1,
        leakage_resistance_d=coefficients.wvpt.leakage_resistance_d,
        well_radius_m=coefficients.wvpt.well_radius_m,
    )
    expected_drawdown = expected_resistance * flow_m3h
    actual_drawdown = -coefficients.wvpt.dp_model(
        index,
        flow_m3h,
        nput=1,
        dx_tussenputten=15.0,
        r_mirrorwel=[],
        temp_wvp=temp_wvp,
        integration_method="gauss",
    )
    np.testing.assert_allclose(actual_drawdown.to_numpy(), expected_drawdown)


# --------------------------------------------------------------------------- #
# Cross-section geometry and accessor methods
# --------------------------------------------------------------------------- #
def test_crosssection_point_at_well_matches_target_well_geometry():
    # An observation point sitting on a well reproduces build_multiwell_geometry
    # (with symmetric neighbor pairs collapsed) term-for-term in total weight, even for a
    # two-sided (-2, b) boundary whose +-b images coincide on the axis.
    nput = 5
    dx = 15.0
    well_radius_m = 0.2
    r_mirrorwel = [(-2.0, 82.0)]

    px, py, well_xs, start_index = crosssection_observation_points(
        nput, dx, [0.0], start="center", orientation="perpendicular"
    )
    assert start_index == nput // 2
    image_offsets = crosssection_image_offsets(r_mirrorwel)
    section = build_crosssection_multiwell(px[0], py[0], well_xs, image_offsets, well_radius_m)
    reference, _ = build_multiwell_geometry(
        dx,
        r_mirrorwel,
        nput,
        target_well_index=start_index,
        distance_scale=1.0 / well_radius_m,
    )

    def weight_by_distance(terms):
        weights = {}
        for multi, distance in terms:
            weights[round(distance, 9)] = weights.get(round(distance, 9), 0.0) + multi
        return weights

    assert weight_by_distance(section) == pytest.approx(weight_by_distance(reference))


def test_crosssection_image_offsets_splits_two_sided_boundary():
    # (-2, b) is two opposite-side canals; it must split to +-b, each strength -1.
    assert sorted(crosssection_image_offsets([(-2.0, 82.0)])) == sorted([
        (-1.0, 82.0),
        (-1.0, -82.0),
    ])
    # A single canal stays one-sided on +b.
    assert crosssection_image_offsets([(-1.0, 75.0)]) == [(-1.0, 75.0)]
    # No boundary -> no images.
    assert crosssection_image_offsets([]) == []


def test_crosssection_image_offsets_refuses_ambiguous_boundary():
    # Asymmetric opposite-side canals: side per canal is unknown -> refuse, ask for offsets.
    with pytest.raises(NotImplementedError, match="boundary_perp_offsets"):
        crosssection_image_offsets([(-1.0, 250.0), (-1.0, 82.0)])


def test_crosssection_image_offsets_override_is_used_verbatim():
    offsets = crosssection_image_offsets(
        [(-1.0, 250.0), (-1.0, 82.0)],
        boundary_perp_offsets=[(-1.0, 250.0), (-1.0, -82.0)],
    )
    assert offsets == [(-1.0, 250.0), (-1.0, -82.0)]


# nput=5 -> center well index 2, end well index 4.
_NESTING_CASES = [
    ("center", "perpendicular", 2),
    ("end", "along", 4),
    ("end", "perpendicular", 4),
]


@pytest.mark.parametrize("start, orientation, target", _NESTING_CASES)
def test_dp_steady_crosssection_zero_distance_nests_to_at_well(start, orientation, target):
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.array([12.0, 24.0, 36.0])
    kwargs = dict(nput=5, dx_tussenputten=15.0, r_mirrorwel=[(-2.0, 82.0)])

    field = coefficients.wvpt.dp_steady_crosssection(
        index, flow_m3h, distances=[0.0, 40.0], start=start, orientation=orientation, **kwargs
    )
    at_well = coefficients.wvpt.dp_steady(index, flow_m3h, target_well_index=target, **kwargs)

    assert list(field.columns) == [0.0, 40.0]
    assert field.columns.name == "distance_m"
    assert field.index.equals(index)
    np.testing.assert_allclose(field[0.0].to_numpy(), at_well.to_numpy(), rtol=0.0, atol=1e-12)


# Both initial conditions, including the production default "steady" (a dropped/forced IC
# inside the cross-section is otherwise invisible -- every column would still be finite).
@pytest.mark.parametrize("initial_condition", ["zero", "steady"])
@pytest.mark.parametrize("method", ["gauss", "kd_grid"])
@pytest.mark.parametrize("start, orientation, target", _NESTING_CASES)
def test_dp_model_crosssection_zero_distance_nests_to_at_well(
    initial_condition, method, start, orientation, target
):
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.full(index.size, 20.0)
    kwargs = dict(
        nput=5,
        dx_tussenputten=15.0,
        r_mirrorwel=[(-2.0, 82.0)],
        initial_condition=initial_condition,
        integration_method=method,
    )

    field = coefficients.wvpt.dp_model_crosssection(
        index, flow_m3h, distances=[0.0, 40.0], start=start, orientation=orientation, **kwargs
    )
    at_well = coefficients.wvpt.dp_model(index, flow_m3h, target_well_index=target, **kwargs)

    assert field.shape == (index.size, 2)
    np.testing.assert_allclose(field[0.0].to_numpy(), at_well.to_numpy(), rtol=0.0, atol=1e-12)


def test_dp_model_crosssection_kd_grid_matches_gauss_offaxis():
    # The production default kd_grid path must agree with gauss off-axis, where the large
    # observation-to-well distances route through kd_grid's near/far split. nt is not
    # load-bearing here (the beta^2*span regime is unchanged), so keep it short.
    index = pd.date_range("2020-01-01", periods=30, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.full(index.size, 20.0)
    kwargs = dict(
        nput=5,
        dx_tussenputten=15.0,
        r_mirrorwel=[],
        distances=[0.0, 40.0, 120.0],
        start="center",
        initial_condition="zero",
    )

    gauss = coefficients.wvpt.dp_model_crosssection(index, flow_m3h, integration_method="gauss", **kwargs)
    kd_grid = coefficients.wvpt.dp_model_crosssection(index, flow_m3h, integration_method="kd_grid", **kwargs)

    np.testing.assert_allclose(kd_grid.to_numpy(), gauss.to_numpy(), rtol=5e-3, atol=1e-6)


def test_dp_steady_crosssection_single_well_matches_de_glee():
    # nput=1, no image: the section value at distance d is the De Glee closed form.
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    kd = 100.0
    leakage_resistance_d = 200.0
    coefficients = _constant_resistance_coefficients(kd=kd, leakage_resistance_d=leakage_resistance_d)
    flow_m3h = np.full(index.size, 24.0)
    distances = np.array([10.0, 40.0, 100.0])

    field = coefficients.wvpt.dp_steady_crosssection(
        index,
        flow_m3h,
        nput=1,
        dx_tussenputten=15.0,
        r_mirrorwel=[],
        distances=distances,
        start="center",
    )

    lam = np.sqrt(kd * leakage_resistance_d)
    expected = -24.0 * 2.0 * k0(distances / lam) / (4.0 * np.pi * kd) * flow_m3h[0]
    np.testing.assert_allclose(field.iloc[0].to_numpy(), expected, rtol=1e-12)


def test_dp_steady_crosssection_applies_temperature_correction():
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    temp_wvp = pd.Series(20.0, index=index)
    flow_m3h = np.full(index.size, 30.0)
    kwargs = dict(nput=5, dx_tussenputten=15.0, r_mirrorwel=[(-2.0, 82.0)], distances=[0.0, 40.0], start="center")

    field_temp = coefficients.wvpt.dp_steady_crosssection(index, flow_m3h, temp_wvp=temp_wvp, **kwargs)
    field_notemp = coefficients.wvpt.dp_steady_crosssection(index, flow_m3h, **kwargs)
    at_well_temp = coefficients.wvpt.dp_steady(
        index, flow_m3h, nput=5, dx_tussenputten=15.0, r_mirrorwel=[(-2.0, 82.0)], temp_wvp=temp_wvp, target_well_index=2
    )

    np.testing.assert_allclose(field_temp[0.0].to_numpy(), at_well_temp.to_numpy(), rtol=0.0, atol=1e-12)
    # The temperature correction must actually move the section (not silently dropped).
    assert np.all(np.abs(field_temp[0.0].to_numpy() - field_notemp[0.0].to_numpy()) > 1e-3)


def test_crosssection_end_perpendicular_differs_from_along_offaxis():
    # At d>0 the end-perpendicular section goes off-axis (+y) while end-along stays on the
    # row (+x); a direction swap that made them identical would otherwise pass nesting.
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.full(index.size, 30.0)
    kwargs = dict(nput=5, dx_tussenputten=15.0, r_mirrorwel=[], distances=[40.0], start="end")

    perp = coefficients.wvpt.dp_steady_crosssection(index, flow_m3h, orientation="perpendicular", **kwargs)
    along = coefficients.wvpt.dp_steady_crosssection(index, flow_m3h, orientation="along", **kwargs)

    assert np.all(np.abs(perp[40.0].to_numpy() - along[40.0].to_numpy()) > 1e-2)


def test_dp_steady_crosssection_two_sided_matches_explicit_and_not_one_sided():
    # The (-2, b) auto-split must equal explicit +-b images, and must DIFFER off-axis from
    # the (wrong) one-sided +b placement while still agreeing on-axis (d=0).
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.full(index.size, 30.0)
    common = dict(nput=5, dx_tussenputten=15.0, distances=[0.0, 30.0, 60.0], start="center")

    auto = coefficients.wvpt.dp_steady_crosssection(index, flow_m3h, r_mirrorwel=[(-2.0, 82.0)], **common)
    explicit = coefficients.wvpt.dp_steady_crosssection(
        index, flow_m3h, r_mirrorwel=[], boundary_perp_offsets=[(-1.0, 82.0), (-1.0, -82.0)], **common
    )
    one_sided = coefficients.wvpt.dp_steady_crosssection(
        index, flow_m3h, r_mirrorwel=[], boundary_perp_offsets=[(-2.0, 82.0)], **common
    )

    np.testing.assert_allclose(auto.to_numpy(), explicit.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(auto[0.0].to_numpy(), one_sided[0.0].to_numpy(), rtol=1e-12)
    assert np.all(np.abs(auto[30.0].to_numpy() - one_sided[30.0].to_numpy()) > 1e-3)


def test_dp_steady_crosssection_drawdown_decays_with_distance():
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.full(index.size, 30.0)

    field = coefficients.wvpt.dp_steady_crosssection(
        index,
        flow_m3h,
        nput=5,
        dx_tussenputten=15.0,
        r_mirrorwel=[],
        distances=[0.0, 50.0, 150.0, 400.0],
        start="center",
    )

    # No boundary: drawdown magnitude must shrink monotonically moving away from the field.
    magnitude = -field.iloc[0].to_numpy()
    assert np.all(np.diff(magnitude) < 0.0)


def test_dp_steady_crosssection_sign_flips_beyond_single_boundary():
    # Documented caveat: with a constant-head canal at b, a perpendicular section flips from
    # drawdown (negative meters) below b to mounding (positive) beyond b. Pin it so a future
    # change does not silently alter the image-method behaviour.
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.full(index.size, 30.0)

    field = coefficients.wvpt.dp_steady_crosssection(
        index,
        flow_m3h,
        nput=5,
        dx_tussenputten=15.0,
        r_mirrorwel=[(-1.0, 50.0)],
        distances=[40.0, 60.0],
        start="center",
    )

    assert np.all(field[40.0].to_numpy() < 0.0)  # drawdown below the boundary
    assert np.all(field[60.0].to_numpy() > 0.0)  # mounding beyond the boundary


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(start="center", orientation="along"), "perpendicular"),
        (dict(start="end", orientation="sideways"), "perpendicular"),
        (dict(start="middle"), "start must be"),
        (dict(distances=[-1.0], start="center"), "nonnegative"),
    ],
)
def test_crosssection_observation_points_validation(kwargs, match):
    kwargs = {"distances": [0.0, 10.0], **kwargs}
    with pytest.raises(ValueError, match=match):
        crosssection_observation_points(5, 15.0, **kwargs)


@pytest.mark.parametrize("method_name", ["dp_steady_crosssection", "dp_model_crosssection"])
def test_crosssection_accessor_refuses_ambiguous_boundary(method_name):
    # The helpful NotImplementedError must surface to the caller of the accessor methods,
    # not just the geometry helper, when canal sides cannot be inferred.
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    method = getattr(coefficients.wvpt, method_name)

    with pytest.raises(NotImplementedError, match="boundary_perp_offsets"):
        method(
            index,
            np.full(index.size, 30.0),
            nput=5,
            dx_tussenputten=15.0,
            r_mirrorwel=[(-1.0, 250.0), (-1.0, 82.0)],
            distances=[0.0, 40.0],
            start="center",
        )


def test_dp_model_multiwell_kd_grid_matches_gauss():
    # The production default path: nput>1 with image wells routes every well through one
    # shared-grid kd_grid convolution. No other test compares that multiwell kd_grid output
    # to an independent integration -- dropping the far-kernel multiplicity weights shifts
    # the multiwell result ~15% while leaving the single-well result identical, so only a
    # multiwell accuracy check catches it. Compare kd_grid to the per-term gauss superposition.
    index = pd.date_range("2018-01-01", periods=120, freq="12h")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    days = np.arange(index.size) * 0.5
    flow_m3h = 240.0 + 60.0 * np.sin(2.0 * np.pi * days / 90.0)

    common = dict(
        nput=3,
        dx_tussenputten=15.0,
        r_mirrorwel=[(-1.0, 50.0)],
        target_well_index=1,
        initial_condition="zero",
    )
    fast = coefficients.wvpt.dp_model(index, flow_m3h, integration_method="kd_grid", **common)
    gauss = coefficients.wvpt.dp_model(index, flow_m3h, integration_method="gauss", **common)

    np.testing.assert_allclose(fast.to_numpy(), gauss.to_numpy(), rtol=5e-3, atol=1e-6)


def test_dp_model_multiwell_kd_grid_steady_ic_matches_gauss():
    # The actual production default path: kd_grid + non-zero ("steady") initial condition +
    # multiwell. The kd_grid steady-IC term is applied in a per-term loop
    # (mult * _variable_kd_initial_drawdown); dropping that multiplicity passes every other
    # test (the zero-IC multiwell test skips the IC loop; the steady-IC multiwell test uses
    # gauss; the single-well IC test has mult==1). This pins kd_grid vs the gauss superposition
    # with a non-zero IC for nput>1 plus image wells.
    index = pd.date_range("2018-01-01", periods=120, freq="12h")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    days = np.arange(index.size) * 0.5
    flow_m3h = 240.0 + 60.0 * np.sin(2.0 * np.pi * days / 90.0)

    common = dict(
        nput=3,
        dx_tussenputten=15.0,
        r_mirrorwel=[(-1.0, 50.0)],
        target_well_index=1,
        initial_condition="steady",
    )
    fast = coefficients.wvpt.dp_model(index, flow_m3h, integration_method="kd_grid", **common)
    gauss = coefficients.wvpt.dp_model(index, flow_m3h, integration_method="gauss", **common)

    np.testing.assert_allclose(fast.to_numpy(), gauss.to_numpy(), rtol=5e-3, atol=1e-6)


def test_dp_model_multiwell_steady_ic_matches_dp_steady():
    # Constant flow + steady initial condition must reproduce the De Glee steady limit for
    # the multiwell geometry too (the nput=1 version is covered elsewhere).
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.full(index.size, 12.0)
    common = dict(nput=3, dx_tussenputten=15.0, r_mirrorwel=[(-1.0, 50.0)], target_well_index=1)

    transient = coefficients.wvpt.dp_model(
        index, flow_m3h, initial_condition="steady", integration_method="gauss", **common
    )
    steady = coefficients.wvpt.dp_steady(index, flow_m3h, **common)

    np.testing.assert_allclose(transient.to_numpy(), steady.to_numpy(), rtol=1e-6)


def test_steady_resistance_matches_hand_de_glee_single_well():
    # Independent known-answer anchor for steady_multiwell_resistance_from_kd: the dp_steady
    # "building block" tests build their expected with this same helper, so they cannot catch
    # a constant-factor or K0-argument error inside it.
    kd, leakage, well_radius_m, nput = 100.0, 200.0, 0.3, 1
    coefficient = steady_multiwell_resistance_from_kd(kd, [(1.0, 1.0)], nput, leakage, well_radius_m)
    hand = 24.0 / nput * 2.0 * k0(well_radius_m / np.sqrt(kd * leakage)) / (4.0 * np.pi * kd)

    np.testing.assert_allclose(float(coefficient), hand, rtol=1e-12)


def test_warmer_water_reduces_drawdown_magnitude():
    # Independent physical-direction check for the temperature->kD->drawdown chain. The
    # existing temperature tests build their expected with the same visc_ratio the code uses,
    # so a flipped viscosity direction passes them; this one would fail. Warmer water is less
    # viscous -> larger effective kD -> smaller drawdown magnitude.
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    coefficients = _constant_resistance_coefficients(kd=100.0)
    flow_m3h = np.full(index.size, 10.0)
    common = dict(nput=1, dx_tussenputten=15.0, r_mirrorwel=[], integration_method="gauss")

    cold = coefficients.wvpt.dp_model(index, flow_m3h, temp_wvp=pd.Series(5.0, index=index), **common)
    warm = coefficients.wvpt.dp_model(index, flow_m3h, temp_wvp=pd.Series(20.0, index=index), **common)
    # dp_model returns negative meters; warmer water => smaller drawdown magnitude everywhere.
    assert np.all(np.abs(warm.to_numpy()) < np.abs(cold.to_numpy()))

    kd_cold = coefficients.wvpt.kD_model(index, temp_wvp=pd.Series(5.0, index=index)).to_numpy()
    kd_warm = coefficients.wvpt.kD_model(index, temp_wvp=pd.Series(20.0, index=index)).to_numpy()
    assert np.all(kd_warm > kd_cold)
