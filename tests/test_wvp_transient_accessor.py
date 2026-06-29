import numpy as np
import pandas as pd
import pytest

import productiecapaciteit.src.weerstand_pandasaccessors as accessor_module
from productiecapaciteit.src.weerstand_pandasaccessors import (  # noqa: F401
    WvpResistanceAccessor,
    WvpTransientResistanceAccessor,
)
from productiecapaciteit.src.wvp_transient_funs import (
    build_multiwell_geometry,
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
