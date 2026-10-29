"""Independent filter-unit, temperature, calibration and provenance oracles."""

# ruff: noqa: INP001

import importlib
import json
import logging
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from productiecapaciteit.src import filter_calibration as calibration


def _viscosity(temperature, reference=12.0):
    return ((1 + 0.0155 * (np.asarray(temperature) - 20)) / (1 + 0.0155 * (reference - 20))) ** -1.572


def _coefficients(*, normalized=False):
    coefficients = pd.DataFrame({
        "datum": pd.to_datetime(["2020-01-01", "2020-01-11"]),
        "offset": [-0.24, -0.12],
        "slope": [-0.012, -0.006],
    })
    if normalized:
        coefficients["temperature_semantics"] = "reference_normalized"
        coefficients["temperature_ref_degC"] = 12.0
    return coefficients


def _native_observations():
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    days_in_period = np.tile(np.arange(10), 2)
    a_reference = np.repeat([-0.2, -0.1], 10) - 0.003 * days_in_period
    temperature = 12 + 8 * np.cos(np.arange(20) * 0.61)
    sensor_flow = 3 + np.arange(20) * 0.17
    sensor_flow[3] *= -1
    frame = pd.DataFrame(
        {
            "gws1": 1.5,
            "q_sensor": sensor_flow,
            "Q": sensor_flow * 4,
            "T_extracted": temperature,
        },
        index=index,
    )
    frame["gws0"] = frame.gws1 + a_reference * sensor_flow * _viscosity(temperature)
    return frame, a_reference


def _fit(frame, **kwargs):
    options = {
        "temperature_column": "T_extracted",
        "sensor_id": "well-17",
        "head_provenance": "direct",
        "upstream_fit_dependencies": [],
        "sensor_flow_column": "q_sensor",
    }
    options.update(kwargs)
    # Dynamic kwargs deliberately include invalid inputs in parameterized tests.
    return calibration.fit_normalized_filter(
        frame, pd.to_datetime(["2020-01-01", "2020-01-11"]), **options  # ty: ignore[invalid-argument-type]
    )


@pytest.mark.parametrize("explicit_metadata", [False, True])
def test_legacy_filter_conversion_does_not_apply_viscosity_twice(explicit_metadata):
    """Keep empirical coefficients empirical while converting signed hourly units."""
    coefficients = _coefficients()
    if explicit_metadata:
        coefficients["temperature_semantics"] = "legacy_empirical"
        coefficients["temperature_ref_degC"] = np.nan
    index = pd.to_datetime(["2020-01-02", "2020-01-12"])
    with pytest.warns(calibration.LegacyFilterCoefficientWarning, match="not.*viscosity"):
        resistance = coefficients.wel.resistance_model(index, temperature=[2.0, 25.0])
    np.testing.assert_allclose(resistance, np.array([0.252, 0.126]) / 24)
    np.testing.assert_allclose(coefficients.wel.a_model(index), [-0.252, -0.126])


def test_normalized_filter_viscosity_units_and_reference_identity():
    """Apply one independently evaluated viscosity ratio to normalized coefficients."""
    coefficients = _coefficients(normalized=True)
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"])
    resistance = coefficients.wel.resistance_model(index, temperature=[4.0, 12.0, 24.0])
    np.testing.assert_allclose(resistance, np.array([0.252, 0.264, 0.276]) / 24 * _viscosity([4, 12, 24]))
    assert resistance.iloc[0] > 0.252 / 24
    assert resistance.iloc[2] < 0.276 / 24


@pytest.mark.parametrize("temperature", [None, [12, np.nan], [12, np.inf], [12, -1], [12, 45], [12]])
def test_normalized_filter_requires_valid_aligned_extracted_temperature(temperature):
    """Reject missing, out-of-range and misaligned temperature specifically."""
    with pytest.raises(calibration.ExtractedWaterTemperatureError, match="extracted-water"):
        _coefficients(normalized=True).wel.resistance_model(
            pd.to_datetime(["2020-01-02", "2020-01-03"]), temperature=temperature
        )


def test_temperature_series_is_label_aligned_not_silently_position_matched():
    """Resolve temperature by timestamp, rejecting an incomplete channel."""
    index = pd.to_datetime(["2020-01-02", "2020-01-03"])
    temperature = pd.Series([24.0, 4.0], index=index[::-1])
    actual = _coefficients(normalized=True).wel.resistance_model(index, temperature=temperature)
    np.testing.assert_allclose(actual, np.array([0.252, 0.264]) / 24 * _viscosity([4, 24]))
    with pytest.raises(calibration.ExtractedWaterTemperatureError):
        _coefficients(normalized=True).wel.resistance_model(index, temperature=temperature.iloc[:1])


def test_cleaning_split_and_one_sided_coefficients_do_not_apply_cleaning_retrospectively():
    """Split at cleaning and preserve the pre-cleaning limit of the preceding interval."""
    coefficients = _coefficients(normalized=True)
    edges = pd.to_datetime(["2020-01-09", "2020-01-13"])
    split = coefficients.wel.split_intervals(edges)
    assert split.equals(pd.to_datetime(["2020-01-09", "2020-01-11", "2020-01-13"]))
    endpoint = split[1:]
    left = coefficients.wel.resistance_model(endpoint, temperature=12.0, side="left")
    right = coefficients.wel.resistance_model(endpoint, temperature=12.0, side="right")
    np.testing.assert_allclose(left, np.array([0.36, 0.132]) / 24)
    np.testing.assert_allclose(right, np.array([0.12, 0.132]) / 24)


def test_metadata_reader_rejects_mixed_unknown_and_missing_reference_semantics():
    """Never infer normalized temperature semantics from partial metadata."""
    for metadata in [
        {"temperature_semantics": ["reference_normalized", "legacy_empirical"], "temperature_ref_degC": 12},
        {"temperature_semantics": "unknown", "temperature_ref_degC": 12},
        {"temperature_semantics": "reference_normalized"},
        {"temperature_ref_degC": 12},
    ]:
        coefficients = _coefficients().assign(**metadata)
        with pytest.raises(ValueError, match="metadata|semantics|reference"):
            calibration.filter_resistance(coefficients, pd.to_datetime(["2020-01-02"]), temperature=12.0)


def test_fit_refits_slope_and_cleaning_offsets_at_native_reference_temperature():
    """Recover known native-resolution parameters with a reverse-flow observation."""
    frame, reference = _native_observations()
    result = _fit(frame)
    np.testing.assert_allclose(result.offset, [-0.2, -0.1])
    np.testing.assert_allclose(result.slope, -0.003)
    np.testing.assert_allclose(calibration.evaluate_filter_coefficient(result, frame.index), reference)
    assert result.temperature_semantics.eq("reference_normalized").all()
    assert result.sensor_id.eq("well-17").all()
    assert result.head_provenance.eq("direct").all()
    assert result.flow_denominator_provenance.eq("measured_individual:q_sensor").all()
    assert json.loads(result.upstream_fit_dependencies.iloc[0]) == []
    assert result.n_observations.eq(len(frame)).all()
    assert json.loads(result.fit_periods.iloc[0]) == [[frame.index[0].isoformat(), frame.index[-1].isoformat()]]


def test_fit_normalizes_before_any_averaging_of_native_head_flow_and_temperature():
    """Recover reference parameters despite correlated intraday flow and temperature."""
    index = pd.date_range("2020-01-01", periods=40, freq="12h")
    temperature = np.tile([4.0, 24.0], 20)
    flow = np.tile([1.0, 7.0], 20)
    reference = np.repeat([-0.2, -0.1], 20) - 0.003 * np.tile(np.arange(20) / 2, 2)
    frame = pd.DataFrame(
        {
            "gws0": 1.5 + reference * flow * _viscosity(temperature),
            "gws1": 1.5,
            "q_sensor": flow,
            "T_extracted": temperature,
        },
        index=index,
    )
    result = _fit(frame)
    np.testing.assert_allclose(result.offset, [-0.2, -0.1], rtol=1e-13)
    np.testing.assert_allclose(result.slope, -0.003, rtol=1e-13)


def test_normalized_fit_minimizes_residual_not_residual_squared():
    """Match a hand-derived ordinary least-squares line rather than a quartic fit."""
    index = pd.date_range("2020-01-01", periods=4)
    observed = np.array([-0.21, -0.24, -0.32, -0.29])
    frame = pd.DataFrame({"gws0": observed * 2, "gws1": 0.0, "q_sensor": 2.0, "T_extracted": 12.0}, index=index)
    result = calibration.fit_normalized_filter(
        frame,
        index[:1],
        temperature_column="T_extracted",
        sensor_id="well-17",
        head_provenance="direct",
        upstream_fit_dependencies=[],
        sensor_flow_column="q_sensor",
    )
    # Hand-derived ordinary least-squares line: mean(a)=-.265, slope=-.032.
    np.testing.assert_allclose(result.offset, [-0.217])
    np.testing.assert_allclose(result.slope, [-0.032])


def test_equal_flow_fallback_is_explicit_and_preserves_unequal_sensor_bias():
    """Expose the n*q_sensor/Q bias even after exact viscosity normalization."""
    frame, _ = _native_observations()
    frame["Q"] = frame.q_sensor * 2
    with pytest.raises(ValueError, match="effective"):
        _fit(frame, sensor_flow_column=None, nput=4)
    effective = _fit(frame, sensor_flow_column=None, nput=4, allow_effective_equal_flow=True)
    np.testing.assert_allclose(effective.offset, [-0.4, -0.2])
    np.testing.assert_allclose(effective.slope, -0.006)
    assert effective.flow_denominator_provenance.eq("effective_equal_flow:Q/nput=4").all()
    direct = _fit(frame, nput=4, allow_effective_equal_flow=True)
    np.testing.assert_allclose(direct.offset, [-0.2, -0.1])


def test_fit_records_declared_periods_and_upstream_reconstruction_dependencies():
    """Exclude held-out windows from fitting while retaining reconstruction lineage."""
    frame, _ = _native_observations()
    periods = [(frame.index[1], frame.index[7]), (frame.index[12], frame.index[18])]
    dependency = {"parameter_fit": "filter-v1", "fit_periods": [["2019-01-01", "2019-12-31"]]}
    frame.loc[frame.index[[0, 8, 9, 10, 11, 19]], "gws0"] = 1e5
    result = _fit(
        frame,
        fit_periods=periods,
        head_provenance="reconstructed",
        upstream_fit_dependencies=[dependency],
    )
    np.testing.assert_allclose(result.offset, [-0.2, -0.1])
    np.testing.assert_allclose(result.slope, -0.003)
    assert result.head_provenance.eq("reconstructed").all()
    assert json.loads(result.upstream_fit_dependencies.iloc[0]) == [dependency]
    assert json.loads(result.fit_periods.iloc[0]) == [[start.isoformat(), end.isoformat()] for start, end in periods]
    assert result.n_observations.eq(14).all()


def test_missing_or_invalid_fit_temperature_has_specific_error():
    """Distinguish unavailable extracted-water temperature from generic fit errors."""
    frame, _ = _native_observations()
    with pytest.raises(calibration.ExtractedWaterTemperatureError, match="T_extracted"):
        _fit(frame.drop(columns="T_extracted"))
    frame.loc[frame.index[3], "T_extracted"] = np.nan
    with pytest.raises(calibration.ExtractedWaterTemperatureError, match="extracted-water"):
        _fit(frame)


def test_fit_rejects_unidentifiable_period_and_missing_declared_sensor_flow():
    """Fail rather than invent unobserved offsets or replace a declared flow sensor."""
    frame, _ = _native_observations()
    with pytest.raises(ValueError, match="identif"):
        _fit(frame.iloc[:5])
    with pytest.raises(ValueError, match="q_sensor"):
        _fit(frame.drop(columns="q_sensor"), nput=4, allow_effective_equal_flow=True)


def test_fit_excludes_unusable_heads_and_flow_before_validating_temperature():
    """Keep missing validation heads separate from valid native calibration rows."""
    frame, _ = _native_observations()
    frame.loc[frame.index[2], "q_sensor"] = 0.0
    frame.loc[frame.index[7], "gws1"] = np.nan
    frame.loc[frame.index[12], "q_sensor"] = np.nan
    frame.loc[frame.index[[2, 7, 12]], "T_extracted"] = np.nan
    result = _fit(frame)
    np.testing.assert_allclose(result.offset, [-0.2, -0.1])
    np.testing.assert_allclose(result.slope, -0.003)
    assert result.n_observations.eq(17).all()


def test_normalized_metadata_survives_csv_without_dataframe_attrs():
    """Reload plain persisted columns and recover physical measured head loss."""
    frame, _ = _native_observations()
    result = _fit(frame)
    restored = pd.read_csv(StringIO(result.to_csv(index=False)), parse_dates=["datum"])
    expected = -frame.gws0.sub(frame.gws1) / frame.q_sensor / 24
    actual = restored.wel.resistance_model(frame.index, temperature=frame.T_extracted)
    np.testing.assert_allclose(actual, expected)


def test_putfilter_report_import_does_not_open_files_or_load_measurements(monkeypatch):
    """Keep pure fitting available without private workbook access or log writes."""

    def reject_io(*_args, **_kwargs):
        pytest.fail("putfilter3 performed report I/O at import time")

    monkeypatch.setattr(logging, "FileHandler", reject_io)
    monkeypatch.setattr(pd, "read_feather", reject_io)
    monkeypatch.setattr(pd, "read_excel", reject_io)
    report = importlib.reload(importlib.import_module("productiecapaciteit.reports.putfilter3"))
    assert callable(report.fit_filter_observations)


def test_report_opt_in_ignores_legacy_configured_slope():
    """Ensure the report does not transfer a legacy empirical slope to normalized space."""
    report = importlib.import_module("productiecapaciteit.reports.putfilter3")
    frame, _ = _native_observations()
    result = report.fit_filter_observations(
        frame,
        pd.to_datetime(["2020-01-01", "2020-01-11"]),
        normalized=True,
        configured_slope=-0.08,
        temperature_column="T_extracted",
        sensor_id="well-17",
        head_provenance="direct",
        upstream_fit_dependencies=[],
        sensor_flow_column="q_sensor",
    )
    np.testing.assert_allclose(result.slope, -0.003)


def test_report_legacy_route_keeps_configured_slope_and_labels_unknown_dependencies(monkeypatch):
    """Preserve historical fit inputs without inventing a temperature reference."""
    report = importlib.import_module("productiecapaciteit.reports.putfilter3")
    frame, _ = _native_observations()
    dates = pd.to_datetime(["2020-01-01", "2020-01-11"])
    frame = frame.drop(columns=["T_extracted", "q_sensor"])
    configured_slope = -0.001

    def historical_fit(observations, cleaning_dates, flow, *, slope, **_kwargs):
        np.testing.assert_allclose(observations, (frame.gws0 - frame.gws1) / (frame.Q / 4))
        assert cleaning_dates.equals(dates)
        assert flow == frame.Q.mean() / 4
        assert slope == configured_slope
        return pd.DataFrame({"datum": dates, "offset": [-0.2, -0.1], "slope": slope})

    monkeypatch.setattr(report, "analyse_a_put2", historical_fit)
    result = report.fit_filter_observations(frame, dates, nput=4, configured_slope=configured_slope)
    np.testing.assert_allclose(result.slope, configured_slope)
    assert result.temperature_semantics.eq("legacy_empirical").all()
    assert result.temperature_ref_degC.isna().all()
    assert json.loads(result.upstream_fit_dependencies.iloc[0]) is None
