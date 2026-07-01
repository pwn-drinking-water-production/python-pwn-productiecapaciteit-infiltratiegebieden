from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import productiecapaciteit.reports.report_wvpweerstand_transient as report
import productiecapaciteit.src.weerstand_pandasaccessors as accessor_module
from productiecapaciteit.reports.report_wvpweerstand_transient import (
    default_transient_coefficients,
    fit_transient_coefficients,
    read_series_workbook,
    reconstruct_aquifer_drawdown,
    resample_transient_observations,
    transient_coefficients_from_sheet,
    transient_drawdown_for_coefficients,
    write_series_workbook,
)


def _ci(nput=1):
    return pd.Series({
        "nput": nput,
        "dx_tussenputten": 15.0,
        "r_mirrorwel": [],
    })


def _filter_coefficients():
    return pd.DataFrame({
        "datum": [pd.Timestamp("2020-01-01")],
        "offset": [-0.1],
        "slope": [0.0],
    })


def _write_filter_workbook(tmp_path, strangen):
    filter_dir = tmp_path / "Filterweerstand"
    filter_dir.mkdir(exist_ok=True)
    with pd.ExcelWriter(filter_dir / "Filterweerstand_modelcoefficienten.xlsx") as writer:
        for strang in strangen:
            _filter_coefficients().to_excel(writer, sheet_name=strang, index=False)


def _patch_main_environment(monkeypatch, tmp_path, config):
    monkeypatch.setattr(report, "results_dir", tmp_path)
    monkeypatch.setattr(report, "plot_styles_dir", tmp_path)
    monkeypatch.setattr(report, "get_config", lambda fn: config)
    monkeypatch.setattr(report.plt.style, "use", lambda *args, **kwargs: None)
    monkeypatch.setattr(report, "plot_fit", lambda *args, **kwargs: tmp_path / "plot.png")


def test_reconstruct_aquifer_drawdown_uses_gws1_and_filter_fallback():
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "Q": [10.0, 10.0],
            "gws0": [1.0, 1.0],
            "gws1": [0.7, np.nan],
            "pandpeil": [2.0, 2.0],
        },
        index=index,
    )

    actual = reconstruct_aquifer_drawdown(df, _ci(), _filter_coefficients())

    assert actual.loc[index[0], "p_omstorting"] == pytest.approx(0.7)
    assert actual.loc[index[0], "drawdown_aquifer"] == pytest.approx(1.3)
    assert actual.loc[index[1], "p_omstorting"] == pytest.approx(2.0)
    assert actual.loc[index[1], "drawdown_aquifer"] == pytest.approx(0.0)


def test_transient_workbook_roundtrip(tmp_path):
    workbook = tmp_path / report.TRANSIENT_WORKBOOK
    sheets = {
        "Q100": default_transient_coefficients(leakage_resistance_d=123.0),
        "IK102": default_transient_coefficients(leakage_resistance_d=456.0),
    }

    write_series_workbook(workbook, sheets)
    actual = read_series_workbook(workbook)

    assert set(actual) == {"Q100", "IK102"}
    assert actual["Q100"]["leakage_resistance_d"] == pytest.approx(123.0)
    assert actual["IK102"]["leakage_resistance_d"] == pytest.approx(456.0)
    assert "gewijzigd" in actual["Q100"].index


def test_default_coefficients_use_fixed_storage_and_radius():
    coefficients = default_transient_coefficients()

    assert coefficients["well_radius_m"] == pytest.approx(0.3)
    assert coefficients["storage_coefficient"] == pytest.approx(0.2)
    assert report.DEFAULT_WELL_RADIUS_M == pytest.approx(0.3)
    assert report.DEFAULT_STORAGE_COEFFICIENT == pytest.approx(0.2)


def test_force_physical_constants_overrides_stored_values():
    stored = default_transient_coefficients(well_radius_m=0.15, storage_coefficient=0.45)

    forced = report.force_physical_constants(stored)

    assert forced["well_radius_m"] == pytest.approx(0.3)
    assert forced["storage_coefficient"] == pytest.approx(0.2)
    # The fitted parameters are untouched.
    assert forced["leakage_resistance_d"] == stored["leakage_resistance_d"]


def test_read_series_workbook_required_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required coefficient workbook"):
        read_series_workbook(tmp_path / "missing.xlsx", required=True)


def test_write_series_workbook_reports_excel_lock(monkeypatch, tmp_path):
    workbook = tmp_path / "locked.xlsx"

    def locked_replace(self, target):
        raise PermissionError("file is locked")

    monkeypatch.setattr(Path, "replace", locked_replace)

    with pytest.raises(PermissionError, match="Close the workbook in Excel"):
        write_series_workbook(
            workbook,
            {"Q100": default_transient_coefficients(leakage_resistance_d=123.0)},
        )


def test_resample_transient_observations_requires_positive_drawdown_and_two_rows():
    index = pd.date_range("2020-01-01", periods=3, freq="12h")
    df = pd.DataFrame(
        {
            "Q": [10.0, 10.0, 10.0],
            "drawdown_aquifer": [1.0, 0.0, -1.0],
        },
        index=index,
    )

    with pytest.raises(ValueError, match="At least two"):
        resample_transient_observations(df, frequency="12h")


def test_report_uses_all_resampled_timesteps_by_default():
    assert report.RESAMPLE_FREQUENCY == "12h"


def test_transient_drawdown_for_coefficients_uses_zero_initial_condition(monkeypatch):
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    dfm = pd.DataFrame({"Q": np.ones(index.size)}, index=index)
    coefficients = default_transient_coefficients()
    captured = {}

    def fake_objective(args, return_result=False, **pextra):
        captured["initial_condition"] = pextra["initial_condition"]
        return np.zeros(index.size)

    monkeypatch.setattr(accessor_module, "objective", fake_objective)

    actual = transient_drawdown_for_coefficients(
        100.0,
        200.0,
        dfm,
        coefficients,
        _ci(),
    )

    assert captured["initial_condition"] == "zero"
    np.testing.assert_allclose(actual.to_numpy(), 0.0)


def test_fit_transient_coefficients_recovers_synthetic_values():
    true_kd = 80.0
    true_leakage = 150.0
    ci = _ci()
    index = pd.date_range("2020-01-01", periods=60, freq="D")
    # Several flow steps so the transient response constrains both parameters.
    flow = np.zeros(index.size)
    flow[5:20] = 12.0
    flow[20:35] = 4.0
    flow[35:50] = 16.0
    flow[50:] = 8.0

    truth = default_transient_coefficients(
        kd_ref_m2_per_d=true_kd,
        leakage_resistance_d=true_leakage,
    )
    observed = transient_drawdown_for_coefficients(
        true_kd,
        true_leakage,
        pd.DataFrame({"Q": flow}, index=index),
        truth,
        ci,
    )
    dfm = pd.DataFrame(
        {"Q": flow, "drawdown_aquifer": observed.to_numpy(dtype=float)},
        index=index,
    )
    seed = truth.copy()
    seed["kD_ref_m2_per_d"] = 200.0
    seed["leakage_resistance_d"] = 500.0

    result = fit_transient_coefficients(dfm, seed, ci)

    assert result["coefficients"]["kD_ref_m2_per_d"] == pytest.approx(true_kd, rel=0.1)
    assert result["coefficients"]["leakage_resistance_d"] == pytest.approx(true_leakage, rel=0.1)
    assert result["optimizer_result"].success is True
    assert result["residuals"].abs().max() < 1e-3
    assert result["residual_innovations"].abs().max() < 1e-3


def test_fit_transient_coefficients_matches_measured_levels(monkeypatch):
    # The objective matches drawdown levels directly (not their first differences),
    # so a measured level with no offset is reproduced and the residuals go to zero.
    true_leakage = 80.0
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    trend = np.arange(index.size, dtype=float)
    dfm = pd.DataFrame(
        {
            "Q": np.full(index.size, 10.0),
            "drawdown_aquifer": true_leakage / 10.0 * trend,
        },
        index=index,
    )
    seed = default_transient_coefficients(kd_ref_m2_per_d=100.0, leakage_resistance_d=200.0)

    def fake_drawdown(kd_ref, leakage, dfm, df_a_wvpt, ci, **kwargs):
        return pd.Series(leakage / 10.0 * trend, index=dfm.index)

    monkeypatch.setattr(report, "transient_drawdown_for_coefficients", fake_drawdown)

    # Pin kD with a near-degenerate bound so the level behaviour is isolated to leakage.
    result = fit_transient_coefficients(
        dfm,
        seed,
        _ci(),
        kd_bounds_m2_per_d=(99.999, 100.001),
        leakage_bounds_d=(10.0, 500.0),
    )

    assert result["coefficients"]["leakage_resistance_d"] == pytest.approx(true_leakage, rel=1e-3)
    np.testing.assert_allclose(result["residuals"].to_numpy(), 0.0, atol=1e-6)


def test_fit_transient_coefficients_is_a_level_fit_not_a_differencing_fit(monkeypatch):
    # A constant offset in the measurements pulls the level fit; a differencing
    # (innovation) objective would ignore it. With model = leakage/10 * trend and
    # observed = offset + true/10 * trend, the linear least-squares slope is
    # a* = sum(trend * observed) / sum(trend^2), i.e. leakage is biased away from
    # the true 80 toward absorbing the offset. An innovation objective would return
    # exactly 80. This is what makes the two objectives distinguishable.
    true_leakage = 80.0
    offset = 5.0
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    trend = np.arange(index.size, dtype=float)
    dfm = pd.DataFrame(
        {
            "Q": np.full(index.size, 10.0),
            "drawdown_aquifer": offset + true_leakage / 10.0 * trend,
        },
        index=index,
    )
    seed = default_transient_coefficients(kd_ref_m2_per_d=100.0, leakage_resistance_d=200.0)

    def fake_drawdown(kd_ref, leakage, dfm, df_a_wvpt, ci, **kwargs):
        return pd.Series(leakage / 10.0 * trend, index=dfm.index)

    monkeypatch.setattr(report, "transient_drawdown_for_coefficients", fake_drawdown)

    # Plain least squares (loss="linear") so the expected biased value is exact.
    result = fit_transient_coefficients(
        dfm,
        seed,
        _ci(),
        kd_bounds_m2_per_d=(99.999, 100.001),
        leakage_bounds_d=(10.0, 500.0),
        loss="linear",
    )

    expected_slope = float(np.sum(trend * dfm.drawdown_aquifer.to_numpy()) / np.sum(trend**2))
    expected_leakage = 10.0 * expected_slope
    assert expected_leakage == pytest.approx(93.64, abs=0.1)  # not 80 -> level fit, not innovation fit
    assert result["coefficients"]["leakage_resistance_d"] == pytest.approx(expected_leakage, rel=2e-3)


def test_fit_transient_coefficients_penalizes_infeasible_candidates(monkeypatch):
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    trend = np.arange(index.size, dtype=float)
    dfm = pd.DataFrame(
        {
            "Q": np.full(index.size, 10.0),
            "drawdown_aquifer": 8.0 * trend,
        },
        index=index,
    )
    seed = default_transient_coefficients(kd_ref_m2_per_d=100.0, leakage_resistance_d=200.0)

    def fake_drawdown(kd_ref, leakage, dfm, df_a_wvpt, ci, **kwargs):
        if leakage > 100.0:
            raise ValueError("cannot bracket kD")
        return pd.Series(leakage / 10.0 * trend, index=dfm.index)

    monkeypatch.setattr(report, "transient_drawdown_for_coefficients", fake_drawdown)

    result = fit_transient_coefficients(
        dfm,
        seed,
        _ci(),
        kd_bounds_m2_per_d=(99.999, 100.001),
        leakage_bounds_d=(10.0, 500.0),
    )

    assert result["coefficients"]["leakage_resistance_d"] == pytest.approx(80.0, rel=1e-3)
    assert result["optimizer_result"].success is True


def test_fit_transient_coefficients_reports_when_no_candidate_is_feasible(monkeypatch):
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    dfm = pd.DataFrame(
        {
            "Q": np.full(index.size, 10.0),
            "drawdown_aquifer": np.ones(index.size),
        },
        index=index,
    )
    seed = default_transient_coefficients(kd_ref_m2_per_d=100.0, leakage_resistance_d=200.0)

    def fake_drawdown(kd_ref, leakage, dfm, df_a_wvpt, ci, **kwargs):
        raise ValueError("cannot bracket kD")

    monkeypatch.setattr(report, "transient_drawdown_for_coefficients", fake_drawdown)

    with pytest.raises(RuntimeError, match="No feasible"):
        fit_transient_coefficients(
            dfm,
            seed,
            _ci(),
            kd_bounds_m2_per_d=(10.0, 500.0),
            leakage_bounds_d=(10.0, 500.0),
        )


def test_main_forces_physical_constants_and_persists_calibrated_workbook(monkeypatch, tmp_path):
    config = pd.DataFrame(
        {
            "nput": [1, 1],
            "dx_tussenputten": [15.0, 15.0],
            "r_mirrorwel": [[], []],
        },
        index=["OK", "BAD"],
    )
    _patch_main_environment(monkeypatch, tmp_path, config)

    # The seed workbook deliberately stores the wrong S/r so that the forcing is observable.
    ok_start = default_transient_coefficients(
        kd_ref_m2_per_d=50.0,
        leakage_resistance_d=300.0,
        well_radius_m=0.15,
        storage_coefficient=0.45,
    )
    bad_start = default_transient_coefficients(
        kd_ref_m2_per_d=60.0,
        leakage_resistance_d=400.0,
        well_radius_m=0.15,
        storage_coefficient=0.45,
    )
    transient_dir = tmp_path / report.RESULTS_SUBDIR
    transient_dir.mkdir()
    write_series_workbook(
        transient_dir / report.TRANSIENT_WORKBOOK,
        {"OK": ok_start, "BAD": bad_start},
    )

    _write_filter_workbook(tmp_path, ["OK", "BAD"])
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    monkeypatch.setattr(
        report,
        "load_observations",
        lambda strang, ci, df_a_filter, **kwargs: pd.DataFrame(
            {"Q": [10.0, 10.0], "drawdown_aquifer": [1.0, 1.0]},
            index=index,
        ),
    )

    captured = {}

    def fake_fit(dfm, df_a_wvpt, ci, **kwargs):
        captured[ci.name] = {
            "r": float(df_a_wvpt["well_radius_m"]),
            "S": float(df_a_wvpt["storage_coefficient"]),
        }
        if ci.name == "BAD":
            raise ValueError("fit failed")
        coeff = df_a_wvpt.copy()
        coeff["kD_ref_m2_per_d"] = 123.0
        coeff["leakage_resistance_d"] = 77.0
        coeff[report.TRANSIENT_MODIFIED_KEY] = pd.Timestamp.now()
        return {
            "coefficients": transient_coefficients_from_sheet(coeff),
            "modeled": pd.Series([1.0, 1.0], index=dfm.index),
            "residuals": pd.Series([0.0, 0.0], index=dfm.index),
            "optimizer_result": None,
        }

    monkeypatch.setattr(report, "fit_transient_coefficients", fake_fit)

    report.main(["OK", "BAD"])

    transient_fp = tmp_path / report.RESULTS_SUBDIR / report.TRANSIENT_WORKBOOK
    actual = read_series_workbook(transient_fp)
    log_text = (tmp_path / report.RESULTS_SUBDIR / report.TRANSIENT_LOG).read_text()

    # Forcing is applied to the seed handed to the fit.
    assert captured["OK"]["r"] == pytest.approx(0.3)
    assert captured["OK"]["S"] == pytest.approx(0.2)

    assert set(actual) == {"OK", "BAD"}
    assert set(actual["OK"].index) == set(report.TRANSIENT_COEFFICIENT_KEYS)
    assert actual["OK"]["kD_ref_m2_per_d"] == pytest.approx(123.0)
    assert actual["OK"]["leakage_resistance_d"] == pytest.approx(77.0)
    assert actual["OK"]["well_radius_m"] == pytest.approx(0.3)
    assert actual["OK"]["storage_coefficient"] == pytest.approx(0.2)
    assert actual["OK"].wvpt.leakage_resistance_d == pytest.approx(77.0)

    # The failed strang keeps its seed parameters but with forced S/r.
    assert actual["BAD"]["kD_ref_m2_per_d"] == pytest.approx(60.0)
    assert actual["BAD"]["leakage_resistance_d"] == pytest.approx(400.0)
    assert actual["BAD"]["well_radius_m"] == pytest.approx(0.3)
    assert actual["BAD"]["storage_coefficient"] == pytest.approx(0.2)

    assert "BAD" in log_text
    assert "fit failed" in log_text


def _make_source_selection_env(monkeypatch, tmp_path):
    config = pd.DataFrame(
        {
            "nput": [1],
            "dx_tussenputten": [15.0],
            "r_mirrorwel": [[]],
        },
        index=["OK"],
    )
    _patch_main_environment(monkeypatch, tmp_path, config)
    _write_filter_workbook(tmp_path, ["OK"])
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    monkeypatch.setattr(
        report,
        "load_observations",
        lambda *args, **kwargs: pd.DataFrame(
            {"Q": [10.0, 10.0], "drawdown_aquifer": [1.0, 1.0]},
            index=index,
        ),
    )

    captured = {}

    def fake_fit(dfm, df_a_wvpt, ci, **kwargs):
        captured["seed_kd"] = float(df_a_wvpt["kD_ref_m2_per_d"])
        coeff = df_a_wvpt.copy()
        coeff["leakage_resistance_d"] = 77.0
        coeff[report.TRANSIENT_MODIFIED_KEY] = pd.Timestamp.now()
        return {
            "coefficients": transient_coefficients_from_sheet(coeff),
            "modeled": pd.Series([1.0, 1.0], index=dfm.index),
            "residuals": pd.Series([0.0, 0.0], index=dfm.index),
            "optimizer_result": None,
        }

    monkeypatch.setattr(report, "fit_transient_coefficients", fake_fit)
    transient_dir = tmp_path / report.RESULTS_SUBDIR
    transient_dir.mkdir()
    return captured, transient_dir


def test_main_seeds_from_defaults_when_no_modelcoefficienten(monkeypatch, tmp_path):
    captured, transient_dir = _make_source_selection_env(monkeypatch, tmp_path)
    # No calibrated workbook exists yet, so the strang is seeded from module defaults.
    assert not (transient_dir / report.TRANSIENT_WORKBOOK).exists()

    report.main(["OK"])

    assert captured["seed_kd"] == pytest.approx(report.DEFAULT_KD_REF_M2_PER_D)
    assert (transient_dir / report.TRANSIENT_WORKBOOK).exists()


def test_main_carries_over_unprocessed_strang_with_forced_constants(monkeypatch, tmp_path):
    captured, transient_dir = _make_source_selection_env(monkeypatch, tmp_path)
    # 'OK' is in the config and gets fitted; 'EXTRA' only exists in the source
    # workbook and must be carried into the output with forced S/r.
    write_series_workbook(
        transient_dir / report.TRANSIENT_WORKBOOK,
        {
            "OK": default_transient_coefficients(kd_ref_m2_per_d=11.0),
            "EXTRA": default_transient_coefficients(
                kd_ref_m2_per_d=33.0,
                well_radius_m=0.15,
                storage_coefficient=0.45,
            ),
        },
    )

    report.main(["OK"])

    actual = read_series_workbook(transient_dir / report.TRANSIENT_WORKBOOK)
    assert set(actual) == {"OK", "EXTRA"}
    # The carried-over strang keeps its seed kD but gets the forced S/r.
    assert actual["EXTRA"]["kD_ref_m2_per_d"] == pytest.approx(33.0)
    assert actual["EXTRA"]["well_radius_m"] == pytest.approx(0.3)
    assert actual["EXTRA"]["storage_coefficient"] == pytest.approx(0.2)


def test_main_seeds_from_modelcoefficienten_when_present(monkeypatch, tmp_path):
    captured, transient_dir = _make_source_selection_env(monkeypatch, tmp_path)
    write_series_workbook(
        transient_dir / report.TRANSIENT_WORKBOOK,
        {"OK": default_transient_coefficients(kd_ref_m2_per_d=22.0)},
    )

    report.main(["OK"])

    assert captured["seed_kd"] == pytest.approx(22.0)


# --------------------------------------------------------------------------- #
# Series-resistance head-loss term (infiltration + borehole wall)
# --------------------------------------------------------------------------- #
def _series_flow(index):
    flow = np.zeros(index.size)
    flow[5:25] = 12.0
    flow[25:45] = 6.0
    flow[45:65] = 16.0
    flow[65:] = 9.0
    return flow


def _series_synthetic(true_kd=80.0, true_leakage=150.0, true_growth=1.0e-5):
    # Aquifer + series head loss on an index that spans the 2015-01-01 datum (so V is signed).
    ci = _ci()
    index = pd.date_range("2014-06-01", periods=80, freq="10D")
    flow = _series_flow(index)
    volume = report.cumulative_extracted_volume_m3(
        pd.Series(flow, index=index), datum=pd.Timestamp("2015-01-01")
    ).to_numpy()
    q95 = float(np.nanpercentile(flow, report.SERIES_FLOW_REF_PERCENTILE))
    truth = default_transient_coefficients(
        kd_ref_m2_per_d=true_kd,
        leakage_resistance_d=true_leakage,
        series_flow_ref_m3_per_h=q95,
        series_growth_per_m3=true_growth,
    )
    dft = pd.DataFrame({"Q": flow, "cumulative_volume_m3": volume}, index=index)
    observed = transient_drawdown_for_coefficients(
        true_kd, true_leakage, dft, truth, ci, series_growth_per_m3=true_growth
    )
    dfm = pd.DataFrame(
        {"Q": flow, "drawdown_aquifer": observed.to_numpy(dtype=float), "cumulative_volume_m3": volume},
        index=index,
    )
    return dfm, truth, ci, q95


def test_default_coefficients_include_series_keys():
    coeff = default_transient_coefficients()

    assert coeff["series_dp_ref_m"] == pytest.approx(report.DEFAULT_SERIES_DP_REF_M)
    assert pd.Timestamp(coeff["series_datum"]) == report.DEFAULT_SERIES_DATUM
    assert coeff["series_growth_per_m3"] == pytest.approx(0.0)
    assert set(report.TRANSIENT_SERIES_KEYS) <= set(coeff.index)


def test_cumulative_extracted_volume_is_signed_around_datum():
    index = pd.date_range("2014-06-01", periods=14, freq="30D")  # straddles 2015-01-01
    flow = pd.Series(np.full(index.size, 10.0), index=index)
    datum = pd.Timestamp("2015-01-01")

    v = report.cumulative_extracted_volume_m3(flow, datum=datum)

    # Signed: negative before the datum, positive after, strictly increasing, zero at the datum.
    assert v[v.index < datum].max() < 0.0
    assert v[v.index > datum].min() > 0.0
    assert np.all(np.diff(v.to_numpy()) > 0.0)
    hours = (index - index[0]).total_seconds().to_numpy() / 3600.0
    datum_hours = (datum - index[0]).total_seconds() / 3600.0
    assert np.interp(datum_hours, hours, v.to_numpy()) == pytest.approx(0.0, abs=1e-6)


def test_series_head_loss_matches_formula_and_baseline_at_datum():
    index = pd.date_range("2014-06-01", periods=8, freq="30D")
    flow = np.full(index.size, 10.0)
    volume = np.linspace(-1.0e5, 1.0e5, index.size)
    coeff = default_transient_coefficients(
        series_dp_ref_m=0.5, series_flow_ref_m3_per_h=16.0, series_growth_per_m3=1.0e-5
    )
    dfm = pd.DataFrame({"Q": flow, "cumulative_volume_m3": volume}, index=index)

    hl = report.series_head_loss(coeff, dfm)

    expected = 0.5 / 16.0 * flow * (1.0 + 1.0e-5 * volume)
    np.testing.assert_allclose(hl.to_numpy(), expected, rtol=1e-12)
    # At the datum (V=0) and the reference flow the loss is exactly the 0.5 m baseline.
    at_datum = pd.DataFrame({"Q": [16.0], "cumulative_volume_m3": [0.0]}, index=index[:1])
    assert float(report.series_head_loss(coeff, at_datum).iloc[0]) == pytest.approx(0.5)


def test_series_head_loss_is_zero_without_volume_or_flow_ref():
    index = pd.date_range("2015-01-01", periods=3, freq="D")
    coeff = default_transient_coefficients(series_flow_ref_m3_per_h=16.0, series_growth_per_m3=1.0e-5)
    no_volume = pd.DataFrame({"Q": [1.0, 2.0, 3.0]}, index=index)
    np.testing.assert_allclose(report.series_head_loss(coeff, no_volume).to_numpy(), 0.0)

    coeff0 = default_transient_coefficients(series_flow_ref_m3_per_h=0.0, series_growth_per_m3=1.0e-5)
    dfm = pd.DataFrame({"Q": [1.0, 2.0, 3.0], "cumulative_volume_m3": [0.0, 1e4, 2e4]}, index=index)
    np.testing.assert_allclose(report.series_head_loss(coeff0, dfm).to_numpy(), 0.0)

    # The guard short-circuits BEFORE the viscosity factor, even with temperature active.
    coeff_sin = default_transient_coefficients(
        series_flow_ref_m3_per_h=0.0, temperature_method="sin", temperature_delta_degc=8.0
    )
    np.testing.assert_allclose(report.series_head_loss(coeff_sin, dfm).to_numpy(), 0.0)


def test_series_head_loss_clips_negative_growth_factor():
    # A growth large enough to drive (1 + g*V) negative before the datum is clipped at 0.
    index = pd.date_range("2015-01-01", periods=3, freq="D")
    coeff = default_transient_coefficients(
        series_dp_ref_m=0.5, series_flow_ref_m3_per_h=10.0, series_growth_per_m3=1.0e-3
    )
    dfm = pd.DataFrame({"Q": [10.0, 10.0, 10.0], "cumulative_volume_m3": [-5000.0, 0.0, 5000.0]}, index=index)

    hl = report.series_head_loss(coeff, dfm).to_numpy()

    assert hl[0] == pytest.approx(0.0)  # 1 + 1e-3*(-5000) = -4 -> clipped
    assert np.all(hl >= 0.0)


def test_fit_recovers_series_growth():
    true_kd, true_leakage, true_growth = 80.0, 150.0, 1.0e-5
    dfm, truth, ci, _ = _series_synthetic(true_kd, true_leakage, true_growth)
    seed = truth.copy()
    seed["kD_ref_m2_per_d"] = report.DEFAULT_KD_REF_M2_PER_D
    seed["leakage_resistance_d"] = report.DEFAULT_LEAKAGE_RESISTANCE_D
    seed["series_growth_per_m3"] = 0.0  # start at no-growth and climb

    result = fit_transient_coefficients(dfm, seed, ci)

    coeff = result["coefficients"]
    assert coeff["kD_ref_m2_per_d"] == pytest.approx(true_kd, rel=0.05)
    assert coeff["leakage_resistance_d"] == pytest.approx(true_leakage, rel=0.1)
    assert coeff["series_growth_per_m3"] == pytest.approx(true_growth, rel=0.1)
    assert result["optimizer_result"].success is True
    assert result["residuals"].abs().max() < 1e-2


def test_fit_growth_defaults_to_zero_without_throughput():
    # No cumulative_volume_m3 column -> the growth is unidentifiable and stays at 0.
    ci = _ci()
    index = pd.date_range("2020-01-01", periods=30, freq="D")
    flow = np.zeros(index.size)
    flow[5:] = 10.0
    truth = default_transient_coefficients(kd_ref_m2_per_d=80.0, leakage_resistance_d=150.0)
    observed = transient_drawdown_for_coefficients(
        80.0, 150.0, pd.DataFrame({"Q": flow}, index=index), truth, ci
    )
    dfm = pd.DataFrame({"Q": flow, "drawdown_aquifer": observed.to_numpy(dtype=float)}, index=index)
    seed = truth.copy()
    seed["kD_ref_m2_per_d"] = 150.0
    seed["leakage_resistance_d"] = 400.0

    result = fit_transient_coefficients(dfm, seed, ci)

    assert result["coefficients"]["series_growth_per_m3"] == pytest.approx(0.0, abs=1e-12)
    assert result["coefficients"]["kD_ref_m2_per_d"] == pytest.approx(80.0, rel=0.1)


def test_transient_coefficients_from_sheet_injects_series_defaults_for_old_sheet():
    sheet = default_transient_coefficients()
    old = sheet.drop(list(report.TRANSIENT_SERIES_KEYS))

    restored = transient_coefficients_from_sheet(old)

    assert restored["series_dp_ref_m"] == pytest.approx(report.DEFAULT_SERIES_DP_REF_M)
    assert restored["series_growth_per_m3"] == pytest.approx(report.DEFAULT_SERIES_GROWTH_PER_M3)
    assert pd.Timestamp(restored["series_datum"]) == report.DEFAULT_SERIES_DATUM


def test_series_params_roundtrip_through_workbook(tmp_path):
    workbook = tmp_path / report.TRANSIENT_WORKBOOK
    sheet = default_transient_coefficients(
        series_dp_ref_m=0.5,
        series_flow_ref_m3_per_h=42.0,
        series_datum=pd.Timestamp("2015-01-01"),
        series_growth_per_m3=7.0e-6,
    )

    write_series_workbook(workbook, {"Q100": sheet})
    actual = read_series_workbook(workbook)["Q100"]

    assert actual["series_flow_ref_m3_per_h"] == pytest.approx(42.0)
    assert actual["series_growth_per_m3"] == pytest.approx(7.0e-6)
    assert pd.Timestamp(actual["series_datum"]) == pd.Timestamp("2015-01-01")


def test_main_computes_q95_and_persists_series_growth(monkeypatch, tmp_path):
    # End-to-end: main -> load_observations (datum threaded) -> Q_95 computed -> REAL fit ->
    # workbook keeps a nonzero growth and the 95th-percentile reference flow.
    dfm, _truth, _ci_unused, q95 = _series_synthetic(true_kd=80.0, true_leakage=150.0, true_growth=1.0e-5)
    config = pd.DataFrame({"nput": [1], "dx_tussenputten": [15.0], "r_mirrorwel": [[]]}, index=["OK"])
    _patch_main_environment(monkeypatch, tmp_path, config)
    _write_filter_workbook(tmp_path, ["OK"])
    transient_dir = tmp_path / report.RESULTS_SUBDIR
    transient_dir.mkdir()
    write_series_workbook(
        transient_dir / report.TRANSIENT_WORKBOOK,
        {"OK": default_transient_coefficients(kd_ref_m2_per_d=100.0, leakage_resistance_d=200.0)},
    )

    captured = {}

    def stub_load(strang, ci, df_a_filter, **kwargs):
        captured["datum"] = kwargs.get("series_datum")
        return dfm.copy()

    monkeypatch.setattr(report, "load_observations", stub_load)

    report.main(["OK"])

    assert captured["datum"] == report.DEFAULT_SERIES_DATUM
    actual = read_series_workbook(transient_dir / report.TRANSIENT_WORKBOOK)["OK"]
    assert actual["series_flow_ref_m3_per_h"] == pytest.approx(q95, rel=1e-6)
    assert actual["series_growth_per_m3"] > 0.0
    assert actual["series_growth_per_m3"] == pytest.approx(1.0e-5, rel=0.15)


def test_series_baseline_present_at_zero_growth():
    # g=0 removes only the GROWTH; the fixed 0.5 m baseline (dp_ref/Q_ref * Q) remains whenever a
    # volume column and a positive reference flow are present. Guards a "g==0 => no clogging"
    # short-circuit that would silently drop the baseline for every freshly-seeded strang.
    ci = _ci()
    index = pd.date_range("2015-01-01", periods=20, freq="10D")
    flow = np.full(index.size, 12.0)
    volume = np.linspace(0.0, 5.0e4, index.size)
    coeff = default_transient_coefficients(
        series_dp_ref_m=0.5, series_flow_ref_m3_per_h=16.0, series_growth_per_m3=0.0
    )
    dfm = pd.DataFrame({"Q": flow, "cumulative_volume_m3": volume}, index=index)

    hl = report.series_head_loss(coeff, dfm)
    np.testing.assert_allclose(hl.to_numpy(), 0.5 / 16.0 * flow, rtol=1e-12)
    assert np.all(hl.to_numpy() > 0.0)

    # The integrated model with the baseline exceeds the volume-less model by exactly the baseline.
    with_baseline = transient_drawdown_for_coefficients(80.0, 150.0, dfm, coeff, ci)
    no_series = transient_drawdown_for_coefficients(
        80.0, 150.0, pd.DataFrame({"Q": flow}, index=index), coeff, ci
    )
    np.testing.assert_allclose(
        (with_baseline - no_series).to_numpy(), 0.5 / 16.0 * flow, rtol=1e-10
    )


def test_fit_rails_growth_to_positivity_bound():
    # Data that want more growth than positivity allows: g must rail to g_upper = 1/|V_min|
    # (the positivity bound), not run away -- catches g_upper_pos -> inf.
    ci = _ci()
    index = pd.date_range("2014-06-01", periods=80, freq="10D")
    flow = _series_flow(index)
    volume = report.cumulative_extracted_volume_m3(
        pd.Series(flow, index=index), datum=pd.Timestamp("2015-01-01")
    ).to_numpy()
    q95 = float(np.nanpercentile(flow, report.SERIES_FLOW_REF_PERCENTILE))
    g_upper_pos = 1.0 / abs(float(volume.min()))
    g_upper_cap = (report.SERIES_MAX_GROWTH_FACTOR - 1.0) / float(volume.max())
    assert g_upper_pos < g_upper_cap  # positivity is the binding bound here
    infeasible_growth = 5.0 * g_upper_pos

    truth = default_transient_coefficients(
        kd_ref_m2_per_d=80.0, leakage_resistance_d=150.0,
        series_flow_ref_m3_per_h=q95, series_growth_per_m3=infeasible_growth,
    )
    dft = pd.DataFrame({"Q": flow, "cumulative_volume_m3": volume}, index=index)
    observed = transient_drawdown_for_coefficients(
        80.0, 150.0, dft, truth, ci, series_growth_per_m3=infeasible_growth
    )
    dfm = pd.DataFrame(
        {"Q": flow, "drawdown_aquifer": observed.to_numpy(dtype=float), "cumulative_volume_m3": volume},
        index=index,
    )
    seed = truth.copy()
    seed["series_growth_per_m3"] = 0.0

    result = fit_transient_coefficients(dfm, seed, ci)

    fitted_g = result["coefficients"]["series_growth_per_m3"]
    assert fitted_g <= g_upper_pos * (1.0 + 1e-9)  # never exceeds the positivity bound
    assert fitted_g == pytest.approx(g_upper_pos, rel=0.1)  # railed to it


def test_load_observations_aligns_signed_volume_onto_model_index(monkeypatch):
    # The signed cumulative volume is built on the native masked flow BEFORE the resample and
    # interpolated onto the dropped-row model index -- so throughput during dropped intervals still
    # counts. Catches zeroing the alignment (which no other test reaches).
    index = pd.date_range("2015-01-01", periods=60, freq="h")
    flow = np.full(index.size, 10.0)
    raw = pd.DataFrame(
        {
            "Datum": index,
            "Q": flow,
            "gws0": np.zeros(index.size),
            "gws1": np.full(index.size, -2.0),
            "pandpeil": np.zeros(index.size),
        }
    )
    # A mid-record block with non-positive aquifer drawdown -> dropped by the resample mask.
    raw.loc[20:30, "pandpeil"] = -5.0
    monkeypatch.setattr(report.pd, "read_feather", lambda fp: raw.copy())
    monkeypatch.setattr(report, "get_false_measurements", lambda *a, **k: np.zeros(index.size, dtype=bool))

    dfm = report.load_observations(
        "Q100", _ci(), _filter_coefficients(), series_datum=pd.Timestamp("2015-01-01")
    )

    native = report.cumulative_extracted_volume_m3(
        pd.Series(flow, index=index), datum=pd.Timestamp("2015-01-01")
    )
    expected = np.interp(
        dfm.index.astype("int64").to_numpy(),
        index.astype("int64").to_numpy(),
        native.to_numpy(dtype=float),
    )
    np.testing.assert_allclose(dfm["cumulative_volume_m3"].to_numpy(), expected, rtol=1e-9)
    assert np.all(np.diff(dfm["cumulative_volume_m3"].to_numpy()) > 0.0)  # counts dropped throughput


def test_cumulative_extracted_volume_constant_flow_magnitude():
    index = pd.date_range("2015-01-01", periods=5, freq="D")
    q = 10.0
    v = report.cumulative_extracted_volume_m3(pd.Series(np.full(index.size, q), index=index), datum=index[0])
    hours = (index - index[0]).total_seconds().to_numpy() / 3600.0
    np.testing.assert_allclose(v.to_numpy(), q * hours, rtol=0.0, atol=1e-9)  # datum=start -> V=q*hours


def test_cumulative_extracted_volume_clips_negative_flow():
    index = pd.date_range("2015-01-01", periods=4, freq="D")
    flow = pd.Series([10.0, -50.0, 10.0, 10.0], index=index)
    v = report.cumulative_extracted_volume_m3(flow, datum=index[0])
    assert np.all(np.diff(v.to_numpy()) >= 0.0)  # negative flow clipped -> monotonic


def test_cumulative_extracted_volume_rejects_unsorted_index():
    index = pd.DatetimeIndex(["2015-01-03", "2015-01-01", "2015-01-02", "2015-01-04"])
    with pytest.raises(ValueError, match="sorted"):
        report.cumulative_extracted_volume_m3(pd.Series(np.full(4, 10.0), index=index), datum=index.min())


def test_cumulative_extracted_volume_rejects_non_datetime_index():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        report.cumulative_extracted_volume_m3(pd.Series([1.0, 2.0, 3.0]), datum=pd.Timestamp("2015-01-01"))


def _series_temperature_coefficients(**overrides):
    # method="sin" with a seasonal amplitude -> a TIME-VARYING viscosity ratio (!= 1), so the
    # element-wise multiply cannot be faked by a scalar broadcast.
    kwargs = dict(
        series_dp_ref_m=0.5,
        series_flow_ref_m3_per_h=16.0,
        series_growth_per_m3=2.0e-5,
        temperature_method="sin",
        temperature_mean_degc=12.0,
        temperature_delta_degc=8.0,
        temperature_ref_degc=12.0,
    )
    kwargs.update(overrides)
    return default_transient_coefficients(**kwargs)


def _positive_series_dfm(index):
    # Strictly positive flow and strictly positive growth factor (1 + g*V > 0), so the series head
    # loss is never 0 and ratios are well-posed.
    flow = np.full(index.size, 12.0)
    volume = np.linspace(-2.0e4, 8.0e4, index.size)  # 1 + 2e-5*V in [0.6, 2.6]
    return pd.DataFrame({"Q": flow, "cumulative_volume_m3": volume}, index=index)


def test_series_head_loss_scales_with_temperature_viscosity():
    # dp_series == (dp_ref/Q_ref)*Q*(1+g*V) * model_viscratio(index), element-wise, with a
    # time-varying viscratio. Kills: not-applied, inverse-direction, baseline-only, and
    # collapse-to-scalar/broadcast.
    index = pd.date_range("2015-01-01", periods=24, freq="15D")  # ~1 yr -> seasonal viscratio
    coeff = _series_temperature_coefficients()
    dfm = _positive_series_dfm(index)

    viscratio = coeff.wvpt.model_viscratio(index).to_numpy(dtype=float)
    assert viscratio.min() < 0.98 and viscratio.max() > 1.02  # genuinely time-varying, != 1

    flow = dfm["Q"].to_numpy(dtype=float)
    volume = dfm["cumulative_volume_m3"].to_numpy(dtype=float)
    expected = 0.5 / 16.0 * flow * (1.0 + 2.0e-5 * volume) * viscratio
    np.testing.assert_allclose(report.series_head_loss(coeff, dfm).to_numpy(), expected, rtol=1e-12)


def test_series_and_aquifer_share_the_same_viscosity_factor():
    # The series scaling factor equals the aquifer's kD viscosity factor (kD_ref_model/kD_model) --
    # the physical rationale: the series stiffens with the SAME viscosity the aquifer uses.
    index = pd.date_range("2015-01-01", periods=24, freq="15D")
    coeff = _series_temperature_coefficients()
    dfm = _positive_series_dfm(index)

    flow = dfm["Q"].to_numpy(dtype=float)
    volume = dfm["cumulative_volume_m3"].to_numpy(dtype=float)
    reference_temp_series = 0.5 / 16.0 * flow * (1.0 + 2.0e-5 * volume)  # strictly > 0
    aquifer_viscratio = (
        coeff.wvpt.kD_ref_model(index) / coeff.wvpt.kD_model(index)
    ).to_numpy(dtype=float)

    series_factor = report.series_head_loss(coeff, dfm).to_numpy() / reference_temp_series
    np.testing.assert_allclose(series_factor, aquifer_viscratio, rtol=1e-12)


def test_series_contribution_through_pipeline_scales_with_temperature():
    # Independent-truth analog of test_series_baseline_present_at_zero_growth, for sin + g != 0:
    # through the FULL transient_drawdown_for_coefficients pipeline, the series contribution equals
    # an INDEPENDENTLY built (dp_ref/Q_ref)*Q*(1+g*V)*model_viscratio. This catches drop/inverse
    # viscratio -- which a self-referential fit test (truth and fit share series_head_loss) cannot,
    # because the viscratio error cancels between the synthetic truth and the fitted model.
    ci = _ci()
    index = pd.date_range("2015-01-01", periods=24, freq="15D")
    flow = np.full(index.size, 12.0)
    volume = np.linspace(-2.0e4, 8.0e4, index.size)
    g = 2.0e-5
    coeff = _series_temperature_coefficients(series_growth_per_m3=g)  # dp_ref=0.5, q_ref=16, sin
    dfm_with = pd.DataFrame({"Q": flow, "cumulative_volume_m3": volume}, index=index)
    dfm_without = pd.DataFrame({"Q": flow}, index=index)  # no volume column -> no series term

    # series_growth_per_m3=g must be passed (it overrides the coefficient growth); the aquifer part
    # is identical between the two calls (same index, same Q), so their difference is the series term.
    with_series = transient_drawdown_for_coefficients(80.0, 150.0, dfm_with, coeff, ci, series_growth_per_m3=g)
    without_series = transient_drawdown_for_coefficients(80.0, 150.0, dfm_without, coeff, ci, series_growth_per_m3=g)

    viscratio = coeff.wvpt.model_viscratio(index).to_numpy(dtype=float)
    expected_series = 0.5 / 16.0 * flow * (1.0 + g * volume) * viscratio
    np.testing.assert_allclose((with_series - without_series).to_numpy(), expected_series, rtol=1e-9)
