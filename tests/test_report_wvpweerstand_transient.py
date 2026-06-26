from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import productiecapaciteit.reports.report_wvpweerstand_transient as report
import productiecapaciteit.reports.report_wvpweerstand_transient_initial as initial_report
import productiecapaciteit.reports.report_wvpweerstand_transient_manual as manual_report
import productiecapaciteit.src.weerstand_pandasaccessors as accessor_module
from productiecapaciteit.reports.report_wvpweerstand_transient import (
    default_transient_coefficients,
    fit_leakage_resistance,
    read_series_workbook,
    reconstruct_aquifer_drawdown,
    resample_transient_observations,
    transient_drawdown_for_leakage,
    write_series_workbook,
)
from productiecapaciteit.reports.report_wvpweerstand_transient_initial import transient_coefficients_from_wvp
from productiecapaciteit.src.wvp_transient_funs import (
    build_multiwell_geometry,
    steady_multiwell_resistance_from_kd,
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


def _steady_coefficients_for_known_kd(kd=100.0, leakage_resistance_d=120.0):
    resistance = steady_multiwell_resistance_from_kd(
        kd,
        [(1.0, 1.0)],
        nput=1,
        leakage_resistance_d=leakage_resistance_d,
        well_radius_m=0.2,
    )
    return pd.Series({
        "offset": -float(resistance),
        "offset_datum": pd.Timestamp("2020-01-01"),
        "slope": 0.0,
        "temp_mean": 12.0,
        "temp_delta": 0.0,
        "time_offset": 0.0,
        "method": "Niet",
        "temp_ref": 12.0,
    })


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
    assert "wvpt_fit_success" not in actual["Q100"].index


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


def test_transient_coefficients_from_wvp_creates_standalone_accessor_sheet():
    steady = _steady_coefficients_for_known_kd()
    transient = default_transient_coefficients(leakage_resistance_d=123.0)
    ci = _ci()

    converted = transient_coefficients_from_wvp(steady, ci, transient)

    assert converted.index.is_unique
    assert set(converted.index) == set(report.TRANSIENT_COEFFICIENT_KEYS)
    assert "offset" not in converted.index
    assert converted["kD_ref_m2_per_d"] > 0.0
    assert converted["kD_ref_slope_m2_per_d_per_d"] == pytest.approx(0.0)
    assert converted["leakage_resistance_d"] == pytest.approx(123.0)
    multiwell, _ = build_multiwell_geometry(
        ci.dx_tussenputten,
        ci.r_mirrorwel,
        ci.nput,
        distance_scale=1.0 / converted["well_radius_m"],
    )
    converted_resistance = steady_multiwell_resistance_from_kd(
        converted["kD_ref_m2_per_d"],
        multiwell,
        ci.nput,
        converted["leakage_resistance_d"],
        converted["well_radius_m"],
    )
    expected_resistance = -float(
        steady.wvp.a_model_reftemp(pd.DatetimeIndex([steady["offset_datum"]])).iloc[0]
    )
    assert converted_resistance == pytest.approx(expected_resistance)


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


def test_transient_drawdown_for_leakage_uses_zero_initial_condition(monkeypatch):
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    dfm = pd.DataFrame({"Q": np.ones(index.size)}, index=index)
    coefficients = default_transient_coefficients()
    captured = {}

    def fake_objective(args, return_result=False, **pextra):
        captured["initial_condition"] = pextra["initial_condition"]
        return np.zeros(index.size)

    monkeypatch.setattr(accessor_module, "objective", fake_objective)

    actual = transient_drawdown_for_leakage(
        100.0,
        dfm,
        coefficients,
        _ci(),
    )

    assert captured["initial_condition"] == "zero"
    np.testing.assert_allclose(actual.to_numpy(), 0.0)


def test_fit_leakage_resistance_recovers_synthetic_value():
    true_leakage = 80.0
    steady = _steady_coefficients_for_known_kd(
        kd=100.0,
        leakage_resistance_d=true_leakage,
    )
    initial_transient = default_transient_coefficients(leakage_resistance_d=200.0)
    true_transient = default_transient_coefficients(leakage_resistance_d=true_leakage)
    ci = _ci()
    index = pd.date_range("2020-01-01", periods=18, freq="D")
    flow = np.r_[0.0, np.full(index.size - 1, 10.0)]

    df_a_true = transient_coefficients_from_wvp(steady, ci, true_transient)
    observed = transient_drawdown_for_leakage(
        true_leakage,
        pd.DataFrame({"Q": flow}, index=index),
        df_a_true,
        ci,
    )
    dfm = pd.DataFrame(
        {
            "Q": flow,
            "drawdown_aquifer": observed.to_numpy(dtype=float),
        },
        index=index,
    )
    df_a_initial = df_a_true.copy()
    df_a_initial["leakage_resistance_d"] = initial_transient["leakage_resistance_d"]

    result = fit_leakage_resistance(
        dfm,
        df_a_initial,
        ci,
        leakage_bounds_d=(10.0, 500.0),
    )

    assert result["coefficients"]["leakage_resistance_d"] == pytest.approx(
        true_leakage,
        rel=0.05,
    )
    assert result["optimizer_result"].success is True
    assert set(result["coefficients"].index) == set(report.TRANSIENT_COEFFICIENT_KEYS)
    assert "wvpt_fit_success" not in result["coefficients"].index
    assert result["residuals"].abs().max() < 1e-5
    assert result["residual_innovations"].abs().max() < 1e-5


def test_fit_leakage_resistance_uses_residual_innovations(monkeypatch):
    true_leakage = 80.0
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    trend = np.arange(index.size, dtype=float)
    dfm = pd.DataFrame(
        {
            "Q": np.full(index.size, 10.0),
            "drawdown_aquifer": 5.0 + true_leakage / 10.0 * trend,
        },
        index=index,
    )
    steady = _steady_coefficients_for_known_kd()
    initial = default_transient_coefficients(leakage_resistance_d=200.0)
    df_a_initial = transient_coefficients_from_wvp(steady, _ci(), initial)

    def fake_drawdown(leakage_resistance_d, dfm, df_a_wvpt, ci, **kwargs):
        return pd.Series(
            leakage_resistance_d / 10.0 * trend,
            index=dfm.index,
        )

    monkeypatch.setattr(report, "transient_drawdown_for_leakage", fake_drawdown)

    result = fit_leakage_resistance(
        dfm,
        df_a_initial,
        _ci(),
        leakage_bounds_d=(10.0, 500.0),
    )

    assert result["coefficients"]["leakage_resistance_d"] == pytest.approx(
        true_leakage,
        rel=1e-3,
    )
    np.testing.assert_allclose(result["residuals"].to_numpy(), -5.0, atol=1e-6)
    np.testing.assert_allclose(result["residual_innovations"].to_numpy(), 0.0, atol=1e-6)


def test_fit_leakage_resistance_penalizes_infeasible_kd_candidates(monkeypatch):
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    trend = np.arange(index.size, dtype=float)
    dfm = pd.DataFrame(
        {
            "Q": np.full(index.size, 10.0),
            "drawdown_aquifer": 8.0 * trend,
        },
        index=index,
    )
    steady = _steady_coefficients_for_known_kd()
    initial = default_transient_coefficients(leakage_resistance_d=200.0)
    df_a_initial = transient_coefficients_from_wvp(steady, _ci(), initial)

    def fake_drawdown(leakage_resistance_d, dfm, df_a_wvpt, ci, **kwargs):
        if leakage_resistance_d > 100.0:
            raise ValueError("cannot bracket kD")
        return pd.Series(
            leakage_resistance_d / 10.0 * trend,
            index=dfm.index,
        )

    monkeypatch.setattr(report, "transient_drawdown_for_leakage", fake_drawdown)

    result = fit_leakage_resistance(
        dfm,
        df_a_initial,
        _ci(),
        leakage_bounds_d=(10.0, 500.0),
    )

    assert result["coefficients"]["leakage_resistance_d"] == pytest.approx(80.0, rel=1e-3)
    assert result["optimizer_result"].success is True


def test_fit_leakage_resistance_reports_when_no_candidate_is_feasible(monkeypatch):
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    dfm = pd.DataFrame(
        {
            "Q": np.full(index.size, 10.0),
            "drawdown_aquifer": np.ones(index.size),
        },
        index=index,
    )
    df_a_initial = transient_coefficients_from_wvp(
        _steady_coefficients_for_known_kd(),
        _ci(),
        default_transient_coefficients(leakage_resistance_d=200.0),
    )

    def fake_drawdown(leakage_resistance_d, dfm, df_a_wvpt, ci, **kwargs):
        raise ValueError("cannot bracket kD")

    monkeypatch.setattr(report, "transient_drawdown_for_leakage", fake_drawdown)

    with pytest.raises(RuntimeError, match="No feasible leakage_resistance_d candidate"):
        fit_leakage_resistance(
            dfm,
            df_a_initial,
            _ci(),
            leakage_bounds_d=(10.0, 500.0),
        )


def test_initial_report_writes_standalone_starting_workbook(monkeypatch, tmp_path):
    config = pd.DataFrame(
        {
            "nput": [1],
            "dx_tussenputten": [15.0],
            "r_mirrorwel": [[]],
        },
        index=["OK"],
    )
    monkeypatch.setattr(initial_report, "results_dir", tmp_path)
    monkeypatch.setattr(initial_report, "get_config", lambda fn: config)

    steady_dir = tmp_path / "Wvpweerstand"
    steady_dir.mkdir()
    steady = _steady_coefficients_for_known_kd()
    write_series_workbook(
        steady_dir / "Wvpweerstand_modelcoefficienten.xlsx",
        {"OK": steady},
    )

    initial_report.main(["OK"])

    start_fp = tmp_path / report.RESULTS_SUBDIR / report.TRANSIENT_START_WORKBOOK
    actual = read_series_workbook(start_fp)["OK"]

    assert "offset" not in actual.index
    assert actual["kD_ref_m2_per_d"] > 0.0
    assert actual["kD_ref_slope_m2_per_d_per_d"] == pytest.approx(0.0)
    assert actual.wvpt.leakage_resistance_d == pytest.approx(report.DEFAULT_LEAKAGE_RESISTANCE_D)


def test_main_logs_failed_strang_and_persists_successful_standalone_workbook(
    monkeypatch,
    tmp_path,
):
    config = pd.DataFrame(
        {
            "nput": [1, 1],
            "dx_tussenputten": [15.0, 15.0],
            "r_mirrorwel": [[], []],
        },
        index=["OK", "BAD"],
    )
    monkeypatch.setattr(report, "results_dir", tmp_path)
    monkeypatch.setattr(report, "plot_styles_dir", tmp_path)
    monkeypatch.setattr(report, "get_config", lambda fn: config)
    monkeypatch.setattr(report.plt.style, "use", lambda *args, **kwargs: None)
    monkeypatch.setattr(report, "plot_fit", lambda *args, **kwargs: tmp_path / "plot.png")

    steady = _steady_coefficients_for_known_kd()
    ok_start = transient_coefficients_from_wvp(steady, _ci(), default_transient_coefficients())
    bad_start = transient_coefficients_from_wvp(steady, _ci(), default_transient_coefficients())
    transient_dir = tmp_path / report.RESULTS_SUBDIR
    transient_dir.mkdir()
    write_series_workbook(
        transient_dir / report.TRANSIENT_START_WORKBOOK,
        {"OK": ok_start, "BAD": bad_start},
    )

    filter_dir = tmp_path / "Filterweerstand"
    filter_dir.mkdir()
    with pd.ExcelWriter(filter_dir / "Filterweerstand_modelcoefficienten.xlsx") as writer:
        _filter_coefficients().to_excel(writer, sheet_name="OK", index=False)
        _filter_coefficients().to_excel(writer, sheet_name="BAD", index=False)

    index = pd.date_range("2020-01-01", periods=2, freq="D")

    def fake_load_observations(strang, ci, df_a_filter):
        return pd.DataFrame(
            {
                "Q": [10.0, 10.0],
                "drawdown_aquifer": [1.0, 1.0],
            },
            index=index,
        )

    def fake_fit_leakage_resistance(dfm, df_a_wvpt, ci, **kwargs):
        if ci.name == "BAD":
            raise ValueError("fit failed")
        return {
            "coefficients": default_transient_coefficients(leakage_resistance_d=77.0),
            "modeled": pd.Series([1.0, 1.0], index=dfm.index),
            "residuals": pd.Series([0.0, 0.0], index=dfm.index),
            "optimizer_result": None,
        }

    monkeypatch.setattr(report, "load_observations", fake_load_observations)
    monkeypatch.setattr(report, "fit_leakage_resistance", fake_fit_leakage_resistance)

    report.main(["OK", "BAD"])

    transient_fp = tmp_path / report.RESULTS_SUBDIR / report.TRANSIENT_WORKBOOK
    actual = read_series_workbook(transient_fp)
    log_text = (tmp_path / report.RESULTS_SUBDIR / report.TRANSIENT_LOG).read_text()

    assert set(actual) == {"OK", "BAD"}
    assert "offset" not in actual["OK"].index
    assert set(actual["OK"].index) == set(report.TRANSIENT_COEFFICIENT_KEYS)
    assert actual["OK"]["leakage_resistance_d"] == pytest.approx(77.0)
    assert actual["OK"].wvpt.leakage_resistance_d == pytest.approx(77.0)
    assert actual["BAD"]["leakage_resistance_d"] == pytest.approx(bad_start["leakage_resistance_d"])
    assert "BAD" in log_text
    assert "fit failed" in log_text


def test_main_reads_legacy_transient_workbook_and_writes_new_name(
    monkeypatch,
    tmp_path,
):
    config = pd.DataFrame(
        {
            "nput": [1],
            "dx_tussenputten": [15.0],
            "r_mirrorwel": [[]],
        },
        index=["OK"],
    )
    monkeypatch.setattr(report, "results_dir", tmp_path)
    monkeypatch.setattr(report, "plot_styles_dir", tmp_path)
    monkeypatch.setattr(report, "get_config", lambda fn: config)
    monkeypatch.setattr(report.plt.style, "use", lambda *args, **kwargs: None)
    monkeypatch.setattr(report, "plot_fit", lambda *args, **kwargs: tmp_path / "plot.png")

    steady = _steady_coefficients_for_known_kd()
    legacy_start = transient_coefficients_from_wvp(steady, _ci(), default_transient_coefficients())
    legacy_dir = tmp_path / report.LEGACY_RESULTS_SUBDIR
    legacy_dir.mkdir()
    write_series_workbook(
        legacy_dir / report.LEGACY_TRANSIENT_START_WORKBOOK,
        {"OK": legacy_start},
    )

    filter_dir = tmp_path / "Filterweerstand"
    filter_dir.mkdir()
    with pd.ExcelWriter(filter_dir / "Filterweerstand_modelcoefficienten.xlsx") as writer:
        _filter_coefficients().to_excel(writer, sheet_name="OK", index=False)

    index = pd.date_range("2020-01-01", periods=2, freq="D")

    monkeypatch.setattr(
        report,
        "load_observations",
        lambda strang, ci, df_a_filter: pd.DataFrame(
            {
                "Q": [10.0, 10.0],
                "drawdown_aquifer": [1.0, 1.0],
            },
            index=index,
        ),
    )
    monkeypatch.setattr(
        report,
        "fit_leakage_resistance",
        lambda dfm, df_a_wvpt, ci, **kwargs: {
            "coefficients": default_transient_coefficients(leakage_resistance_d=77.0),
            "modeled": pd.Series([1.0, 1.0], index=dfm.index),
            "residuals": pd.Series([0.0, 0.0], index=dfm.index),
            "optimizer_result": None,
        },
    )

    report.main(["OK"])

    new_fp = tmp_path / report.RESULTS_SUBDIR / report.TRANSIENT_WORKBOOK
    assert new_fp.exists()
    assert not (tmp_path / report.RESULTS_SUBDIR / report.LEGACY_TRANSIENT_WORKBOOK).exists()


def test_manual_report_default_cases_cover_radius_storage_grid():
    actual = {(case["well_radius_m"], case["storage_coefficient"]) for case in manual_report.MANUAL_CASES}

    assert actual == {(r_well, storage) for r_well in (0.10, 0.25, 0.40) for storage in (0.10, 0.20, 0.30)}


def test_manual_report_writes_summary_plots_and_manual_workbook(monkeypatch, tmp_path):
    config = pd.DataFrame(
        {
            "nput": [1],
            "dx_tussenputten": [15.0],
            "r_mirrorwel": [[]],
        },
        index=["OK"],
    )
    monkeypatch.setattr(manual_report, "results_dir", tmp_path)
    monkeypatch.setattr(manual_report, "plot_styles_dir", tmp_path)
    monkeypatch.setattr(manual_report, "get_config", lambda fn: config)

    steady_dir = tmp_path / "Wvpweerstand"
    steady_dir.mkdir()
    steady = _steady_coefficients_for_known_kd()
    write_series_workbook(
        steady_dir / "Wvpweerstand_modelcoefficienten.xlsx",
        {"OK": steady},
    )
    transient_dir = tmp_path / report.RESULTS_SUBDIR
    transient_dir.mkdir()
    write_series_workbook(
        transient_dir / report.TRANSIENT_START_WORKBOOK,
        {"OK": transient_coefficients_from_wvp(steady, _ci(), default_transient_coefficients())},
    )
    filter_dir = tmp_path / "Filterweerstand"
    filter_dir.mkdir()
    with pd.ExcelWriter(filter_dir / "Filterweerstand_modelcoefficienten.xlsx") as writer:
        _filter_coefficients().to_excel(writer, sheet_name="OK", index=False)

    index = pd.date_range("2020-01-01", periods=3, freq="D")
    monkeypatch.setattr(
        manual_report,
        "load_observations",
        lambda strang, ci, df_a_filter: pd.DataFrame(
            {
                "Q": [0.0, 10.0, 10.0],
                "drawdown_aquifer": [1.0, 1.2, 1.3],
            },
            index=index,
        ),
    )

    def fake_plot_fit(strang, dfm, df_a_wvpt, modeled_drawdown, output_dir, ci, **kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f"{manual_report.safe_name(strang)}.png"
        fig_path.write_text("plot")
        return fig_path

    monkeypatch.setattr(manual_report, "plot_fit", fake_plot_fit)

    summary = manual_report.main(
        ["OK"],
        manual_cases=[
            {
                "name": "manual_r010",
                "well_radius_m": 0.1,
                "storage_coefficient": 0.2,
                "leakage_resistance_d": 120.0,
            },
            {
                "name": "manual_r040",
                "well_radius_m": 0.4,
                "storage_coefficient": 0.2,
                "leakage_resistance_d": 120.0,
            },
        ],
    )

    output_dir = tmp_path / report.RESULTS_SUBDIR / manual_report.MANUAL_RESULTS_SUBDIR
    assert (output_dir / manual_report.MANUAL_SUMMARY_CSV).exists()
    assert (output_dir / manual_report.MANUAL_WORKBOOK).exists()
    assert set(summary["case"]) == {"manual_r010", "manual_r040"}
    assert set(summary["well_radius_m"]) == {0.1, 0.4}
    assert all(Path(path).exists() for path in summary["plot"])
    manual_sheets = read_series_workbook(output_dir / manual_report.MANUAL_WORKBOOK)
    assert set(manual_sheets) == {"OK_manual_r010", "OK_manual_r040"}
    assert (tmp_path / report.RESULTS_SUBDIR / report.TRANSIENT_START_WORKBOOK).exists()
    assert not (tmp_path / report.RESULTS_SUBDIR / report.TRANSIENT_WORKBOOK).exists()
