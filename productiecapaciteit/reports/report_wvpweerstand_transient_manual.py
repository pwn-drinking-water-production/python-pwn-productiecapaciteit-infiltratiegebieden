"""Manually calibrate transient WVP coefficients and plot diagnostics."""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from productiecapaciteit import plot_styles_dir, results_dir
from productiecapaciteit.reports.report_wvpweerstand_transient import (
    CONFIG_FN,
    DEFAULT_KD_BOUNDS_M2_PER_D,
    LEGACY_RESULTS_SUBDIR,
    LEGACY_TRANSIENT_START_WORKBOOK,
    LEGACY_TRANSIENT_WORKBOOK,
    RESULTS_SUBDIR,
    TRANSIENT_START_WORKBOOK,
    TRANSIENT_WORKBOOK,
    configure_logging,
    default_transient_coefficients,
    first_existing_path,
    load_observations,
    plot_fit,
    read_series_workbook,
    transient_coefficients_from_sheet,
    write_series_workbook,
)
from productiecapaciteit.reports.report_wvpweerstand_transient_initial import transient_coefficients_from_wvp
from productiecapaciteit.src.strang_analyse_fun2 import get_config

MANUAL_RESULTS_SUBDIR = "handmatige_kalibratie"
MANUAL_SUMMARY_CSV = "Wvptweerstand_handmatige_kalibratie.csv"
MANUAL_WORKBOOK = "Wvptweerstand_handmatige_modelcoefficienten.xlsx"
MANUAL_LOG = "Wvptweerstand_handmatige_kalibratie.log"

# Edit these values when manually calibrating. kD_ref is recalculated from the
# steady WVP coefficients by default, so changing r_well/S/leakage preserves the
# steady resistance while changing the transient dynamics.
MANUAL_WELL_RADIUS_M = (0.10, 0.25, 0.40)
MANUAL_STORAGE_COEFFICIENT = (0.10, 0.20, 0.30)
MANUAL_CASES = [
    {
        "name": f"r_well_{well_radius_m:.2f}m_S_{storage_coefficient:.1f}",
        "well_radius_m": well_radius_m,
        "storage_coefficient": storage_coefficient,
    }
    for well_radius_m in MANUAL_WELL_RADIUS_M
    for storage_coefficient in MANUAL_STORAGE_COEFFICIENT
]


def safe_name(value):
    """Return a filesystem-safe name for manual case folders."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def excel_sheet_name(strang, case_name, used_names):
    """Return a unique Excel sheet name within the 31-character limit."""
    base = safe_name(f"{strang}_{case_name}")[:31] or "case"
    candidate = base
    suffix = 1
    while candidate in used_names:
        tail = f"_{suffix}"
        candidate = f"{base[: 31 - len(tail)]}{tail}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def configure_manual_logging(output_dir):
    """Configure manual-calibration logging without overwriting fit logs."""
    logger = configure_logging(output_dir)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(output_dir / MANUAL_LOG, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def source_transient_workbook():
    """Return the preferred existing transient coefficient workbook."""
    output_dir = results_dir / RESULTS_SUBDIR
    legacy_output_dir = results_dir / LEGACY_RESULTS_SUBDIR
    return first_existing_path(
        output_dir / TRANSIENT_WORKBOOK,
        output_dir / TRANSIENT_START_WORKBOOK,
        legacy_output_dir / LEGACY_TRANSIENT_WORKBOOK,
        legacy_output_dir / LEGACY_TRANSIENT_START_WORKBOOK,
    )


def coefficients_for_manual_case(
    steady_coefficients,
    ci,
    base_transient_coefficients,
    manual_case,
    *,
    kd_bounds_m2_per_d=DEFAULT_KD_BOUNDS_M2_PER_D,
):
    """Build WVPT coefficients for one manual case."""
    transient_seed = transient_coefficients_from_sheet(base_transient_coefficients).copy()
    for key in (
        "well_radius_m",
        "storage_coefficient",
        "leakage_resistance_d",
        "temperature_mean_degC",
        "temperature_delta_degC",
        "temperature_ref_degC",
        "temperature_time_offset_d",
        "temperature_method",
    ):
        if key in manual_case:
            transient_seed[key] = manual_case[key]

    if manual_case.get("recalibrate_kd_ref_from_steady", True):
        coefficients = transient_coefficients_from_wvp(
            steady_coefficients,
            ci,
            transient_seed,
            kd_bounds_m2_per_d=kd_bounds_m2_per_d,
        )
    else:
        coefficients = transient_seed

    for key in (
        "kD_ref_m2_per_d",
        "kD_ref_slope_m2_per_d_per_d",
        "kD_ref_datum",
    ):
        if key in manual_case:
            coefficients[key] = manual_case[key]

    return transient_coefficients_from_sheet(coefficients)


def evaluate_manual_case(
    strang,
    ci,
    dfm,
    steady_coefficients,
    base_transient_coefficients,
    manual_case,
    output_dir,
    *,
    kd_bounds_m2_per_d=DEFAULT_KD_BOUNDS_M2_PER_D,
):
    """Evaluate and plot one manual calibration case."""
    case_name = str(manual_case["name"])
    coefficients = coefficients_for_manual_case(
        steady_coefficients,
        ci,
        base_transient_coefficients,
        manual_case,
        kd_bounds_m2_per_d=kd_bounds_m2_per_d,
    )
    modeled = (
        -coefficients.wvpt.dp_model(
            dfm.index,
            dfm.Q,
            ci.nput,
            ci.dx_tussenputten,
            ci.r_mirrorwel,
            target_well_index=manual_case.get("target_well_index"),
            initial_condition=manual_case.get("initial_condition", "steady"),
            frac_step_max=manual_case.get("frac_step_max", 0.95),
            tmax_days_cap=manual_case.get("tmax_days_cap"),
            max_workers=manual_case.get("max_workers"),
        )
    ).rename("wvpt_drawdown")
    residuals = modeled - dfm.drawdown_aquifer
    case_dir = Path(output_dir) / safe_name(strang) / safe_name(case_name)
    fig_path = plot_fit(
        f"{strang} {case_name}",
        dfm,
        coefficients,
        modeled,
        case_dir,
        ci,
        target_well_index=manual_case.get("target_well_index"),
    )

    return {
        "coefficients": coefficients,
        "modeled": modeled,
        "residuals": residuals.rename("wvpt_manual_residual"),
        "summary": {
            "strang": strang,
            "case": case_name,
            "plot": str(fig_path),
            "well_radius_m": float(coefficients["well_radius_m"]),
            "storage_coefficient": float(coefficients["storage_coefficient"]),
            "leakage_resistance_d": float(coefficients["leakage_resistance_d"]),
            "kD_ref_m2_per_d": float(coefficients["kD_ref_m2_per_d"]),
            "residual_mean_m": float(residuals.mean()),
            "residual_std_m": float(residuals.std()),
            "residual_rmse_m": float(np.sqrt(np.mean(np.square(residuals)))),
            "residual_abs_max_m": float(residuals.abs().max()),
            "modeled_min_m": float(modeled.min()),
            "modeled_max_m": float(modeled.max()),
        },
    }


def main(strangen=None, manual_cases=None):
    """Run manual WVPT calibration cases for selected strangen."""
    output_dir = results_dir / RESULTS_SUBDIR / MANUAL_RESULTS_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_manual_logging(output_dir)

    try:
        import matplotlib.pyplot as plt

        plt.style.use(plot_styles_dir / "unhcrpyplotstyle.mplstyle")
        plt.style.use(plot_styles_dir / "types" / "line.mplstyle")
    except OSError:
        logger.warning("Could not load plot style files; using matplotlib defaults")

    config = get_config(CONFIG_FN)
    if strangen is not None:
        if isinstance(strangen, str):
            strangen = [strangen]
        config = config.loc[list(strangen)]

    manual_cases = MANUAL_CASES if manual_cases is None else manual_cases

    steady_fp = results_dir / "Wvpweerstand" / "Wvpweerstand_modelcoefficienten.xlsx"
    filter_fp = results_dir / "Filterweerstand" / "Filterweerstand_modelcoefficienten.xlsx"
    steady_sheets = read_series_workbook(steady_fp, required=True)
    transient_sheets = read_series_workbook(source_transient_workbook())
    filter_sheets = pd.read_excel(filter_fp, sheet_name=None)

    rows = []
    coefficient_sheets = {}
    used_sheet_names = set()
    for strang, ci in config.iterrows():
        logger.info("Strang: %s", strang)
        try:
            dfm = load_observations(strang, ci, filter_sheets[strang])
            base_transient = transient_sheets.get(strang, default_transient_coefficients())
            for manual_case in manual_cases:
                result = evaluate_manual_case(
                    strang,
                    ci,
                    dfm,
                    steady_sheets[strang],
                    base_transient,
                    manual_case,
                    output_dir,
                )
                rows.append(result["summary"])
                sheet_name = excel_sheet_name(strang, manual_case["name"], used_sheet_names)
                coefficient_sheets[sheet_name] = result["coefficients"]
                logger.info("Saved manual case %s", result["summary"]["plot"])
        except (KeyError, FileNotFoundError, ValueError, RuntimeError):
            logger.exception("Skipping %s after manual WVPT evaluation failure", strang)

    summary = pd.DataFrame(rows)
    summary_fp = output_dir / MANUAL_SUMMARY_CSV
    summary.to_csv(summary_fp, index=False)
    if coefficient_sheets:
        write_series_workbook(output_dir / MANUAL_WORKBOOK, coefficient_sheets)
    logger.info("Wrote manual WVPT calibration summary to %s", summary_fp)
    return summary


if __name__ == "__main__":
    main()
