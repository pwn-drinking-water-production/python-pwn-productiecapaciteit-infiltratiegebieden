"""Create standalone transient WVP starting coefficients from steady WVP sheets."""

import logging
from pathlib import Path

import pandas as pd

from productiecapaciteit import results_dir
from productiecapaciteit.reports.report_wvpweerstand_transient import (
    CONFIG_FN,
    DEFAULT_KD_BOUNDS_M2_PER_D,
    RESULTS_SUBDIR,
    TRANSIENT_INITIAL_LOG,
    TRANSIENT_START_WORKBOOK,
    default_transient_coefficients,
    normalize_transient_coefficients,
    read_series_workbook,
    sheet_to_series,
    write_series_workbook,
)
from productiecapaciteit.src.strang_analyse_fun2 import get_config
from productiecapaciteit.src.weerstand_pandasaccessors import WvpResistanceAccessor  # noqa: F401
from productiecapaciteit.src.wvp_transient_funs import build_multiwell_geometry, solve_steady_multiwell_kd


def configure_logging(output_dir):
    """Configure initialization logging after the output directory exists."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(
        output_dir / TRANSIENT_INITIAL_LOG,
        mode="w",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def transient_coefficients_from_wvp(
    steady_coefficients,
    ci,
    transient_coefficients=None,
    kd_bounds_m2_per_d=DEFAULT_KD_BOUNDS_M2_PER_D,
):
    """Create standalone WVPT coefficients from one steady WVP coefficient sheet."""
    steady = sheet_to_series(steady_coefficients)
    transient = (
        default_transient_coefficients()
        if transient_coefficients is None
        else normalize_transient_coefficients(transient_coefficients)
    )
    well_radius_m = float(transient["well_radius_m"])
    leakage_resistance_d = float(transient["leakage_resistance_d"])
    multiwell, _ = build_multiwell_geometry(
        ci.dx_tussenputten,
        ci.r_mirrorwel,
        ci.nput,
        distance_scale=1.0 / well_radius_m,
        include_self=True,
        self_distance=1.0,
    )
    reference_datum = pd.Timestamp(steady["offset_datum"])
    reference_resistance = -float(steady.wvp.a_model_reftemp([reference_datum]).iloc[0])
    kd_ref = solve_steady_multiwell_kd(
        reference_resistance,
        multiwell,
        ci.nput,
        leakage_resistance_d,
        well_radius_m,
        kd_min=kd_bounds_m2_per_d[0],
        kd_max=kd_bounds_m2_per_d[1],
    )
    return default_transient_coefficients(
        kd_ref_m2_per_d=kd_ref,
        kd_ref_slope_m2_per_d_per_d=0.0,
        kd_ref_datum=reference_datum,
        temperature_mean_degc=steady["temp_mean"],
        temperature_delta_degc=steady["temp_delta"],
        temperature_ref_degc=steady["temp_ref"],
        temperature_time_offset_d=steady["time_offset"],
        temperature_method=steady["method"],
        well_radius_m=well_radius_m,
        storage_coefficient=transient["storage_coefficient"],
        leakage_resistance_d=leakage_resistance_d,
    )


def main(strangen=None):
    """Write standalone WVPT starting coefficients for selected strangen."""
    output_dir = results_dir / RESULTS_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)

    config = get_config(CONFIG_FN)
    if strangen is not None:
        if isinstance(strangen, str):
            strangen = [strangen]
        config = config.loc[list(strangen)]

    steady_fp = results_dir / "Wvpweerstand" / "Wvpweerstand_modelcoefficienten.xlsx"
    start_fp = output_dir / TRANSIENT_START_WORKBOOK
    steady_sheets = read_series_workbook(steady_fp, required=True)
    start_sheets = read_series_workbook(start_fp)

    updated_start_sheets = dict(start_sheets)

    for strang in config.index:
        try:
            previous = start_sheets.get(strang, default_transient_coefficients())
            updated_start_sheets[strang] = transient_coefficients_from_wvp(
                steady_sheets[strang],
                config.loc[strang],
                previous,
            )
            logger.info("Converted %s steady WVP coefficients to standalone WVPT", strang)
        except (KeyError, AttributeError, ValueError):
            logger.exception("Skipping %s after WVPT initialization failure", strang)

    write_series_workbook(start_fp, updated_start_sheets)
    logger.info("Wrote standalone WVPT starting coefficients to %s", start_fp)


if __name__ == "__main__":
    main()
