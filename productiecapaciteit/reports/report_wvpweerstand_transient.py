"""Fit transient WVP leakage resistance coefficients and plot diagnostics."""

import logging
import tempfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from productiecapaciteit import data_dir, plot_styles_dir, results_dir
from productiecapaciteit.src.strang_analyse_fun2 import (
    get_config,
    get_false_measurements,
)
from productiecapaciteit.src.weerstand_pandasaccessors import (
    WellResistanceAccessor,  # noqa: F401
    WvpTransientResistanceAccessor,  # noqa: F401
)
from productiecapaciteit.src.wvp_transient_funs import (
    build_multiwell_geometry,
    steady_multiwell_resistance_from_kd,
)

CONFIG_FN = "strang_props7.csv"
RESAMPLE_FREQUENCY = "12h"
FIT_INITIAL_CONDITION = "zero"
BAD_DATA_RULES = [
    "Unrealistic flow",
    "Tijdens spuien",
    "Tijdens proppen",
    "Little flow",
]

DEFAULT_WELL_RADIUS_M = 0.2
DEFAULT_STORAGE_COEFFICIENT = 0.2
DEFAULT_LEAKAGE_RESISTANCE_D = 200.0
DEFAULT_LEAKAGE_BOUNDS_D = (1.0, 100_000.0)
DEFAULT_KD_BOUNDS_M2_PER_D = (1.0, 1_000.0)
DEFAULT_FIT_MAX_NFEV = 200
MIN_TRANSIENT_OBSERVATIONS = 2
DEFAULT_KD_REF_DATUM = pd.Timestamp("2020-01-01")

RESULTS_SUBDIR = "Wvptweerstand"
TRANSIENT_START_WORKBOOK = "Wvptweerstand_startwaarden.xlsx"
TRANSIENT_WORKBOOK = "Wvptweerstand_modelcoefficienten.xlsx"
TRANSIENT_LOG = "Wvptweerstandcoefficient.log"
TRANSIENT_INITIAL_LOG = "Wvptweerstand_initial.log"
TRANSIENT_FIGURE_PREFIX = "Wvptweerstandcoefficient"
LEGACY_RESULTS_SUBDIR = "Wvpweerstand_transient"
LEGACY_TRANSIENT_START_WORKBOOK = "Wvpweerstand_transient_startwaarden.xlsx"
LEGACY_TRANSIENT_WORKBOOK = "Wvpweerstand_transient_modelcoefficienten.xlsx"
TRANSIENT_REFERENCE_KEYS = (
    "kD_ref_m2_per_d",
    "kD_ref_slope_m2_per_d_per_d",
    "kD_ref_datum",
)
TRANSIENT_TEMPERATURE_KEYS = (
    "temperature_mean_degC",
    "temperature_delta_degC",
    "temperature_ref_degC",
    "temperature_time_offset_d",
    "temperature_method",
)
TRANSIENT_PHYSICAL_KEYS = (
    "well_radius_m",
    "storage_coefficient",
    "leakage_resistance_d",
)
TRANSIENT_MODIFIED_KEY = "gewijzigd"
TRANSIENT_LEGACY_MODIFIED_KEYS = ("wvpt_gewijzigd",)
TRANSIENT_COEFFICIENT_KEYS = (
    *TRANSIENT_REFERENCE_KEYS,
    *TRANSIENT_TEMPERATURE_KEYS,
    *TRANSIENT_PHYSICAL_KEYS,
    TRANSIENT_MODIFIED_KEY,
)
MODEL_FAILURE_EXCEPTIONS = (ValueError, RuntimeError, FloatingPointError, OverflowError)


def sheet_to_series(sheet):
    """Convert a single-column coefficient sheet to a Series."""
    if isinstance(sheet, pd.Series):
        return sheet.copy()
    series = sheet.squeeze("columns")
    if not isinstance(series, pd.Series):
        msg = "Coefficient sheets must contain exactly one data column"
        raise TypeError(msg)
    return series


def read_series_workbook(path, *, required=False):
    """Read an Excel workbook with one Series-like sheet per strang."""
    path = Path(path)
    if not path.exists():
        if required:
            msg = f"Required coefficient workbook does not exist: {path}"
            raise FileNotFoundError(msg)
        return {}
    sheets = pd.read_excel(path, sheet_name=None, index_col=0)
    return {name: sheet_to_series(sheet) for name, sheet in sheets.items()}


def first_existing_path(*paths):
    """Return the first existing path, or the first candidate when none exist."""
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return Path(paths[0])


def write_series_workbook(path, sheets):
    """Write Series sheets atomically, replacing the workbook on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.",
            suffix=path.suffix,
            dir=path.parent,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            for sheet_name, series in sheets.items():
                sheet_to_series(series).to_excel(writer, sheet_name=sheet_name)
        try:
            tmp_path.replace(path)
        except PermissionError as exc:
            msg = f"Could not replace {path}. Close the workbook in Excel and retry."
            raise PermissionError(msg) from exc
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def default_transient_coefficients(
    kd_ref_m2_per_d=100.0,
    kd_ref_slope_m2_per_d_per_d=0.0,
    kd_ref_datum=DEFAULT_KD_REF_DATUM,
    temperature_mean_degc=12.0,
    temperature_delta_degc=0.0,
    temperature_ref_degc=12.0,
    temperature_time_offset_d=0.0,
    temperature_method="Niet",
    well_radius_m=DEFAULT_WELL_RADIUS_M,
    storage_coefficient=DEFAULT_STORAGE_COEFFICIENT,
    leakage_resistance_d=DEFAULT_LEAKAGE_RESISTANCE_D,
):
    """Return default transient physical coefficients and fit metadata."""
    data = {
        "kD_ref_m2_per_d": float(kd_ref_m2_per_d),
        "kD_ref_slope_m2_per_d_per_d": float(kd_ref_slope_m2_per_d_per_d),
        "kD_ref_datum": pd.Timestamp(kd_ref_datum),
        "temperature_mean_degC": float(temperature_mean_degc),
        "temperature_delta_degC": float(temperature_delta_degc),
        "temperature_ref_degC": float(temperature_ref_degc),
        "temperature_time_offset_d": float(temperature_time_offset_d),
        "temperature_method": str(temperature_method),
        "well_radius_m": float(well_radius_m),
        "storage_coefficient": float(storage_coefficient),
        "leakage_resistance_d": float(leakage_resistance_d),
        TRANSIENT_MODIFIED_KEY: pd.Timestamp.now(),
    }
    return pd.Series(data)


def normalize_transient_coefficients(coefficients):
    """Return transient coefficients using the accessor-ready metadata names."""
    series = sheet_to_series(coefficients)
    for legacy_key in TRANSIENT_LEGACY_MODIFIED_KEYS:
        if legacy_key in series.index and TRANSIENT_MODIFIED_KEY not in series.index:
            series.loc[TRANSIENT_MODIFIED_KEY] = series.loc[legacy_key]
    defaults = default_transient_coefficients()
    if TRANSIENT_MODIFIED_KEY not in series.index:
        series.loc[TRANSIENT_MODIFIED_KEY] = defaults.loc[TRANSIENT_MODIFIED_KEY]
    return series


def transient_coefficients_from_sheet(coefficients):
    """Extract standalone transient coefficients from a coefficient sheet."""
    series = normalize_transient_coefficients(coefficients)
    missing = [key for key in TRANSIENT_COEFFICIENT_KEYS if key not in series.index]
    if missing:
        msg = f"Missing transient WVP coefficient(s): {', '.join(missing)}"
        raise AttributeError(msg)
    return series.loc[[key for key in TRANSIENT_COEFFICIENT_KEYS if key in series.index]]


def leakage_bounds_from_coefficients(transient_coefficients):
    """Read persisted leakage bounds from coefficients, falling back to defaults."""
    series = sheet_to_series(transient_coefficients)
    return (
        float(
            series.get(
                "wvpt_leakage_lower_bound_d",
                series.get("leakage_lower_bound_d", DEFAULT_LEAKAGE_BOUNDS_D[0]),
            )
        ),
        float(
            series.get(
                "wvpt_leakage_upper_bound_d",
                series.get("leakage_upper_bound_d", DEFAULT_LEAKAGE_BOUNDS_D[1]),
            )
        ),
    )


def with_leakage_resistance(df_a_wvpt, leakage_resistance_d):
    """Return a transient coefficient Series with an updated leakage resistance."""
    out = sheet_to_series(df_a_wvpt)
    out["leakage_resistance_d"] = float(leakage_resistance_d)
    return out


def reconstruct_aquifer_drawdown(df, ci, df_a_filter):
    """Add observed aquifer drawdown to a measurement DataFrame."""
    out = df.copy()
    q_per_well = out.Q / ci.nput
    p_omstorting_reconstructed = out.gws0 - df_a_filter.wel.dp_model(out.index, q_per_well)
    out["p_omstorting"] = out.gws1.where(out.gws1.notna(), p_omstorting_reconstructed)
    out["drawdown_aquifer"] = out.pandpeil - out.p_omstorting
    return out


def resample_transient_observations(df, frequency=RESAMPLE_FREQUENCY):
    """Return right-labeled interval means usable as step forcing for Hantush."""
    dfm = df.resample(frequency, label="right", closed="right").mean()
    mask = np.isfinite(dfm.Q) & np.isfinite(dfm.drawdown_aquifer) & (dfm.drawdown_aquifer > 0.0)
    dfm = dfm.loc[mask].copy()
    if dfm.empty:
        msg = "No positive finite aquifer drawdown observations remain"
        raise ValueError(msg)
    if dfm.index.size < MIN_TRANSIENT_OBSERVATIONS:
        msg = "At least two transient observations are required"
        raise ValueError(msg)
    return dfm


def load_observations(
    strang,
    ci,
    df_a_filter,
    frequency=RESAMPLE_FREQUENCY,
    bad_data_rules=None,
):
    """Load, filter, reconstruct and resample observations for one strang."""
    df_fp = data_dir / "Merged" / f"{strang}.feather"
    df = pd.read_feather(df_fp)
    df["Datum"] = pd.to_datetime(df["Datum"])
    df.set_index("Datum", inplace=True)

    rules = BAD_DATA_RULES if bad_data_rules is None else bad_data_rules
    untrusted_measurements = get_false_measurements(
        df,
        ci,
        extend_hours=10,
        include_rules=rules,
    )
    df.loc[untrusted_measurements, :] = np.nan
    df = reconstruct_aquifer_drawdown(df, ci, df_a_filter)
    return resample_transient_observations(df, frequency=frequency)


def transient_drawdown_for_leakage(
    leakage_resistance_d,
    dfm,
    df_a_wvpt,
    ci,
    target_well_index=None,
    initial_condition=FIT_INITIAL_CONDITION,
):
    """Calculate transient drawdown for one leakage resistance value."""
    df_a_trial = with_leakage_resistance(df_a_wvpt, leakage_resistance_d)
    return (
        -df_a_trial.wvpt.dp_model(
            dfm.index,
            dfm.Q,
            ci.nput,
            ci.dx_tussenputten,
            ci.r_mirrorwel,
            target_well_index=target_well_index,
            initial_condition=initial_condition,
        )
    ).rename("wvpt_drawdown")


def fit_leakage_resistance(  # noqa: C901
    dfm,
    df_a_wvpt,
    ci,
    leakage_bounds_d=DEFAULT_LEAKAGE_BOUNDS_D,
    target_well_index=None,
    initial_condition=FIT_INITIAL_CONDITION,
):
    """Fit leakage resistance in log space using residual innovations."""
    lower, upper = np.asarray(leakage_bounds_d, dtype=float)
    if not np.isfinite([lower, upper]).all() or lower <= 0.0 or upper <= lower:
        msg = f"Expected 0 < lower < upper, got {leakage_bounds_d}"
        raise ValueError(msg)

    x0 = float(df_a_wvpt["leakage_resistance_d"])
    if not lower <= x0 <= upper:
        msg = f"Initial leakage_resistance_d={x0:g} outside bounds {leakage_bounds_d}"
        raise ValueError(msg)

    observed = dfm.drawdown_aquifer.to_numpy(dtype=float)
    valid_observed = np.isfinite(observed)
    valid_innovation = valid_observed[1:] & valid_observed[:-1]
    if valid_innovation.sum() == 0:
        msg = "drawdown_aquifer must contain at least two consecutive finite observations"
        raise ValueError(msg)
    observed_scale = np.nanmax(np.abs(observed[valid_observed]))
    penalty = max(float(observed_scale), 1.0) * 1.0e6
    model_cache = {}
    best_fit = {"cost": np.inf, "leakage_resistance_d": None, "modeled": None}
    last_model_error = {"message": ""}

    def evaluate(leakage_resistance_d):
        if leakage_resistance_d in model_cache:
            return model_cache[leakage_resistance_d]
        try:
            modeled = transient_drawdown_for_leakage(
                leakage_resistance_d,
                dfm,
                df_a_wvpt,
                ci,
                target_well_index=target_well_index,
                initial_condition=initial_condition,
            )
        except MODEL_FAILURE_EXCEPTIONS as exc:
            last_model_error["message"] = str(exc)
            model_cache[leakage_resistance_d] = (None, exc)
            return model_cache[leakage_resistance_d]
        model_cache[leakage_resistance_d] = (modeled, None)
        return model_cache[leakage_resistance_d]

    def residual_innovations(modeled_values):
        modeled_values = np.asarray(modeled_values, dtype=float)
        valid_model = np.isfinite(modeled_values)
        valid_pair = valid_innovation & valid_model[1:] & valid_model[:-1]
        innovations = np.full(valid_innovation.sum(), penalty, dtype=float)
        residuals = modeled_values - observed
        candidate_innovations = np.diff(residuals)
        innovations[valid_pair[valid_innovation]] = candidate_innovations[valid_pair]
        return innovations

    def residual_values(leakage_resistance_d):
        modeled, error = evaluate(leakage_resistance_d)
        if error is not None:
            return np.full(valid_innovation.sum(), penalty, dtype=float)

        innovations = residual_innovations(modeled.to_numpy(dtype=float))
        cost = float(np.dot(innovations, innovations))
        if cost < best_fit["cost"]:
            best_fit.update({
                "cost": cost,
                "leakage_resistance_d": leakage_resistance_d,
                "modeled": modeled,
            })
        return innovations

    def feasible_start():
        modeled, error = evaluate(x0)
        if error is None and np.isfinite(modeled.to_numpy(dtype=float)).any():
            return x0
        candidates = np.unique(np.r_[x0, np.geomspace(lower, upper, num=31)])
        for candidate in candidates:
            modeled, error = evaluate(float(candidate))
            if error is None and np.isfinite(modeled.to_numpy(dtype=float)).any():
                return float(candidate)
        msg = (
            "No feasible leakage_resistance_d candidate produced a valid "
            f"transient model inside bounds {leakage_bounds_d}"
        )
        if last_model_error["message"]:
            msg = f"{msg}; last error: {last_model_error['message']}"
        raise RuntimeError(msg)

    def residual(log_leakage_resistance):
        leakage_resistance_d = float(np.exp(log_leakage_resistance[0]))
        return residual_values(leakage_resistance_d)

    x0 = feasible_start()

    result = least_squares(
        residual,
        x0=[np.log(x0)],
        bounds=([np.log(lower)], [np.log(upper)]),
        loss="arctan",
        f_scale=0.5,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=DEFAULT_FIT_MAX_NFEV,
    )
    if not result.success:
        msg = f"Fitting leakage_resistance_d failed: {result.message}"
        raise RuntimeError(msg)

    leakage_resistance_d = float(np.exp(result.x[0]))
    modeled, error = evaluate(leakage_resistance_d)
    if error is not None:
        if best_fit["modeled"] is None:
            msg = "Optimizer ended at an infeasible leakage_resistance_d and no feasible candidate was evaluated"
            raise RuntimeError(msg) from error
        leakage_resistance_d = float(best_fit["leakage_resistance_d"])
        modeled = best_fit["modeled"]

    residuals = modeled.to_numpy(dtype=float) - observed
    transient_coefficients = transient_coefficients_from_sheet(with_leakage_resistance(df_a_wvpt, leakage_resistance_d))
    transient_coefficients[TRANSIENT_MODIFIED_KEY] = pd.Timestamp.now()

    return {
        "coefficients": transient_coefficients,
        "modeled": modeled,
        "residuals": pd.Series(
            data=residuals,
            index=dfm.index,
            name="wvpt_residual",
        ),
        "residual_innovations": pd.Series(
            data=np.diff(residuals),
            index=dfm.index[1:],
            name="wvpt_residual_innovation",
        ),
        "optimizer_result": result,
    }


def plot_fit(
    strang,
    dfm,
    df_a_wvpt,
    modeled_drawdown,
    output_dir,
    ci,
    target_well_index=None,
):
    """Plot measured, steady and transient WVP drawdown diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kd = df_a_wvpt.wvpt.kD_model(dfm.index)
    multiwell, multiwell_counts = build_multiwell_geometry(
        ci.dx_tussenputten,
        ci.r_mirrorwel,
        ci.nput,
        target_well_index=target_well_index,
        distance_scale=1.0 / df_a_wvpt.wvpt.well_radius_m,
        include_self=True,
        self_distance=1.0,
    )
    steady_resistance = steady_multiwell_resistance_from_kd(
        kd.to_numpy(dtype=float),
        multiwell,
        ci.nput,
        df_a_wvpt.wvpt.leakage_resistance_d,
        df_a_wvpt.wvpt.well_radius_m,
    )
    steady_drawdown = pd.Series(
        steady_resistance * dfm.Q.to_numpy(dtype=float),
        index=dfm.index,
        name="wvpt_steady_drawdown",
    )
    observed = dfm.drawdown_aquifer
    residuals = modeled_drawdown - observed

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    ax0.plot(observed.index, observed, c="C0", label="Gemeten", lw=0.8)
    ax0.plot(
        steady_drawdown.index,
        steady_drawdown,
        c="C1",
        label="Steady WVP model",
        lw=0.8,
    )
    ax0.plot(
        modeled_drawdown.index,
        modeled_drawdown,
        c="C2",
        label="Transient WVP model",
        lw=0.8,
    )
    ax0.legend(loc=(0, 1), ncol=3)
    ax0.set_ylabel("Drukverlies wvp bij gemeten Q (m)")

    ax1.axhline(0.0, c="black", lw=0.8)
    ax1.plot(residuals.index, residuals, c="C3", lw=0.8)
    ax1.set_ylabel("Model - gemeten (m)")

    ax2.plot(dfm.index, dfm.Q, c="C4", lw=0.8)
    ax2.set_ylabel("Q totaal (m3/h)")

    ax3.plot(kd.index, kd, c="C5", lw=0.8)
    ax3.set_ylabel("kD(t) (m2/d)")
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax3.xaxis.get_major_locator()))

    lower, upper = leakage_bounds_from_coefficients(df_a_wvpt)
    fig.suptitle(
        f"{strang}: nobs={dfm.index.size}, nput={multiwell_counts['nput']}, "
        f"leakage={df_a_wvpt['leakage_resistance_d']:.3g} d "
        f"(bounds {lower:.3g}-{upper:.3g}), "
        f"mirror terms={multiwell_counts['self_mirrorwell_terms'] + multiwell_counts['neighbor_mirrorwell_terms']}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig_path = output_dir / f"{TRANSIENT_FIGURE_PREFIX} - {strang}.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path


def configure_logging(output_dir):
    """Configure report logging after the output directory exists."""
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
        output_dir / TRANSIENT_LOG,
        mode="w",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main(strangen=None):
    """Fit and persist transient WVP coefficients for selected strangen."""
    output_dir = results_dir / RESULTS_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)

    plt.style.use(plot_styles_dir / "unhcrpyplotstyle.mplstyle")
    plt.style.use(plot_styles_dir / "types" / "line.mplstyle")

    config = get_config(CONFIG_FN)
    if strangen is not None:
        if isinstance(strangen, str):
            strangen = [strangen]
        config = config.loc[list(strangen)]

    filter_fp = results_dir / "Filterweerstand" / "Filterweerstand_modelcoefficienten.xlsx"
    transient_start_fp = output_dir / TRANSIENT_START_WORKBOOK
    transient_fp = output_dir / TRANSIENT_WORKBOOK

    if not filter_fp.exists():
        msg = f"Required filter coefficient workbook does not exist: {filter_fp}"
        raise FileNotFoundError(msg)
    filter_sheets = pd.read_excel(filter_fp, sheet_name=None)
    transient_source_fp = transient_fp if transient_fp.exists() else transient_start_fp
    legacy_output_dir = results_dir / LEGACY_RESULTS_SUBDIR
    transient_source_fp = first_existing_path(
        transient_source_fp,
        legacy_output_dir / LEGACY_TRANSIENT_WORKBOOK,
        legacy_output_dir / LEGACY_TRANSIENT_START_WORKBOOK,
    )
    transient_sheets = read_series_workbook(transient_source_fp, required=True)
    updated_transient_sheets = {
        sheet_name: transient_coefficients_from_sheet(sheet) for sheet_name, sheet in transient_sheets.items()
    }

    for strang, ci in config.iterrows():
        logger.info("Strang: %s", strang)
        try:
            transient_coefficients = transient_coefficients_from_sheet(transient_sheets[strang])
            df_a_wvpt = transient_coefficients

            dfm = load_observations(strang, ci, filter_sheets[strang])
            fit_result = fit_leakage_resistance(
                dfm,
                df_a_wvpt,
                ci,
                leakage_bounds_d=leakage_bounds_from_coefficients(transient_sheets[strang]),
            )
            updated_transient_sheets[strang] = transient_coefficients_from_sheet(fit_result["coefficients"])
            write_series_workbook(transient_fp, updated_transient_sheets)
            logger.info("Persisted accessor-ready WVP transient coefficients to %s", transient_fp)

            optimizer_result = fit_result.get("optimizer_result")
            if optimizer_result is not None and np.any(optimizer_result.active_mask):
                logger.warning(
                    "%s fitted leakage_resistance_d hit a bound: %.3g d",
                    strang,
                    fit_result["coefficients"]["leakage_resistance_d"],
                )

            fig_path = plot_fit(
                strang,
                dfm,
                updated_transient_sheets[strang],
                fit_result["modeled"],
                output_dir,
                ci,
            )
            logger.info("Saved result to %s", fig_path)
        except (KeyError, FileNotFoundError, ValueError, RuntimeError):
            logger.exception("Skipping %s after transient WVP fit failure", strang)


if __name__ == "__main__":
    main()
