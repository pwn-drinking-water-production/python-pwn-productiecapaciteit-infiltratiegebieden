"""Calibrate the transient WVP leakage model against measured aquifer drawdown.

Two coefficients are fitted per strang:

* ``kD_ref_m2_per_d`` -- transmissivity at the reference temperature, and
* ``leakage_resistance_d`` -- the temperature-independent leakage resistance ``c``.

The storage coefficient ``S`` and the well radius ``r`` are held fixed (they are
poorly identified by drawdown alone) at :data:`DEFAULT_STORAGE_COEFFICIENT` and
:data:`DEFAULT_WELL_RADIUS_M`.

Excel data flow (everything lives in ``results/Wvptweerstand/``)
---------------------------------------------------------------
* Seed (input):  ``Wvptweerstand_modelcoefficienten.xlsx`` when it already exists,
  otherwise ``Wvptweerstand_startwaarden.xlsx`` (created by
  ``report_wvpweerstand_transient_initial.py``).
* Result (output): ``Wvptweerstand_modelcoefficienten.xlsx``.

So the very first run starts from the starting-value workbook; every later run
continues from the previously calibrated workbook. The measurement and filter
inputs come from ``data/Merged/<strang>.feather`` and
``results/Filterweerstand/Filterweerstand_modelcoefficienten.xlsx``.

Fitting target
--------------
The fit minimises the residuals (model - measured) of the aquifer drawdown
*levels* directly: the measured drawdown magnitude is what pins ``kD_ref`` (in the
leaky-Hantush model drawdown scales like ``Q / (4 pi kD)``), and its dynamics pin
the leakage ``c``. Differencing the residuals instead (innovations) throws the
level away and leaves ``kD`` and ``c`` jointly unidentified, so it is not used.
The model runs with a zero initial condition, which leaves a short warm-up
transient at the start of the record; over the multi-year records this is a small
fraction of the data. The diagnostic plot shows the residuals and their
innovations (first differences) on the same axis.
"""

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
    WellResistanceAccessor,  # noqa: F401  (registers the ``.wel`` accessor)
    WvpTransientResistanceAccessor,  # noqa: F401  (registers the ``.wvpt`` accessor)
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

# Held fixed during calibration (not fitted).
DEFAULT_WELL_RADIUS_M = 0.3
DEFAULT_STORAGE_COEFFICIENT = 0.2

# Fitted parameters: starting values and search bounds.
DEFAULT_KD_REF_M2_PER_D = 100.0
DEFAULT_KD_BOUNDS_M2_PER_D = (1.0, 5_000.0)
DEFAULT_LEAKAGE_RESISTANCE_D = 200.0
DEFAULT_LEAKAGE_BOUNDS_D = (1.0, 100_000.0)

DEFAULT_FIT_MAX_NFEV = 200
# Robust loss for the level-matching fit (the only outlier handling beyond the
# bad-data rules). ``f_scale`` is in meters: residuals beyond it are down-weighted,
# which also tames the zero-IC warm-up transient. 0.5 m sits well above the typical
# inlier residual (~0.3 m) yet keeps the leakage from railing to the confined bound.
DEFAULT_FIT_LOSS = "arctan"
DEFAULT_FIT_F_SCALE_M = 0.5
MIN_TRANSIENT_OBSERVATIONS = 2
DEFAULT_KD_REF_DATUM = pd.Timestamp("2020-01-01")

RESULTS_SUBDIR = "Wvptweerstand"
TRANSIENT_START_WORKBOOK = "Wvptweerstand_startwaarden.xlsx"
TRANSIENT_WORKBOOK = "Wvptweerstand_modelcoefficienten.xlsx"
TRANSIENT_LOG = "Wvptweerstandcoefficient.log"
TRANSIENT_INITIAL_LOG = "Wvptweerstand_initial.log"
TRANSIENT_FIGURE_PREFIX = "Wvptweerstandcoefficient"

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
TRANSIENT_COEFFICIENT_KEYS = (
    *TRANSIENT_REFERENCE_KEYS,
    *TRANSIENT_TEMPERATURE_KEYS,
    *TRANSIENT_PHYSICAL_KEYS,
    TRANSIENT_MODIFIED_KEY,
)
MODEL_FAILURE_EXCEPTIONS = (ValueError, RuntimeError, FloatingPointError, OverflowError)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Coefficient workbook helpers
# --------------------------------------------------------------------------- #
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
    kd_ref_m2_per_d=DEFAULT_KD_REF_M2_PER_D,
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
    """Ensure the coefficient Series carries a ``gewijzigd`` timestamp."""
    series = sheet_to_series(coefficients)
    if TRANSIENT_MODIFIED_KEY not in series.index:
        series.loc[TRANSIENT_MODIFIED_KEY] = pd.Timestamp.now()
    return series


def transient_coefficients_from_sheet(coefficients):
    """Extract the accessor-ready transient coefficients from a sheet."""
    series = normalize_transient_coefficients(coefficients)
    missing = [key for key in TRANSIENT_COEFFICIENT_KEYS if key not in series.index]
    if missing:
        msg = f"Missing transient WVP coefficient(s): {', '.join(missing)}"
        raise AttributeError(msg)
    return series.loc[list(TRANSIENT_COEFFICIENT_KEYS)]


def force_physical_constants(
    coefficients,
    well_radius_m=DEFAULT_WELL_RADIUS_M,
    storage_coefficient=DEFAULT_STORAGE_COEFFICIENT,
):
    """Pin the non-fitted physical constants (``S`` and ``r``) to fixed values."""
    out = sheet_to_series(coefficients)
    out["well_radius_m"] = float(well_radius_m)
    out["storage_coefficient"] = float(storage_coefficient)
    return out


def with_transient_parameters(coefficients, kd_ref_m2_per_d, leakage_resistance_d):
    """Return a coefficient Series with updated fitted parameters."""
    out = sheet_to_series(coefficients)
    out["kD_ref_m2_per_d"] = float(kd_ref_m2_per_d)
    out["leakage_resistance_d"] = float(leakage_resistance_d)
    return out


# --------------------------------------------------------------------------- #
# Observation loading
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Forward model and calibration
# --------------------------------------------------------------------------- #
def transient_drawdown_for_coefficients(
    kd_ref_m2_per_d,
    leakage_resistance_d,
    dfm,
    df_a_wvpt,
    ci,
    target_well_index=None,
    initial_condition=FIT_INITIAL_CONDITION,
):
    """Transient aquifer drawdown for a (kD_ref, leakage) pair, as positive meters."""
    trial = with_transient_parameters(df_a_wvpt, kd_ref_m2_per_d, leakage_resistance_d)
    return (
        -trial.wvpt.dp_model(
            dfm.index,
            dfm.Q,
            ci.nput,
            ci.dx_tussenputten,
            ci.r_mirrorwel,
            target_well_index=target_well_index,
            initial_condition=initial_condition,
        )
    ).rename("wvpt_drawdown")


def fit_transient_coefficients(  # noqa: C901
    dfm,
    df_a_wvpt,
    ci,
    kd_bounds_m2_per_d=DEFAULT_KD_BOUNDS_M2_PER_D,
    leakage_bounds_d=DEFAULT_LEAKAGE_BOUNDS_D,
    target_well_index=None,
    initial_condition=FIT_INITIAL_CONDITION,
    loss=DEFAULT_FIT_LOSS,
    f_scale=DEFAULT_FIT_F_SCALE_M,
):
    """Fit ``kD_ref_m2_per_d`` and ``leakage_resistance_d`` in log space.

    The residual vector is (model - measured) drawdown at every finite
    observation, i.e. the measured drawdown *levels* are matched directly. The
    drawdown magnitude pins ``kD_ref`` and its dynamics pin the leakage ``c``;
    differencing the residuals would leave the two jointly unidentified. A robust
    ``loss``/``f_scale`` down-weights outliers and the zero-IC warm-up transient.
    """
    kd_lower, kd_upper = np.asarray(kd_bounds_m2_per_d, dtype=float)
    leak_lower, leak_upper = np.asarray(leakage_bounds_d, dtype=float)
    for name, lower, upper in (
        ("kD bounds", kd_lower, kd_upper),
        ("leakage bounds", leak_lower, leak_upper),
    ):
        if not np.isfinite([lower, upper]).all() or lower <= 0.0 or upper <= lower:
            msg = f"Expected 0 < lower < upper for {name}, got ({lower}, {upper})"
            raise ValueError(msg)

    kd0 = float(df_a_wvpt["kD_ref_m2_per_d"])
    leak0 = float(df_a_wvpt["leakage_resistance_d"])
    if not kd_lower <= kd0 <= kd_upper:
        msg = f"Initial kD_ref_m2_per_d={kd0:g} outside bounds {tuple(kd_bounds_m2_per_d)}"
        raise ValueError(msg)
    if not leak_lower <= leak0 <= leak_upper:
        msg = f"Initial leakage_resistance_d={leak0:g} outside bounds {tuple(leakage_bounds_d)}"
        raise ValueError(msg)

    observed = dfm.drawdown_aquifer.to_numpy(dtype=float)
    valid = np.isfinite(observed)
    n_resid = int(valid.sum())
    if n_resid < MIN_TRANSIENT_OBSERVATIONS:
        msg = "drawdown_aquifer must contain at least two finite observations"
        raise ValueError(msg)
    observed_scale = np.nanmax(np.abs(observed[valid]))
    penalty = max(float(observed_scale), 1.0) * 1.0e6
    model_cache = {}
    best_fit = {"cost": np.inf, "params": None, "modeled": None}
    last_model_error = {"message": ""}

    def evaluate(kd_ref, leakage):
        key = (kd_ref, leakage)
        if key in model_cache:
            return model_cache[key]
        try:
            modeled = transient_drawdown_for_coefficients(
                kd_ref,
                leakage,
                dfm,
                df_a_wvpt,
                ci,
                target_well_index=target_well_index,
                initial_condition=initial_condition,
            )
        except MODEL_FAILURE_EXCEPTIONS as exc:
            last_model_error["message"] = str(exc)
            model_cache[key] = (None, exc)
            return model_cache[key]
        model_cache[key] = (modeled, None)
        return model_cache[key]

    def residual_values(kd_ref, leakage):
        modeled, error = evaluate(kd_ref, leakage)
        if error is not None:
            return np.full(n_resid, penalty, dtype=float)
        modeled_values = modeled.to_numpy(dtype=float)
        residuals = np.full(n_resid, penalty, dtype=float)
        good = np.isfinite(modeled_values[valid])
        residuals[good] = (modeled_values[valid] - observed[valid])[good]
        cost = float(np.dot(residuals, residuals))
        if cost < best_fit["cost"]:
            best_fit.update({"cost": cost, "params": (kd_ref, leakage), "modeled": modeled})
        return residuals

    def feasible_start():
        modeled, error = evaluate(kd0, leak0)
        if error is None and np.isfinite(modeled.to_numpy(dtype=float)).any():
            return kd0, leak0
        kd_candidates = np.unique(np.r_[kd0, np.geomspace(kd_lower, kd_upper, num=7)])
        leak_candidates = np.unique(np.r_[leak0, np.geomspace(leak_lower, leak_upper, num=7)])
        for kd_candidate in kd_candidates:
            for leak_candidate in leak_candidates:
                modeled, error = evaluate(float(kd_candidate), float(leak_candidate))
                if error is None and np.isfinite(modeled.to_numpy(dtype=float)).any():
                    return float(kd_candidate), float(leak_candidate)
        msg = (
            "No feasible (kD_ref, leakage_resistance_d) candidate produced a valid "
            f"transient model inside bounds kD={tuple(kd_bounds_m2_per_d)}, "
            f"leakage={tuple(leakage_bounds_d)}"
        )
        if last_model_error["message"]:
            msg = f"{msg}; last error: {last_model_error['message']}"
        raise RuntimeError(msg)

    def residual(log_params):
        kd_ref = float(np.exp(log_params[0]))
        leakage = float(np.exp(log_params[1]))
        return residual_values(kd_ref, leakage)

    kd_start, leak_start = feasible_start()

    result = least_squares(
        residual,
        x0=[np.log(kd_start), np.log(leak_start)],
        bounds=(
            [np.log(kd_lower), np.log(leak_lower)],
            [np.log(kd_upper), np.log(leak_upper)],
        ),
        loss=loss,
        f_scale=f_scale,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=DEFAULT_FIT_MAX_NFEV,
    )
    if not result.success:
        msg = f"Fitting transient WVP coefficients failed: {result.message}"
        raise RuntimeError(msg)

    kd_ref = float(np.exp(result.x[0]))
    leakage = float(np.exp(result.x[1]))
    modeled, error = evaluate(kd_ref, leakage)
    if error is not None:
        if best_fit["modeled"] is None:
            msg = "Optimizer ended at an infeasible parameter pair and no feasible candidate was evaluated"
            raise RuntimeError(msg) from error
        kd_ref, leakage = best_fit["params"]
        modeled = best_fit["modeled"]

    residuals = modeled.to_numpy(dtype=float) - observed
    coefficients = transient_coefficients_from_sheet(
        with_transient_parameters(df_a_wvpt, kd_ref, leakage)
    )
    coefficients[TRANSIENT_MODIFIED_KEY] = pd.Timestamp.now()

    return {
        "coefficients": coefficients,
        "modeled": modeled,
        "residuals": pd.Series(data=residuals, index=dfm.index, name="wvpt_residual"),
        "residual_innovations": pd.Series(
            data=np.diff(residuals),
            index=dfm.index[1:],
            name="wvpt_residual_innovation",
        ),
        "optimizer_result": result,
    }


# --------------------------------------------------------------------------- #
# Plotting and logging
# --------------------------------------------------------------------------- #
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
    innovations = residuals.diff()

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    ax0.plot(observed.index, observed, c="C0", label="Gemeten", lw=0.8)
    ax0.plot(steady_drawdown.index, steady_drawdown, c="C1", label="Steady WVP model", lw=0.8)
    ax0.plot(modeled_drawdown.index, modeled_drawdown, c="C2", label="Transient WVP model", lw=0.8)
    ax0.legend(loc=(0, 1), ncol=3)
    ax0.set_ylabel("Drukverlies wvp bij gemeten Q (m)")

    ax1.axhline(0.0, c="black", lw=0.8)
    ax1.plot(residuals.index, residuals, c="C3", lw=0.8, label="Residu (model - gemeten)")
    ax1.plot(
        innovations.index,
        innovations,
        c="C0",
        lw=0.8,
        label="Innovatie (Δ residu, gefit)",
    )
    ax1.legend(loc="upper right", ncol=2)
    ax1.set_ylabel("Model - gemeten (m)")

    ax2.plot(dfm.index, dfm.Q, c="C4", lw=0.8)
    ax2.set_ylabel("Q totaal (m3/h)")

    ax3.plot(kd.index, kd, c="C5", lw=0.8)
    ax3.set_ylabel("kD(t) (m2/d)")
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax3.xaxis.get_major_locator()))

    fig.suptitle(
        f"{strang}: nobs={dfm.index.size}, nput={multiwell_counts['nput']}, "
        f"kD_ref={df_a_wvpt['kD_ref_m2_per_d']:.4g} m2/d, "
        f"leakage={df_a_wvpt['leakage_resistance_d']:.4g} d "
        f"(S={df_a_wvpt['storage_coefficient']:.3g}, r={df_a_wvpt['well_radius_m']:.3g} m), "
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
    report_logger = logging.getLogger(__name__)
    report_logger.setLevel(logging.INFO)
    for handler in report_logger.handlers[:]:
        report_logger.removeHandler(handler)
        handler.close()
    report_logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(output_dir / TRANSIENT_LOG, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    report_logger.addHandler(file_handler)
    report_logger.addHandler(stream_handler)
    return report_logger


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(strangen=None):
    """Fit and persist transient WVP coefficients for the selected strangen."""
    output_dir = results_dir / RESULTS_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    report_logger = configure_logging(output_dir)

    plt.style.use(plot_styles_dir / "unhcrpyplotstyle.mplstyle")
    plt.style.use(plot_styles_dir / "types" / "line.mplstyle")

    config = get_config(CONFIG_FN)
    if strangen is not None:
        if isinstance(strangen, str):
            strangen = [strangen]
        config = config.loc[list(strangen)]

    filter_fp = results_dir / "Filterweerstand" / "Filterweerstand_modelcoefficienten.xlsx"
    if not filter_fp.exists():
        msg = f"Required filter coefficient workbook does not exist: {filter_fp}"
        raise FileNotFoundError(msg)
    filter_sheets = pd.read_excel(filter_fp, sheet_name=None)

    start_fp = output_dir / TRANSIENT_START_WORKBOOK
    coefficient_fp = output_dir / TRANSIENT_WORKBOOK
    # Continue from the previously calibrated workbook, or start fresh from the
    # starting-value workbook on the very first run.
    source_fp = coefficient_fp if coefficient_fp.exists() else start_fp
    if not source_fp.exists():
        msg = (
            f"No transient WVP coefficients found ({coefficient_fp} or {start_fp}). "
            "Run report_wvpweerstand_transient_initial.py first."
        )
        raise FileNotFoundError(msg)
    report_logger.info("Seeding transient WVP coefficients from %s", source_fp)
    report_logger.info("Writing calibrated transient WVP coefficients to %s", coefficient_fp)

    source_sheets = read_series_workbook(source_fp, required=True)
    calibrated_sheets = {
        name: force_physical_constants(transient_coefficients_from_sheet(sheet))
        for name, sheet in source_sheets.items()
    }

    for strang, ci in config.iterrows():
        report_logger.info("Strang: %s", strang)
        try:
            seed = force_physical_constants(transient_coefficients_from_sheet(source_sheets[strang]))
            dfm = load_observations(strang, ci, filter_sheets[strang])
            fit_result = fit_transient_coefficients(dfm, seed, ci)
            calibrated_sheets[strang] = transient_coefficients_from_sheet(fit_result["coefficients"])
            write_series_workbook(coefficient_fp, calibrated_sheets)

            coeff = calibrated_sheets[strang]
            report_logger.info(
                "%s calibrated: kD_ref=%.4g m2/d, leakage=%.4g d (S=%.3g, r=%.3g m)",
                strang,
                coeff["kD_ref_m2_per_d"],
                coeff["leakage_resistance_d"],
                coeff["storage_coefficient"],
                coeff["well_radius_m"],
            )
            optimizer_result = fit_result.get("optimizer_result")
            if optimizer_result is not None and np.any(optimizer_result.active_mask):
                report_logger.warning(
                    "%s: a fitted parameter hit its bound (active_mask=%s)",
                    strang,
                    optimizer_result.active_mask,
                )

            fig_path = plot_fit(strang, dfm, calibrated_sheets[strang], fit_result["modeled"], output_dir, ci)
            report_logger.info("Saved figure to %s", fig_path)
        except (KeyError, FileNotFoundError, ValueError, RuntimeError):
            report_logger.exception("Skipping %s after transient WVP fit failure", strang)


if __name__ == "__main__":
    main()
