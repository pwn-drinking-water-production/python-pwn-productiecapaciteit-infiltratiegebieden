"""Calibrate the transient WVP leakage model against measured aquifer drawdown.

Two coefficients are fitted per strang:

* ``kD_ref_m2_per_d`` -- transmissivity at the reference temperature, and
* ``leakage_resistance_d`` -- the temperature-independent leakage resistance ``c``.

The storage coefficient ``S`` and the well radius ``r`` are held fixed (they are
poorly identified by drawdown alone) at :data:`DEFAULT_STORAGE_COEFFICIENT` and
:data:`DEFAULT_WELL_RADIUS_M`.

Excel data flow (everything lives in ``results/Wvptweerstand/``)
---------------------------------------------------------------
* Seed (input):  ``Wvptweerstand_modelcoefficienten.xlsx`` when it already exists.
* Result (output): ``Wvptweerstand_modelcoefficienten.xlsx``.

So the very first run seeds each strang from module defaults
(:func:`default_transient_coefficients`); every later run continues from the
previously calibrated workbook. The measurement and filter inputs come from
``data/Merged/<strang>.feather`` and
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

import itertools
import logging
import tempfile
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import shapely
import shapely.ops
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares
from scipy.spatial.distance import cdist

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
    CANAL_SIDES,
    build_multiwell_geometry,
    crosssection_image_offsets,
    infer_lower_timestep,
    objective,
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

# Series-resistance (clogging) head-loss term. The lumped infiltration + borehole-wall head
# loss is a FIXED ~0.5 m at the reference high flow at the datum, and grows multiplicatively
# with cumulative sanitized throughput:
#     dp_series(t) = (dp_ref / Q_ref) * Q(t) * (1 + g * V(t))
# where V(t) is the SIGNED cumulative volume from the datum (negative before, positive after),
# so kD_ref stays the physical aquifer transmissivity (the baseline is explicit, not absorbed).
# dp_ref (0.5 m), Q_ref (the 95th-percentile flow) and the datum are FIXED/known inputs; only
# the growth rate g is fitted. g is bounded so that (1 + g*V) >= 0 over the record and, absent
# pre-datum data, so the growth factor (1 + g*V) cannot exceed SERIES_MAX_GROWTH_FACTOR (the total
# loss also carries the temperature viscosity factor).
# dp_ref is a FIXED assumption (not fitted); if the true infiltration+borehole baseline differs,
# kD_ref absorbs the difference. flow_ref defaults to 0.0 -- a "not yet calibrated" sentinel that
# makes series_head_loss return 0 (the q_ref<=0 guard) until main sets it to the 95th-pct flow, so
# a defaulted/legacy sheet applied without recalibration never adds a spurious baseline.
DEFAULT_SERIES_DP_REF_M = 0.5
DEFAULT_SERIES_FLOW_REF_M3_PER_H = 0.0
DEFAULT_SERIES_DATUM = pd.Timestamp("2015-01-01")
DEFAULT_SERIES_GROWTH_PER_M3 = 0.0
SERIES_FLOW_REF_PERCENTILE = 95.0
SERIES_MAX_GROWTH_FACTOR = 50.0

RESULTS_SUBDIR = "Wvptweerstand"
TRANSIENT_WORKBOOK = "Wvptweerstand_modelcoefficienten.xlsx"
TRANSIENT_LOG = "Wvptweerstandcoefficient.log"
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
# Series-resistance term. These are report-level coefficients: the ``.wvpt`` accessor (the pure
# aquifer model) ignores them, so they only need to round-trip through the workbook.
TRANSIENT_SERIES_KEYS = (
    "series_dp_ref_m",
    "series_flow_ref_m3_per_h",
    "series_datum",
    "series_growth_per_m3",
)
TRANSIENT_MODIFIED_KEY = "gewijzigd"
TRANSIENT_COEFFICIENT_KEYS = (
    *TRANSIENT_REFERENCE_KEYS,
    *TRANSIENT_TEMPERATURE_KEYS,
    *TRANSIENT_PHYSICAL_KEYS,
    *TRANSIENT_SERIES_KEYS,
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
    series_dp_ref_m=DEFAULT_SERIES_DP_REF_M,
    series_flow_ref_m3_per_h=DEFAULT_SERIES_FLOW_REF_M3_PER_H,
    series_datum=DEFAULT_SERIES_DATUM,
    series_growth_per_m3=DEFAULT_SERIES_GROWTH_PER_M3,
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
        "series_dp_ref_m": float(series_dp_ref_m),
        "series_flow_ref_m3_per_h": float(series_flow_ref_m3_per_h),
        "series_datum": pd.Timestamp(series_datum),
        "series_growth_per_m3": float(series_growth_per_m3),
        TRANSIENT_MODIFIED_KEY: pd.Timestamp.now(),
    }
    return pd.Series(data)


def normalize_transient_coefficients(coefficients):
    """Ensure the coefficient Series carries a ``gewijzigd`` timestamp and the series-term keys.

    Workbooks calibrated before the series term existed lack the ``series_*`` keys; inject their
    defaults so old workbooks load and round-trip with the new keys added. The injected
    ``series_flow_ref_m3_per_h`` default is 0, so an uncalibrated sheet adds no series head loss
    (``main`` sets it to the 95th-pct flow before fitting; note growth 0 removes only the GROWTH,
    the 0.5 m baseline remains once a real reference flow is set). Shared chokepoint that
    :func:`transient_coefficients_from_sheet` (and thus every seeding path) calls.
    """
    series = sheet_to_series(coefficients)
    if TRANSIENT_MODIFIED_KEY not in series.index:
        series.loc[TRANSIENT_MODIFIED_KEY] = pd.Timestamp.now()
    series_defaults = {
        "series_dp_ref_m": DEFAULT_SERIES_DP_REF_M,
        "series_flow_ref_m3_per_h": DEFAULT_SERIES_FLOW_REF_M3_PER_H,
        "series_datum": DEFAULT_SERIES_DATUM,
        "series_growth_per_m3": DEFAULT_SERIES_GROWTH_PER_M3,
    }
    for key, default in series_defaults.items():
        if key not in series.index:
            series.loc[key] = default
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


def with_transient_parameters(coefficients, kd_ref_m2_per_d, leakage_resistance_d, series_growth_per_m3):
    """Return a coefficient Series with the three fitted parameters updated."""
    out = sheet_to_series(coefficients)
    out["kD_ref_m2_per_d"] = float(kd_ref_m2_per_d)
    out["leakage_resistance_d"] = float(leakage_resistance_d)
    out["series_growth_per_m3"] = float(series_growth_per_m3)
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


def cumulative_extracted_volume_m3(flow_m3h, datum):
    """SIGNED cumulative sanitized extracted volume [m3] relative to ``datum``.

    ``flow_m3h`` is the total strang flow (m3/h) whose untrusted rows have already been set to
    NaN by the ``get_false_measurements`` mask in :func:`load_observations`. Interior gaps are
    time-interpolated so the integral has no holes; leading/trailing unknown flow is treated as
    zero, and negative flow (non-physical for an extraction strang) is clipped to zero so the
    raw cumulative is monotonic. The integral is rebased to zero at ``datum`` but NOT clipped, so
    it is negative before the datum and positive after -- the clogging head loss was smaller than
    its datum baseline in the past and grows afterwards.
    """
    q = flow_m3h.astype(float)
    if not isinstance(q.index, pd.DatetimeIndex):
        raise TypeError("flow_m3h must be indexed by a DatetimeIndex")
    if not q.index.is_monotonic_increasing:
        raise ValueError("flow_m3h index must be sorted ascending to integrate cumulative volume")
    q = q.interpolate(method="time", limit_area="inside").fillna(0.0).clip(lower=0.0)
    hours = (q.index - q.index[0]).total_seconds().to_numpy() / 3600.0
    cumulative = cumulative_trapezoid(q.to_numpy(dtype=float), x=hours, initial=0.0)
    datum_hours = (pd.Timestamp(datum) - q.index[0]).total_seconds() / 3600.0
    cumulative = cumulative - np.interp(datum_hours, hours, cumulative)  # signed, zero at datum
    return pd.Series(cumulative, index=q.index, name="cumulative_volume_m3")


def series_head_loss(coefficients, dfm):
    """Lumped infiltration + borehole-wall head loss [m], positive meters.

    ``dp_series(t) = viscratio(T) * (dp_ref / Q_ref) * Q(t) * (1 + g * V(t))`` -- the fixed 0.5 m
    baseline at the reference high flow AND reference temperature, scaled by the actual flow, grown
    by the fitted rate ``g`` over the signed cumulative throughput ``V(t)``, and scaled by the
    viscosity ratio for the modeled temperature. Returns zeros when the volume column is absent or
    the reference flow is non-positive (the series term is then unidentifiable).
    """
    dp_ref = float(coefficients["series_dp_ref_m"])
    q_ref = float(coefficients["series_flow_ref_m3_per_h"])
    growth_rate = float(coefficients["series_growth_per_m3"])
    if "cumulative_volume_m3" not in dfm or q_ref <= 0.0:
        return pd.Series(0.0, index=dfm.index, name="series_head_loss")
    volume = dfm["cumulative_volume_m3"].to_numpy(dtype=float)
    flow = dfm["Q"].to_numpy(dtype=float)
    growth = np.maximum(1.0 + growth_rate * volume, 0.0)  # positivity safety net
    # A viscous series resistance scales with dynamic viscosity mu(T): multiply by the SAME
    # viscosity ratio the aquifer kD uses (model_viscratio -- the report runs dp_model with
    # temp_wvp=None, so both use the temperature model). dp_ref is defined at the reference
    # temperature; the growth (1 + g*V) is a temperature-free geometric clogging term. Under
    # temperature_method="Niet" viscratio == 1, so this is a bit-for-bit no-op there.
    viscratio = coefficients.wvpt.model_viscratio(dfm.index).to_numpy(dtype=float)
    return pd.Series(dp_ref / q_ref * flow * growth * viscratio, index=dfm.index, name="series_head_loss")


def load_observations(
    strang,
    ci,
    df_a_filter,
    frequency=RESAMPLE_FREQUENCY,
    bad_data_rules=None,
    series_datum=DEFAULT_SERIES_DATUM,
):
    """Load, filter, reconstruct and resample observations for one strang.

    Adds a ``cumulative_volume_m3`` column (signed sanitized cumulative throughput from
    ``series_datum``) that drives the series-resistance growth. It is built at native resolution
    from the get_false_measurements-masked flow -- BEFORE the resample drops rows -- then
    interpolated onto the (dropped-row) model index so throughput during dropped intervals counts.
    """
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
    volume_native = cumulative_extracted_volume_m3(df["Q"], datum=series_datum)
    dfm = resample_transient_observations(df, frequency=frequency)
    dfm["cumulative_volume_m3"] = np.interp(
        dfm.index.astype("int64").to_numpy(),
        df.index.astype("int64").to_numpy(),
        volume_native.to_numpy(dtype=float),
    )
    return dfm


# --------------------------------------------------------------------------- #
# Forward model and calibration
# --------------------------------------------------------------------------- #
def transient_drawdown_for_coefficients(
    kd_ref_m2_per_d,
    leakage_resistance_d,
    dfm,
    df_a_wvpt,
    ci,
    series_growth_per_m3=0.0,
    target_well_index=None,
    initial_condition=FIT_INITIAL_CONDITION,
    aquifer_cache=None,
):
    """Modeled aquifer drawdown = leaky-aquifer Hantush + series head loss, as positive meters.

    The Hantush part depends only on ``(kd_ref_m2_per_d, leakage_resistance_d)``, not on the
    series growth. Pass the same ``aquifer_cache`` dict for calls that share ``dfm``, ``df_a_wvpt``,
    ``ci``, ``target_well_index`` and ``initial_condition`` (one fit) to reuse it across growth values.
    """
    trial = with_transient_parameters(df_a_wvpt, kd_ref_m2_per_d, leakage_resistance_d, series_growth_per_m3)
    key = (kd_ref_m2_per_d, leakage_resistance_d)
    if aquifer_cache is not None and key in aquifer_cache:
        aquifer = aquifer_cache[key]
    else:
        aquifer = -trial.wvpt.dp_model(
            dfm.index,
            dfm.Q,
            ci.nput,
            ci.dx_tussenputten,
            ci.r_mirrorwel,
            target_well_index=target_well_index,
            initial_condition=initial_condition,
            # Observations are 12 h interval means labeled at the right edge
            # (resample label="right"); apply each mean on the interval it covers.
            flow_label="right",
        )
        if aquifer_cache is not None:
            aquifer_cache[key] = aquifer
    return (aquifer + series_head_loss(trial, dfm)).rename("wvpt_drawdown")


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
    """Fit ``kD_ref_m2_per_d``, ``leakage_resistance_d`` and the series growth ``g``.

    The residual vector is (model - measured) drawdown at every finite
    observation, i.e. the measured drawdown *levels* are matched directly. The
    drawdown magnitude pins ``kD_ref`` and its dynamics pin the leakage ``c``;
    differencing the residuals would leave the two jointly unidentified. A robust
    ``loss``/``f_scale`` down-weights outliers and the zero-IC warm-up transient.

    ``kD_ref`` and ``leakage`` are fitted in log space; the series growth ``g``
    (``series_growth_per_m3``) is fitted in linear space. Because the series head loss is LINEAR
    in ``g`` with a FIXED baseline, the joint fit is well-conditioned (no multimodality from the
    clogging), unlike a kD-modifying term. ``g`` is bounded ``[0, g_upper]`` so that
    ``(1 + g*V) >= 0`` over the record and, without pre-datum data, the growth factor ``(1 + g*V)``
    cannot exceed ``SERIES_MAX_GROWTH_FACTOR`` (the total loss also carries the viscosity factor).
    ``x_scale="jac"`` handles the tiny scale of ``g``.
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

    # Growth bounds (g >= 0, resistance only grows). Both are applied via the min(): positivity,
    # (1 + g*V) >= 0 at the earliest/most-negative (pre-datum) point; and a growth cap,
    # (1 + g*V) <= SERIES_MAX_GROWTH_FACTOR at the record end. The cap is the binding constraint
    # only when there is no pre-datum data (v_min >= 0); otherwise the positivity bound is tighter.
    if "cumulative_volume_m3" in dfm:
        volume = dfm["cumulative_volume_m3"].to_numpy(dtype=float)
        v_min, v_max = float(np.nanmin(volume)), float(np.nanmax(volume))
    else:
        v_min = v_max = 0.0
    g_upper_pos = (1.0 / abs(v_min)) if v_min < 0.0 else np.inf
    g_upper_cap = ((SERIES_MAX_GROWTH_FACTOR - 1.0) / v_max) if v_max > 0.0 else np.inf
    g_upper = min(g_upper_pos, g_upper_cap)
    if not np.isfinite(g_upper):
        g_upper = np.finfo(float).tiny  # no throughput -> growth unidentifiable, pin g ~ 0
    g0 = float(np.clip(float(df_a_wvpt["series_growth_per_m3"]), 0.0, g_upper))

    observed = dfm.drawdown_aquifer.to_numpy(dtype=float)
    valid = np.isfinite(observed)
    n_resid = int(valid.sum())
    if n_resid < MIN_TRANSIENT_OBSERVATIONS:
        msg = "drawdown_aquifer must contain at least two finite observations"
        raise ValueError(msg)
    observed_scale = np.nanmax(np.abs(observed[valid]))
    penalty = max(float(observed_scale), 1.0) * 1.0e6
    model_cache = {}
    aquifer_cache = {}  # Hantush part per (kd_ref, leakage); the growth-only Jacobian column reuses it
    best_fit = {"cost": np.inf, "params": None, "modeled": None}
    last_model_error = {"message": ""}

    def rank_cost(residuals):
        # Rank best_fit by the same robust loss least_squares minimizes (not raw SSE), so the
        # feasible fallback below selects the robust optimum among feasible points. Monotonic
        # surrogates of scipy's robust cost are sufficient for ranking.
        if loss == "arctan":
            return float(np.sum(np.arctan((residuals / f_scale) ** 2)))
        if loss == "soft_l1":
            return float(np.sum(np.sqrt(1.0 + (residuals / f_scale) ** 2) - 1.0))
        if loss == "cauchy":
            return float(np.sum(np.log1p((residuals / f_scale) ** 2)))
        if loss == "huber":
            z = (residuals / f_scale) ** 2
            return float(np.sum(np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)))
        return float(np.dot(residuals, residuals))

    def evaluate(kd_ref, leakage, growth):
        key = (kd_ref, leakage, growth)
        if key in model_cache:
            return model_cache[key]
        try:
            modeled = transient_drawdown_for_coefficients(
                kd_ref,
                leakage,
                dfm,
                df_a_wvpt,
                ci,
                series_growth_per_m3=growth,
                target_well_index=target_well_index,
                initial_condition=initial_condition,
                aquifer_cache=aquifer_cache,
            )
        except MODEL_FAILURE_EXCEPTIONS as exc:
            last_model_error["message"] = str(exc)
            model_cache[key] = (None, exc)
            return model_cache[key]
        model_cache[key] = (modeled, None)
        return model_cache[key]

    def residual_values(kd_ref, leakage, growth):
        modeled, error = evaluate(kd_ref, leakage, growth)
        if error is not None:
            return np.full(n_resid, penalty, dtype=float)
        modeled_values = modeled.to_numpy(dtype=float)
        residuals = np.full(n_resid, penalty, dtype=float)
        good = np.isfinite(modeled_values[valid])
        residuals[good] = (modeled_values[valid] - observed[valid])[good]
        cost = rank_cost(residuals)
        if cost < best_fit["cost"]:
            best_fit.update({"cost": cost, "params": (kd_ref, leakage, growth), "modeled": modeled})
        return residuals

    def feasible_start():
        modeled, error = evaluate(kd0, leak0, g0)
        if error is None and np.isfinite(modeled.to_numpy(dtype=float)).any():
            return kd0, leak0
        kd_candidates = np.unique(np.r_[kd0, np.geomspace(kd_lower, kd_upper, num=7)])
        leak_candidates = np.unique(np.r_[leak0, np.geomspace(leak_lower, leak_upper, num=7)])
        for kd_candidate in kd_candidates:
            for leak_candidate in leak_candidates:
                modeled, error = evaluate(float(kd_candidate), float(leak_candidate), g0)
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

    def residual(params):
        kd_ref = float(np.exp(params[0]))
        leakage = float(np.exp(params[1]))
        growth = float(params[2])
        return residual_values(kd_ref, leakage, growth)

    kd_start, leak_start = feasible_start()

    result = least_squares(
        residual,
        x0=[np.log(kd_start), np.log(leak_start), g0],
        bounds=(
            [np.log(kd_lower), np.log(leak_lower), 0.0],
            [np.log(kd_upper), np.log(leak_upper), g_upper],
        ),
        # Auto-scale from the Jacobian: the linear-space growth is orders of magnitude smaller
        # than the O(1) log parameters, so a fixed x_scale would starve or overshoot it.
        x_scale="jac",
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
    growth = float(result.x[2])
    used_fallback = False
    modeled, error = evaluate(kd_ref, leakage, growth)
    if error is not None:
        if best_fit["modeled"] is None:
            msg = "Optimizer ended at an infeasible parameter pair and no feasible candidate was evaluated"
            raise RuntimeError(msg) from error
        used_fallback = True
        kd_ref, leakage, growth = best_fit["params"]
        modeled = best_fit["modeled"]

    residuals = modeled.to_numpy(dtype=float) - observed
    coefficients = transient_coefficients_from_sheet(with_transient_parameters(df_a_wvpt, kd_ref, leakage, growth))
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
        "used_fallback": used_fallback,
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
    _multiwell, multiwell_counts = build_multiwell_geometry(
        ci.dx_tussenputten,
        ci.r_mirrorwel,
        ci.nput,
        target_well_index=target_well_index,
        distance_scale=1.0 / df_a_wvpt.wvpt.well_radius_m,
        include_self=True,
        self_distance=1.0,
    )
    steady_drawdown = (
        -df_a_wvpt.wvpt.dp_steady(
            dfm.index,
            dfm.Q,
            ci.nput,
            ci.dx_tussenputten,
            ci.r_mirrorwel,
            target_well_index=target_well_index,
        )
    ).rename("wvpt_steady_drawdown")
    observed = dfm.drawdown_aquifer
    residuals = modeled_drawdown - observed
    innovations = residuals.diff()
    series = series_head_loss(df_a_wvpt, dfm)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    ax0.plot(observed.index, observed, c="C0", label="Gemeten", lw=0.8)
    ax0.plot(steady_drawdown.index, steady_drawdown, c="C1", label="Steady WVP model", lw=0.8)
    ax0.plot(modeled_drawdown.index, modeled_drawdown, c="C2", label="Transient WVP model", lw=0.8)
    ax0.plot(series.index, series, c="C6", label="Serieweerstand (infil.+boorgat)", lw=0.8)
    ax0.legend(loc=(0, 1), ncol=4)
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
        f"leakage={df_a_wvpt['leakage_resistance_d']:.4g} d, "
        f"serie {df_a_wvpt['series_dp_ref_m']:.3g} m @ {df_a_wvpt['series_flow_ref_m3_per_h']:.3g} m3/h, "
        f"groei={df_a_wvpt['series_growth_per_m3']:.3g}/m3 "
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
# 2D head map
# --------------------------------------------------------------------------- #
# Transient drawdown around a strang in map coordinates: every pumping well plus a line-sink along
# each shore of the open water that the wells see (facing_shores), in one leaky aquifer that also
# continues under and behind the water. All sources share the calibrated transient response of one
# well: a well pumping the time shape phi_p(t) gives the drawdown table distance_table(phi_p) at a set
# of radii. The wells pump q(t) per well (shape 0). A shore piece infiltrates
# q'(xi, t) = -sum_n sum_p a_np P_n(xi) phi_p(t) per metre (Legendre polynomials P_n along the piece,
# shapes: q, first-order lags of q and copies modulated by kD and by the bed resistance), realised by
# point sinks on a line SINK_OFFSET into the water. The coefficients a_np follow from one least-squares
# fit of the bed-resistance (Robin) condition s = R(t) q' at control points on every shore over all
# times (solve_transient_linesinks); R = 0 is a fixed head. A field for any set of times is one matrix
# product of the tables with a radial kernel precomputed per set of points (radial_kernel).


def row_segments(well_number, xy, *, max_step_factor=3.0):
    """Order wells along the row by well number and split the row at large gaps.

    Parameters
    ----------
    well_number : array-like
        Well number of each well (the mpcode suffix), shape ``(n,)``.
    xy : array-like
        Well coordinates, shape ``(n, 2)``.
    max_step_factor : float, default 3.0
        A step between consecutive wells longer than this factor times the median step starts a
        new segment.

    Returns
    -------
    list of ndarray
        Per segment, the indices into ``xy`` in row order.
    """
    order = np.argsort(np.asarray(well_number), kind="stable")
    steps = np.hypot(*np.diff(np.asarray(xy, dtype=float)[order], axis=0).T)
    breaks = np.flatnonzero(steps > max_step_factor * np.median(steps)) + 1
    return np.split(order, breaks)


def well_row_normals(xy):
    """Return the unit normals of one row segment, pointing to the left of the row direction.

    The tangent at a well is the line from the well behind to the well in front (one-sided at
    the row ends); the normal is that tangent rotated by +90 degrees.

    Parameters
    ----------
    xy : array-like
        Well coordinates of one segment in row order, shape ``(n, 2)`` with ``n >= 2``.

    Returns
    -------
    ndarray
        Unit normals, shape ``(n, 2)``.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] < 2:
        msg = "A row segment needs at least two wells to define a direction"
        raise ValueError(msg)
    tangent = np.gradient(xy, axis=0)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return normal / np.linalg.norm(normal, axis=1, keepdims=True)


def prepare_water(polygons, *, close_m=8.0, simplify_m=1.0, open_m=0.5):
    """Merge open-water polygons into water bodies.

    A closing (buffer out and back in by ``close_m``) bridges the gaps that culverts and bridges
    leave between the polygons of one canal, so a canal is one body without extra tips. The union
    is then simplified, and an opening (buffer in and back out by ``open_m``) removes slivers
    narrower than ``2 open_m``, which have no room for a sink line ``open_m`` into the water.

    Parameters
    ----------
    polygons : array-like of shapely.Polygon
        Open-water polygons.
    close_m : float, default 8.0
        Closing distance in meters; gaps up to twice this width are closed.
    simplify_m : float, default 1.0
        Simplification tolerance in meters.
    open_m : float, default 0.5
        Opening distance in meters, the sink offset.

    Returns
    -------
    shapely.Geometry
        (Multi)polygon of the water bodies.
    """
    merged = shapely.buffer(shapely.buffer(shapely.union_all(polygons), close_m), -close_m)
    return shapely.buffer(shapely.buffer(shapely.simplify(merged, simplify_m), -open_m), open_m)


def facing_shores(xy, water, search_m, *, edge_m, min_length_m=20.0, eps_m=0.05):
    """Shore pieces of the water that the wells see.

    Every ring of every water body is split into edges of at most ``edge_m``. An edge is kept when
    the sight line from at least one well within ``search_m`` to the edge midpoint (stopped ``eps_m``
    short of it) crosses no water, the edge's own body included, so far banks and shores behind
    other water drop out. Runs of kept edges are merged per body (also across the start of a ring)
    and pieces shorter than ``min_length_m`` are dropped. Rings are oriented with the water on the
    left of each piece.

    Parameters
    ----------
    xy : array-like
        Well coordinates, shape ``(n, 2)``.
    water : shapely.Geometry
        Water bodies from :func:`prepare_water`.
    search_m : float
        Largest sight-line length in meters.
    edge_m : float
        Longest edge in meters.
    min_length_m : float, default 20.0
        Shortest shore piece in meters.
    eps_m : float, default 0.05
        Distance in meters by which a sight line stops short of the edge midpoint.

    Returns
    -------
    shores : ndarray of shapely.LineString
        Shore pieces, water on the left.
    bodies : ndarray of int
        Index of the water body (part of ``water``) of each piece.
    """
    xy = np.asarray(xy, dtype=float)
    parts = shapely.get_parts(shapely.orient_polygons(water))
    rings = shapely.segmentize(shapely.get_rings(parts), edge_m)
    ring_body = np.repeat(np.arange(parts.size), shapely.get_num_interior_rings(parts) + 1)
    coords, ring = shapely.get_coordinates(rings, return_index=True)
    same_ring = ring[1:] == ring[:-1]
    start, end, body = coords[:-1][same_ring], coords[1:][same_ring], ring_body[ring[:-1][same_ring]]
    middle = 0.5 * (start + end)

    distance = cdist(middle, xy)
    edge, well = np.nonzero(distance < search_m)
    toward_well = (xy[well] - middle[edge]) / distance[edge, well][:, None]
    sight_lines = shapely.linestrings(np.stack([xy[well], middle[edge] + eps_m * toward_well], axis=1))
    blocked = np.zeros(edge.size, dtype=bool)
    blocked[shapely.STRtree(parts).query(sight_lines, predicate="intersects")[0]] = True
    keep = np.zeros(middle.shape[0], dtype=bool)
    keep[edge[~blocked]] = True

    kept_bodies, piece_index = np.unique(body[keep], return_inverse=True)
    edges = shapely.linestrings(np.stack([start[keep], end[keep]], axis=1))
    merged = shapely.line_merge(shapely.multilinestrings(edges, indices=piece_index), directed=True)
    shores, merged_index = shapely.get_parts(merged, return_index=True)
    long_enough = shapely.length(shores) >= min_length_m
    return shores[long_enough], kept_bodies[merged_index[long_enough]]


def split_shores(shores, wells_xy, *, scale=1.0, min_m=20.0, max_m=160.0):
    """Cut shore pieces into sub-pieces no longer than ``scale`` times their distance to the wells.

    The head along a shore varies on the scale of its distance to the nearest well, so a
    low-order Legendre basis per sub-piece resolves it where one global basis per piece cannot
    (wiggly banks close to the wells). A sub-piece starting at distance ``d`` from the nearest well
    is ``clip(scale * d, min_m, max_m)`` long; a last remainder shorter than ``min_m / 2`` is added
    to the sub-piece before it.

    Parameters
    ----------
    shores : array-like of shapely.LineString
        Shore pieces (:func:`facing_shores`).
    wells_xy : array-like
        Well coordinates, shape ``(n, 2)``.
    scale : float, default 1.0
        Sub-piece length relative to the distance to the nearest well.
    min_m, max_m : float, default 20.0 and 160.0
        Shortest and longest sub-piece in meters.

    Returns
    -------
    pieces : ndarray of shapely.LineString
        Sub-pieces in order along each shore, with its direction.
    parent : ndarray of int
        Index of the shore piece of every sub-piece.
    """
    wells = shapely.multipoints(np.asarray(wells_xy, dtype=float))
    pieces, parent = [], []
    for i, line in enumerate(shores):
        cuts = [0.0]
        while cuts[-1] < line.length:
            distance = shapely.distance(line.interpolate(cuts[-1]), wells)
            cuts.append(min(cuts[-1] + np.clip(scale * distance, min_m, max_m), line.length))
        if len(cuts) > 2 and cuts[-1] - cuts[-2] < 0.5 * min_m:
            del cuts[-2]
        pieces += [shapely.ops.substring(line, start, end) for start, end in itertools.pairwise(cuts)]
        parent += [i] * (len(cuts) - 1)
    return np.array(pieces), np.array(parent)


def lagged_histories(index, values, time_constants_d, initial_value):
    """First-order lags of a piecewise-constant history.

    ``values[i]`` applies on ``(index[i - 1], index[i]]`` (``flow_label="right"``), so each lag is
    exact across gaps: ``y_i = y_{i-1} e^{-dt/T} + values_i (1 - e^{-dt/T})``. Before ``index[0]``
    the history is steady at ``initial_value``.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Model times.
    values : array-like
        History values, shape ``(len(index),)``.
    time_constants_d : array-like
        Lag time constants in days.
    initial_value : float
        Steady value before ``index[0]``.

    Returns
    -------
    ndarray
        Lagged histories, shape ``(len(index), len(time_constants_d))``.
    """
    values = np.asarray(values, dtype=float)
    dt = np.diff(pd.DatetimeIndex(index)) / pd.Timedelta("1D")
    decay = np.exp(-dt[:, None] / np.asarray(time_constants_d, dtype=float)[None, :])
    lagged = np.empty((values.size, decay.shape[1]))
    lagged[0] = initial_value
    for i in range(1, values.size):
        lagged[i] = decay[i - 1] * lagged[i - 1] + (1.0 - decay[i - 1]) * values[i]
    return lagged


def time_shapes(index, q_per_well_m3d, initial_q_m3d, time_constants_d, modulations=()):
    """Time shapes of the line-sink inflow and their steady pre-period values.

    The base shapes are ``q`` and its first-order lags (:func:`lagged_histories`). Every modulation
    ``f`` (the kD model, the bed-resistance factor) adds the copies ``base * (f / mean(f) - 1)``;
    a constant modulation (constant kD, fixed head, constant temperature) adds nothing.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Model times.
    q_per_well_m3d : array-like
        Flow per well in m3/d (``flow_label="right"``).
    initial_q_m3d : float
        Steady flow per well before ``index[0]``.
    time_constants_d : array-like
        Lag time constants in days.
    modulations : sequence of array-like, default ()
        Series on ``index``, positive or all zero.

    Returns
    -------
    shapes : ndarray
        Shape ``(len(index), K)``; column 0 is ``q``.
    initial : ndarray
        Steady pre-period value of every shape, shape ``(K,)``.
    """
    q = np.asarray(q_per_well_m3d, dtype=float)
    base = np.column_stack([q, lagged_histories(index, q, time_constants_d, initial_q_m3d)])
    base_initial = np.full(base.shape[1], float(initial_q_m3d))
    shapes, initial = [base], [base_initial]
    for modulation in modulations:
        modulation = np.asarray(modulation, dtype=float)
        if np.ptp(modulation) == 0.0:
            continue
        relative = modulation / modulation.mean() - 1.0
        shapes.append(base * relative[:, None])
        initial.append(base_initial * relative[0])
    return np.hstack(shapes), np.concatenate(initial)


def bed_resistance(coefficients, index, t_bodem_degc, r_bed_12c_d_per_m):
    """Canal-bed resistance over time.

    ``R(t) = R_12 * visc_ratio(T_bodem, temp_ref=12)``: ``R_12`` is the resistance at 12 degC,
    whatever the reference temperature of the WVPT sheet. Gaps in ``T_bodem`` are interpolated in
    time; missing values at the ends are filled with the nearest value, with a warning. An empty
    (NaN) ``R_12`` means a fixed head: ``R = 0``.

    Parameters
    ----------
    coefficients : pandas.Series
        Calibrated transient WVP coefficients (``.wvpt`` accessor).
    index : pandas.DatetimeIndex
        Model times.
    t_bodem_degc : array-like
        Infiltration-water temperature in degC on ``index``.
    r_bed_12c_d_per_m : float
        Bed resistance at 12 degC in d/m (head drop per infiltration per metre of shore, m2/d).

    Returns
    -------
    ndarray
        Bed resistance in d/m, shape ``(len(index),)``.
    """
    index = pd.DatetimeIndex(index)
    if not np.isfinite(r_bed_12c_d_per_m):
        return np.zeros(index.size)
    temperature = pd.Series(np.asarray(t_bodem_degc, dtype=float), index=index)
    filled = temperature.interpolate(method="time", limit_area="inside")
    if filled.isna().all():
        msg = "T_bodem has no values"
        raise ValueError(msg)
    if filled.isna().any():
        warnings.warn(
            f"T_bodem is missing at {filled.isna().sum()} leading or trailing times; filled with the nearest value",
            stacklevel=2,
        )
        filled = filled.ffill().bfill()
    return r_bed_12c_d_per_m * coefficients.wvpt.visc_ratio(filled.to_numpy(), temp_ref=12.0)


def strip_bed_resistance(drop_m, flow_per_m_m2d, kd_m2_per_d, leakage_resistance_d, canal_offsets_m):
    """Bed resistance that gives a head drop over the bed of the nearest canal of a 1D strip.

    The row is a line source ``flow_per_m`` (m2/d) at ``y = 0`` in a leaky aquifer; each canal is a
    line-sink at its offset whose infiltration ``sigma`` obeys ``s = R sigma`` (``s`` drawdown). The
    drawdown of a line source is ``Q' lambda / (2 kD) exp(-|y| / lambda)``. For one canal, or two at
    the same distance ``b``, ``R = D (1 + e^{-2b/lambda}) / (Q' e^{-b/lambda} - 2 kD D / lambda)``
    (drop the ``e^{-2b/lambda}`` term for one canal); in general ``R`` is the root of the drop on the
    nearest canal. The drop cannot exceed ``Q' lambda e^{-b/lambda} / (2 kD)`` (``R`` to infinity).

    Parameters
    ----------
    drop_m : float
        Head drop over the bed of the nearest canal in meters.
    flow_per_m_m2d : float
        Row flow per metre of row in m2/d.
    kd_m2_per_d : float
        Transmissivity in m2/d.
    leakage_resistance_d : float
        Leakage resistance in days.
    canal_offsets_m : array-like
        Signed canal offsets from the row in meters.

    Returns
    -------
    float
        Bed resistance in d/m, NaN when the drop is not reachable.
    """
    leakage_factor = np.sqrt(kd_m2_per_d * leakage_resistance_d)
    offsets = np.asarray(canal_offsets_m, dtype=float)
    unit_drawdown = leakage_factor / (2.0 * kd_m2_per_d)
    coupling = unit_drawdown * np.exp(-np.abs(offsets[:, None] - offsets[None, :]) / leakage_factor)
    forcing = flow_per_m_m2d * unit_drawdown * np.exp(-np.abs(offsets) / leakage_factor)
    nearest = np.argmin(np.abs(offsets))
    if drop_m >= forcing[nearest]:
        return np.nan

    def excess_drop(resistance):
        infiltration = np.linalg.solve(coupling + resistance * np.eye(offsets.size), forcing)
        return resistance * infiltration[nearest] - drop_m

    upper = 1.0
    while excess_drop(upper) < 0.0:
        upper *= 2.0
    return brentq(excess_drop, 0.0, upper, xtol=1e-15, rtol=4.0 * np.finfo(float).eps)


def initial_bed_resistances(config, sheets, drop_m=0.5):
    """Bed resistance per strang that gives a head drop over the nearest canal bed at Q95.

    Uses :func:`strip_bed_resistance` with the calibrated ``kD_ref`` and leakage resistance, the
    canals of ``r_mirrorwel`` and the row as a line source ``Q' = Q95 * 24 / (nput * dx)`` with Q95
    the sheet's ``series_flow_ref_m3_per_h``. The ``R_bed_12C_d_per_m`` row of strang_props7.csv was
    produced with it (``drop_m = 0.5``) from the recalibrated WVPT workbook.

    Parameters
    ----------
    config : pandas.DataFrame
        Strang configuration (:func:`get_config`), one row per strang.
    sheets : dict
        Transient WVP coefficient sheet per strang (:func:`read_series_workbook`).
    drop_m : float, default 0.5
        Head drop over the bed of the nearest canal in meters.

    Returns
    -------
    pandas.Series
        Bed resistance at 12 degC in d/m per strang; NaN without a sheet or when the drop is not
        reachable.
    """
    resistances = {}
    for strang, ci in config.iterrows():
        if strang not in sheets:
            resistances[strang] = np.nan
            continue
        coefficients = transient_coefficients_from_sheet(sheets[strang])
        flow_per_m = coefficients["series_flow_ref_m3_per_h"] * 24.0 / (ci.nput * ci.dx_tussenputten)
        offsets = [offset for _, offset in crosssection_image_offsets(ci.r_mirrorwel)]
        resistances[strang] = strip_bed_resistance(
            drop_m, flow_per_m, coefficients["kD_ref_m2_per_d"], coefficients["leakage_resistance_d"], offsets
        )
    return pd.Series(resistances, name="R_bed_12C_d_per_m", dtype=float)


def distance_table(coefficients, index, q_per_well_m3d, radii, *, initial_condition, integration_method="kd_grid"):
    """Transient drawdown of one well at a set of distances.

    Each distance ``r`` is solved as a well of radius ``r`` (``alpha`` scaled by
    ``r / well_radius``), so every column gets the exact near-window integral of the kd_grid
    method. Treating ``r > well_radius`` as a point source instead misses the kernel peak under
    time-varying kD (decimeters near the well).

    Parameters
    ----------
    coefficients : pandas.Series
        Calibrated transient WVP coefficients (``.wvpt`` accessor).
    index : pandas.DatetimeIndex
        Model times.
    q_per_well_m3d : array-like
        Flow per well in m3/d; ``q[i]`` applies on ``(index[i - 1], index[i]]`` (``flow_label="right"``).
    radii : array-like
        Distances from the well in meters, each at least the well radius.
    initial_condition : str or float
        ``"zero"``, ``"steady"`` or a steady pre-period flow per well in m3/d.
    integration_method : str, default "kd_grid"
        Integration method of :func:`hantush_variable_kd`.

    Returns
    -------
    ndarray
        Drawdown in positive meters, shape ``(len(index), len(radii))``.
    """
    wvpt = coefficients.wvpt
    index = pd.DatetimeIndex(index)
    pextra = {
        "index": index,
        "Q_obs": np.asarray(q_per_well_m3d, dtype=float),
        "kD": wvpt.kD_model(index).to_numpy(dtype=float),
        "dt_lower": infer_lower_timestep(index),
        "multiwell": [(1.0, 1.0)],
        "multiwell_contains_r_self": True,
        "initial_condition": initial_condition,
        "integration_method": integration_method,
        "flow_label": "right",
    }
    alpha_per_m = wvpt.alpha / wvpt.well_radius_m
    return np.column_stack([
        objective([alpha_per_m * radius, wvpt.beta], return_result=True, **pextra)
        for radius in np.asarray(radii, dtype=float)
    ])


def radial_kernel(points, sources, strengths, radii, well_radius_m, *, max_elements=20_000_000):
    """Precomputed spatial kernel of distance tables.

    A table row is interpolated in ``ln r`` by a not-a-knot cubic spline, which is linear in the
    row values: ``s(r) = sum_j card_j(ln r) table_j`` with the cardinal splines ``card_j``. The
    drawdown at point ``x`` of sources ``m`` with strengths ``S[m, k]`` on tables ``k`` is then
    ``sum_j sum_k table[j, k] G[x, j, k]`` with ``G[x, j, k] = sum_m card_j(ln d_xm) S[m, k]``, so
    the field for any set of times is ``tables.reshape(n_times, -1) @ G.reshape(len(points), -1).T``.

    Parameters
    ----------
    points : array-like
        Evaluation points, shape ``(n, 2)``.
    sources : array-like
        Source coordinates, shape ``(m, 2)``.
    strengths : array-like
        Strength of every source per table, shape ``(m, K)``.
    radii : array-like
        Increasing table distances in meters, starting at the well radius.
    well_radius_m : float
        Distances are clipped at the well radius.
    max_elements : int, default 20_000_000
        Points are processed in chunks of at most ``max_elements / 8`` (point, source) pairs.

    Returns
    -------
    ndarray
        Kernel ``G``, shape ``(n, len(radii), K)``.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    sources = np.asarray(sources, dtype=float).reshape(-1, 2)
    strengths = np.asarray(strengths, dtype=float).reshape(sources.shape[0], -1)
    radii = np.asarray(radii, dtype=float)
    log_radii = np.log(radii)
    # On interval i the cardinal splines are card_j(x) = sum_p coef[p, i, j] (x - log_radii[i]) ** (3 - p).
    coef = CubicSpline(log_radii, np.eye(radii.size)).c
    n_intervals = radii.size - 1
    kernel = np.empty((points.shape[0], radii.size, strengths.shape[1]))
    chunk = max(1, max_elements // (8 * sources.shape[0]))
    for start in range(0, points.shape[0], chunk):
        distance = np.maximum(cdist(sources, points[start : start + chunk]), well_radius_m)
        if distance.max() > radii[-1]:
            msg = f"Largest source distance {distance.max():.1f} m exceeds the table range {radii[-1]:.1f} m"
            raise ValueError(msg)
        n_points = distance.shape[1]
        log_distance = np.log(distance)
        interval = np.clip(np.searchsorted(log_radii, log_distance, side="right") - 1, 0, n_intervals - 1)
        offset = log_distance - log_radii[interval]
        powers = np.stack([offset**3, offset**2, offset, np.ones_like(offset)], axis=-1)
        # Sum the source strengths per (point, power, interval), then contract with the spline coefficients.
        rows = (np.arange(n_points)[:, None] * 4 + np.arange(4)) * n_intervals + interval[..., None]
        binning = scipy.sparse.csc_array(
            (powers.ravel(), rows.ravel(), np.arange(sources.shape[0] + 1) * 4 * n_points),
            shape=(n_points * 4 * n_intervals, sources.shape[0]),
        )
        binned = (binning @ strengths).reshape(n_points, 4, n_intervals, -1)
        kernel[start : start + n_points] = np.einsum("xpik,pij->xjk", binned, coef, optimize=True)
    return kernel


def legendre_orders(lengths_m, *, order_spacing_m=80.0, max_order=20):
    """Legendre order per shore piece: ``clip(round(L / order_spacing_m), 2, max_order)``."""
    return np.clip(np.round(np.asarray(lengths_m, dtype=float) / order_spacing_m).astype(int), 2, max_order)


def _legendre_basis(lines, orders, fractions_per_line):
    """Points on lines and their block Legendre basis.

    Returns the points, the line index of each point and the matrix ``P[i, col]`` with
    ``P_n(2 f_i - 1)`` in column ``offset[line_i] + n`` for ``n <= orders[line_i]``.
    """
    line = np.repeat(np.arange(len(lines)), [f.size for f in fractions_per_line])
    fraction = np.concatenate(fractions_per_line)
    points = shapely.get_coordinates(shapely.line_interpolate_point(lines[line], fraction, normalized=True))
    offsets = np.concatenate([[0], np.cumsum(orders + 1)])
    degree = np.arange(orders.max() + 1)
    valid = degree[None, :] <= orders[line][:, None]
    rows, cols = np.nonzero(valid)
    vander = np.polynomial.legendre.legvander(2.0 * fraction - 1.0, orders.max())
    basis = np.zeros((fraction.size, offsets[-1]))
    basis[rows, offsets[line][rows] + cols] = vander[valid]
    return points, line, basis


def linesink_nodes(sinks, orders, spacing_m):
    """Point sinks at about ``spacing_m`` along the sink lines and their Legendre basis.

    Parameters
    ----------
    sinks : array-like of shapely.LineString
        Sink lines.
    orders : array-like of int
        Legendre order per line.
    spacing_m : float
        Largest node spacing in meters; each line gets equal spacings ``h`` with nodes at the
        midpoints.

    Returns
    -------
    nodes : ndarray
        Node coordinates, shape ``(n, 2)``.
    line : ndarray of int
        Line index of every node.
    basis : ndarray
        ``h * P_n(xi)`` per node and basis function (pumping of a node per unit coefficient, m).
    """
    sinks = np.asarray(sinks)
    lengths = shapely.length(sinks)
    counts = np.maximum(np.ceil(lengths / spacing_m).astype(int), 1)
    fractions = [(np.arange(count) + 0.5) / count for count in counts]
    nodes, line, basis = _legendre_basis(sinks, np.asarray(orders), fractions)
    return nodes, line, basis * (lengths / counts)[line][:, None]


def solve_transient_linesinks(
    tables,
    radii,
    wells_xy,
    shores,
    shapes,
    bed_resistance_d_per_m,
    well_radius_m,
    *,
    sink_offset_m=0.5,
    order_spacing_m=80.0,
    max_order=20,
    svd_rtol=1e-12,
    max_elements=20_000_000,
):
    """Transient Legendre line-sinks along shores with a bed resistance.

    Each shore piece gets a sink line ``sink_offset_m`` into the water with point sinks every
    ``sink_offset_m / 2`` and Legendre polynomials up to :func:`legendre_orders`. The drawdown at
    ``3 (N + 1)`` cosine-spaced control points on every shore (both ends included) must satisfy
    ``s = R(t) q'`` at every time, with ``q' = -sum_n sum_p a_np P_n phi_p(t)`` the infiltration per
    metre. The least-squares problem over all times is compressed exactly: every time row is linear
    in the time functions (table rows and ``R phi``), so their SVD ``U s V^T`` gives the same normal
    equations with the pseudo-times ``s V^T`` (singular values below ``svd_rtol`` times the largest
    dropped). The rows are reduced block-wise by QR and solved with scaled columns.

    Parameters
    ----------
    tables : array-like
        :func:`distance_table` of every shape, shape ``(n_times, len(radii), K)``; shape 0 is the
        well flow ``q``.
    radii : array-like
        Table distances in meters.
    wells_xy : array-like
        Well coordinates, shape ``(n_wells, 2)``; each pumps shape 0.
    shores : array-like of shapely.LineString
        Shore pieces, water on the left (:func:`facing_shores`).
    shapes : array-like
        Time shapes ``phi_p`` in m3/d, shape ``(n_times, K)``.
    bed_resistance_d_per_m : array-like
        Bed resistance ``R(t)`` in d/m (:func:`bed_resistance`), shape ``(n_times,)``.
    well_radius_m : float
        Distances are clipped at the well radius.
    sink_offset_m : float, default 0.5
        Distance of the sink line into the water in meters.
    order_spacing_m, max_order : float, int
        Legendre order per piece, see :func:`legendre_orders`.
    svd_rtol : float or None, default 1e-12
        Relative singular-value cutoff of the time compression; None solves on all times.
    max_elements : int, default 20_000_000
        Memory bound of the kernel chunks and of the least-squares blocks.

    Returns
    -------
    dict
        ``sinks`` (sink lines), ``orders``, ``coefficients`` (``a``, shape ``(K, n_basis)``),
        ``controls`` (shape ``(n_controls, 2)``), ``inflow_per_m`` (``q'`` at the controls in m2/d,
        shape ``(n_times, n_controls)``) and ``shore_residual_m``
        (``s - R q'`` at the controls in m, same shape).
    """
    tables = np.asarray(tables, dtype=float)
    n_times, n_radii, n_shapes = tables.shape
    shapes = np.asarray(shapes, dtype=float).reshape(n_times, n_shapes)
    resistance = np.asarray(bed_resistance_d_per_m, dtype=float).reshape(n_times)
    shores = np.asarray(shores)
    orders = legendre_orders(shapely.length(shores), order_spacing_m=order_spacing_m, max_order=max_order)
    sinks = shapely.offset_curve(shores, sink_offset_m)

    nodes, _, node_basis = linesink_nodes(sinks, orders, sink_offset_m / 2.0)
    counts = 3 * (orders + 1)
    fractions = [(1.0 - np.cos(np.pi * np.arange(count) / (count - 1))) / 2.0 for count in counts]
    controls, _, control_basis = _legendre_basis(shores, orders, fractions)
    kernel_sinks = radial_kernel(controls, nodes, node_basis, radii, well_radius_m, max_elements=max_elements)
    kernel_wells = radial_kernel(
        controls, wells_xy, np.ones(len(wells_xy)), radii, well_radius_m, max_elements=max_elements
    )[..., 0]

    time_functions = np.column_stack([tables.reshape(n_times, -1), resistance[:, None] * shapes])
    if svd_rtol is not None:
        _, singular_values, vt = np.linalg.svd(time_functions, full_matrices=False)
        keep = singular_values > svd_rtol * singular_values[0]
        time_functions = singular_values[keep, None] * vt[keep]

    n_controls, n_basis = control_basis.shape
    n_columns = n_shapes * n_basis
    # At least as many rows per block as columns, else each QR costs more than the rows it adds.
    block = max(-(-(n_columns + 1) // n_controls), max_elements // (n_controls * (n_columns + 1)))
    triangle = np.empty((0, n_columns + 1))
    for start in range(0, time_functions.shape[0], block):
        rows = time_functions[start : start + block]
        table_rows = rows[:, : n_radii * n_shapes].reshape(-1, n_radii, n_shapes)
        system = np.einsum("tjk,cjn->tckn", table_rows, kernel_sinks, optimize=True)
        system += rows[:, None, n_radii * n_shapes :, None] * control_basis[None, :, None, :]
        rhs = -table_rows[:, :, 0] @ kernel_wells.T
        stacked = np.column_stack([system.reshape(-1, n_columns), rhs.reshape(-1)])
        triangle = np.linalg.qr(np.vstack([triangle, stacked]), mode="r")
    scale = np.linalg.norm(triangle[:, :-1], axis=0)
    coefficients = (np.linalg.lstsq(triangle[:, :-1] / scale, triangle[:, -1], rcond=None)[0] / scale).reshape(
        n_shapes, n_basis
    )

    control_kernel = np.einsum("cjn,kn->cjk", kernel_sinks, coefficients)
    control_kernel[:, :, 0] += kernel_wells
    drawdown = drawdown_from_kernel(tables, control_kernel)
    inflow_per_m = -shapes @ (control_basis @ coefficients.T).T
    return {
        "sinks": sinks,
        "orders": orders,
        "coefficients": coefficients,
        "controls": controls,
        "inflow_per_m": inflow_per_m,
        "shore_residual_m": drawdown - resistance[:, None] * inflow_per_m,
    }


def linesink_sources(solution, spacing_m):
    """Point sinks of a solved line-sink model for field evaluation.

    Parameters
    ----------
    solution : dict
        Result of :func:`solve_transient_linesinks`.
    spacing_m : float
        Largest node spacing in meters. Fields closer than about this distance to a sink line
        (in the water) are not resolved.

    Returns
    -------
    nodes : ndarray
        Node coordinates, shape ``(n, 2)``.
    piece : ndarray of int
        Shore piece of every node.
    strengths : ndarray
        Pumping of every node per shape (negative = infiltration), shape ``(n, K)``, in the table
        unit; the infiltration of a set of nodes is ``-shapes @ strengths[nodes].sum(axis=0)``.
    """
    nodes, piece, basis = linesink_nodes(solution["sinks"], solution["orders"], spacing_m)
    return nodes, piece, basis @ solution["coefficients"].T


def map_sources(solution, wells_xy, spacing_m):
    """All sources of a solved line-sink model for :func:`radial_kernel`.

    Parameters
    ----------
    solution : dict
        Result of :func:`solve_transient_linesinks`.
    wells_xy : array-like
        Well coordinates, shape ``(n_wells, 2)``; each pumps shape 0.
    spacing_m : float
        Largest sink-node spacing in meters, see :func:`linesink_sources`.

    Returns
    -------
    sources : ndarray
        The wells followed by the sink nodes, shape ``(n_wells + n, 2)``.
    strengths : ndarray
        Strength of every source per shape, shape ``(n_wells + n, K)``.
    """
    wells_xy = np.asarray(wells_xy, dtype=float)
    nodes, _, strengths = linesink_sources(solution, spacing_m)
    well_strengths = np.zeros((wells_xy.shape[0], strengths.shape[1]))
    well_strengths[:, 0] = 1.0
    return np.concatenate([wells_xy, nodes]), np.concatenate([well_strengths, strengths])


def infiltration_by_body_and_side(solution, shapes, bodies, wells_xy, normals, index, spacing_m):
    """Infiltration per water body and side of the well row over time.

    Every sink node counts for the water body of its shore piece and for the side of the row it
    lies on as seen from its nearest well: ``"left"`` along that well's left normal
    (:func:`well_row_normals`, left of increasing well number, as the sides of ``r_mirrorwel``),
    else ``"right"``.

    Parameters
    ----------
    solution : dict
        Result of :func:`solve_transient_linesinks`.
    shapes : array-like
        Time shapes of the solve, shape ``(len(index), K)``.
    bodies : array-like of int
        Water body of every shore piece.
    wells_xy, normals : array-like
        Well coordinates and their left normals, shape ``(n_wells, 2)``.
    index : pandas.DatetimeIndex
        Model times.
    spacing_m : float
        Sink-node spacing in meters, see :func:`linesink_sources`.

    Returns
    -------
    pandas.DataFrame
        Infiltration in the table unit (m3/d) per ``(body, side)`` column.
    """
    wells_xy = np.asarray(wells_xy, dtype=float)
    normals = np.asarray(normals, dtype=float)
    nodes, piece, strengths = linesink_sources(solution, spacing_m)
    nearest = cdist(nodes, wells_xy).argmin(axis=1)
    left = np.einsum("ij,ij->i", nodes - wells_xy[nearest], normals[nearest]) > 0.0
    groups = pd.MultiIndex.from_arrays(
        [np.asarray(bodies)[piece], np.where(left, CANAL_SIDES[0], CANAL_SIDES[1])], names=["body", "side"]
    )
    group_strengths = pd.DataFrame(strengths).groupby(groups).sum()
    return pd.DataFrame(
        -np.asarray(shapes, dtype=float) @ group_strengths.to_numpy().T, index=index, columns=group_strengths.index
    )


def drawdown_from_kernel(tables, kernel):
    """Drawdown at the kernel's points for every time of the tables.

    Parameters
    ----------
    tables : array-like
        Distance tables, shape ``(n_times, n_radii, K)``.
    kernel : array-like
        :func:`radial_kernel`, shape ``(n_points, n_radii, K)``.

    Returns
    -------
    ndarray
        Drawdown in meters, shape ``(n_times, n_points)``.
    """
    tables = np.asarray(tables, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    return tables.reshape(tables.shape[0], -1) @ kernel.reshape(kernel.shape[0], -1).T


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

    coefficient_fp = output_dir / TRANSIENT_WORKBOOK
    # Continue from the previously calibrated workbook; on the very first run
    # (no calibrated workbook yet) seed each strang from module defaults.
    source_sheets = read_series_workbook(coefficient_fp)
    if source_sheets:
        report_logger.info("Seeding transient WVP coefficients from %s", coefficient_fp)
    else:
        report_logger.info(
            "No calibrated workbook at %s; seeding each strang from module defaults",
            coefficient_fp,
        )
    report_logger.info("Writing calibrated transient WVP coefficients to %s", coefficient_fp)

    calibrated_sheets = {
        name: force_physical_constants(transient_coefficients_from_sheet(sheet))
        for name, sheet in source_sheets.items()
    }

    for strang, ci in config.iterrows():
        report_logger.info("Strang: %s", strang)
        try:
            source_sheet = source_sheets.get(strang, default_transient_coefficients())
            seed = force_physical_constants(transient_coefficients_from_sheet(source_sheet))
            dfm = load_observations(strang, ci, filter_sheets[strang], series_datum=seed["series_datum"])
            # The reference "high flow" for the 0.5 m baseline is the 95th percentile of the
            # sanitized flow, recomputed and stored each run.
            seed["series_flow_ref_m3_per_h"] = float(
                np.nanpercentile(dfm["Q"].to_numpy(dtype=float), SERIES_FLOW_REF_PERCENTILE)
            )
            fit_result = fit_transient_coefficients(dfm, seed, ci)
            calibrated_sheets[strang] = transient_coefficients_from_sheet(fit_result["coefficients"])
            write_series_workbook(coefficient_fp, calibrated_sheets)

            coeff = calibrated_sheets[strang]
            report_logger.info(
                "%s calibrated: kD_ref=%.4g m2/d, leakage=%.4g d, "
                "serie=%.3g m @ %.4g m3/h, groei=%.3g /m3 (S=%.3g, r=%.3g m)",
                strang,
                coeff["kD_ref_m2_per_d"],
                coeff["leakage_resistance_d"],
                coeff["series_dp_ref_m"],
                coeff["series_flow_ref_m3_per_h"],
                coeff["series_growth_per_m3"],
                coeff["storage_coefficient"],
                coeff["well_radius_m"],
            )
            optimizer_result = fit_result.get("optimizer_result")
            if (
                optimizer_result is not None
                and not fit_result.get("used_fallback")
                and np.any(optimizer_result.active_mask)
            ):
                report_logger.warning(
                    "%s: a fitted parameter hit its bound (active_mask=%s)",
                    strang,
                    optimizer_result.active_mask,
                )

            fig_path = plot_fit(strang, dfm, calibrated_sheets[strang], fit_result["modeled"], output_dir, ci)
            report_logger.info("Saved figure to %s", fig_path)
        except (KeyError, FileNotFoundError, ValueError, RuntimeError, OverflowError, FloatingPointError):
            report_logger.exception("Skipping %s after transient WVP fit failure", strang)


if __name__ == "__main__":
    main()
