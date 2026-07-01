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

import logging
import tempfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
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
# pre-datum data, so the loss cannot grow past SERIES_MAX_GROWTH_FACTOR x the baseline.
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

    ``dp_series(t) = (dp_ref / Q_ref) * Q(t) * (1 + g * V(t))`` -- the fixed 0.5 m baseline at the
    reference high flow, scaled by the actual flow and grown by the fitted rate ``g`` over the
    signed cumulative throughput ``V(t)``. Returns zeros when the volume column is absent or the
    reference flow is non-positive (the series term is then unidentifiable).
    """
    # Not viscosity(temperature)-scaled: a laminar series resistance physically scales with mu(T),
    # but this is consistent with the aquifer term only while temperature_method="Niet"
    # (viscratio==1). If "sin" is ever enabled, multiply this by wvpt.model_viscratio(index) too.
    dp_ref = float(coefficients["series_dp_ref_m"])
    q_ref = float(coefficients["series_flow_ref_m3_per_h"])
    growth_rate = float(coefficients["series_growth_per_m3"])
    if "cumulative_volume_m3" not in dfm or q_ref <= 0.0:
        return pd.Series(0.0, index=dfm.index, name="series_head_loss")
    volume = dfm["cumulative_volume_m3"].to_numpy(dtype=float)
    flow = dfm["Q"].to_numpy(dtype=float)
    growth = np.maximum(1.0 + growth_rate * volume, 0.0)  # positivity safety net
    return pd.Series(dp_ref / q_ref * flow * growth, index=dfm.index, name="series_head_loss")


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
):
    """Modeled aquifer drawdown = leaky-aquifer Hantush + series head loss, as positive meters."""
    trial = with_transient_parameters(
        df_a_wvpt, kd_ref_m2_per_d, leakage_resistance_d, series_growth_per_m3
    )
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
    ``(1 + g*V) >= 0`` over the record and, without pre-datum data, the loss cannot grow past
    ``SERIES_MAX_GROWTH_FACTOR`` x the baseline. ``x_scale="jac"`` handles the tiny scale of ``g``.
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
    coefficients = transient_coefficients_from_sheet(
        with_transient_parameters(df_a_wvpt, kd_ref, leakage, growth)
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
            dfm = load_observations(
                strang, ci, filter_sheets[strang], series_datum=seed["series_datum"]
            )
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
