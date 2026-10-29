"""Pure evaluation and native-resolution calibration of measured well-filter losses."""

import json
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import lsq_linear

from productiecapaciteit.src.weerstand_pandasaccessors import WvpResistanceAccessor

_MAX_WATER_TEMPERATURE_DEGC = 40.0


class ExtractedWaterTemperatureError(ValueError):
    """Missing, misaligned or invalid extracted-water temperature."""


class LegacyFilterCoefficientWarning(UserWarning):
    """An empirical filter coefficient has no reference-temperature interpretation."""


def extracted_water_viscosity_ratio(
    temperature: npt.ArrayLike | pd.Series | None,
    index: pd.DatetimeIndex,
    *,
    reference_temperature: float,
) -> np.ndarray:
    """Validate extracted-water temperature and use the existing viscosity helper.

    Parameters
    ----------
    temperature : array-like, Series or None
        Extracted-water temperature in degC. A scalar is an explicitly prescribed
        constant; Series are aligned by timestamp. No interpolation is performed.
    index : DatetimeIndex
        Required observation or model timestamps.
    reference_temperature : float
        Reference temperature in degC.

    Returns
    -------
    ndarray
        Dynamic viscosity relative to its value at the reference temperature.

    Raises
    ------
    ExtractedWaterTemperatureError
        Temperature is missing, unaligned, nonfinite or outside [0, 40] degC.
    """
    if not np.isfinite(reference_temperature) or not 0 <= reference_temperature <= _MAX_WATER_TEMPERATURE_DEGC:
        message = "Invalid extracted-water reference temperature; require [0, 40] degC"
        raise ExtractedWaterTemperatureError(message)
    try:
        if isinstance(temperature, pd.Series):
            temperature = temperature.reindex(index)
        values = np.asarray(temperature, dtype=float)
        if values.ndim == 0:
            values = np.full(len(index), values.item())
    except (ValueError, TypeError) as exc:
        message = f"Invalid or missing extracted-water temperature: {exc}"
        raise ExtractedWaterTemperatureError(message) from exc
    if (
        temperature is None
        or values.shape != (len(index),)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > _MAX_WATER_TEMPERATURE_DEGC))
    ):
        message = "Invalid or missing extracted-water temperature; require finite, aligned values in [0, 40] degC"
        raise ExtractedWaterTemperatureError(message)
    return np.asarray(WvpResistanceAccessor.visc_ratio(values, temp_ref=reference_temperature), dtype=float)


def _coefficient_arrays(coefficients):
    required = {"datum", "offset", "slope"}
    if not required.issubset(coefficients.columns) or coefficients.empty:
        message = "Filter coefficients require nonempty datum, offset and slope columns"
        raise ValueError(message)
    dates = pd.DatetimeIndex(coefficients.datum)
    if dates.hasnans or not dates.is_monotonic_increasing or not dates.is_unique:
        message = "Filter coefficient dates must be valid and strictly increasing"
        raise ValueError(message)
    values = coefficients[["offset", "slope"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values > 0):
        message = "Filter offsets and slopes must be finite and nonpositive"
        raise ValueError(message)
    return dates, values[:, 0], values[:, 1]


def evaluate_filter_coefficient(
    coefficients: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    side: str = "right",
) -> pd.Series:
    """Evaluate the stored signed hourly coefficient, including cleaning limits.

    Parameters
    ----------
    coefficients : DataFrame
        Legacy-compatible ``datum``, ``offset`` (h/m2), ``slope`` (h/m2/day).
        Values retain their stored empirical or reference-temperature semantics.
    index : DatetimeIndex
        Evaluation timestamps. Before the first datum, hold its offset constant.
    side : {"right", "left"}, default "right"
        At a cleaning date use its new offset or the preceding period's limit.
        For right-labelled forcing intervals, split at cleanings and use "left"
        at the interval endpoint to avoid applying cleaning retrospectively.

    Returns
    -------
    Series
        Signed coefficient in h/m2, without a viscosity correction.
    """
    if side not in {"left", "right"}:
        message = "side must be 'left' or 'right'"
        raise ValueError(message)
    dates, offsets, slopes = _coefficient_arrays(coefficients)
    index = pd.DatetimeIndex(index)
    if index.hasnans:
        message = "Filter evaluation timestamps must not contain NaT"
        raise ValueError(message)
    period = np.maximum(dates.searchsorted(index, side=side) - 1, 0)
    days = np.maximum(np.asarray((index - dates.take(period)) / pd.Timedelta(days=1)), 0.0)
    return pd.Series(offsets[period] + slopes[period] * days, index=index, name="a_filter_h_per_m2")


def _temperature_metadata(coefficients):
    if "temperature_semantics" not in coefficients:
        if "temperature_ref_degC" in coefficients and coefficients.temperature_ref_degC.notna().any():
            message = "Filter reference metadata requires explicit temperature_semantics"
            raise ValueError(message)
        return "legacy_empirical", None
    semantics = coefficients.temperature_semantics
    if semantics.isna().any() or semantics.nunique() != 1:
        message = "Filter temperature semantics metadata must be present and identical in every row"
        raise ValueError(message)
    meaning = semantics.iloc[0]
    if meaning == "legacy_empirical":
        return meaning, None
    if meaning != "reference_normalized":
        message = f"Unknown filter temperature semantics metadata: {meaning}"
        raise ValueError(message)
    if "temperature_ref_degC" not in coefficients:
        message = "Normalized filter metadata requires temperature_ref_degC reference"
        raise ValueError(message)
    reference = coefficients.temperature_ref_degC
    if reference.isna().any() or reference.nunique() != 1:
        message = "Normalized filter reference metadata must be finite and identical in every row"
        raise ValueError(message)
    try:
        value = float(reference.iloc[0])
    except (TypeError, ValueError) as exc:
        message = "Normalized filter reference metadata must be numeric"
        raise ValueError(message) from exc
    return meaning, value


def filter_resistance(
    coefficients: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    temperature: npt.ArrayLike | pd.Series | None = None,
    side: str = "right",
) -> pd.Series:
    """Return nonnegative filter resistance for solver flow in m3/day.

    Parameters
    ----------
    coefficients : DataFrame
        Signed hourly filter coefficients and optional persisted temperature
        metadata. Absent metadata means legacy empirical coefficients.
    index : DatetimeIndex
        Model evaluation timestamps.
    temperature : array-like, Series or None, optional
        Extracted-water temperature in degC, required for normalized coefficients.
        Ignored for legacy coefficients: applying viscosity again is not valid.
    side : {"right", "left"}, default "right"
        One-sided cleaning-date convention; see ``evaluate_filter_coefficient``.

    Returns
    -------
    Series
        ``Rf_day = -a_hour / 24``, in day/m2, multiplied by the viscosity ratio
        only when explicit reference-normalized metadata is present.

    Warns
    -----
    LegacyFilterCoefficientWarning
        Legacy coefficients have no reference-temperature interpretation.
    """
    coefficient = evaluate_filter_coefficient(coefficients, index, side=side)
    semantics, reference = _temperature_metadata(coefficients)
    resistance = -coefficient / 24
    if semantics == "reference_normalized":
        resistance *= extracted_water_viscosity_ratio(
            temperature, pd.DatetimeIndex(coefficient.index), reference_temperature=reference
        )
    else:
        warnings.warn(
            "Legacy empirical filter coefficients are not reference-normalized; viscosity is not applied a second time.",
            LegacyFilterCoefficientWarning,
            stacklevel=2,
        )
    return resistance.rename("Rf_d_per_m2")


def split_filter_intervals(edges: pd.DatetimeIndex, coefficients: pd.DataFrame) -> pd.DatetimeIndex:
    """Insert cleaning dates into forcing edges without changing the forcing span.

    Parameters
    ----------
    edges : DatetimeIndex
        Strictly increasing forcing boundaries, including the initial instant.
    coefficients : DataFrame
        Filter coefficient periods; every datum after the first is a cleaning.

    Returns
    -------
    DatetimeIndex
        Sorted union with internal cleaning dates. This does not invent flow
        observations: the caller must carry the original interval flow onto
        both subintervals and evaluate their endpoint resistance on the left.
    """
    edges = pd.DatetimeIndex(edges)
    if len(edges) <= 1 or edges.hasnans or not edges.is_unique or not edges.is_monotonic_increasing:
        message = "Forcing edges must contain at least two strictly increasing valid dates"
        raise ValueError(message)
    dates, _, _ = _coefficient_arrays(coefficients)
    cleaning = dates[1:]
    return edges.union(cleaning[(cleaning > edges[0]) & (cleaning < edges[-1])]).sort_values()


def _fit_period_selection(index, fit_periods):
    periods = [(index[0], index[-1])] if fit_periods is None else fit_periods
    if not periods:
        message = "At least one fitting period is required"
        raise ValueError(message)
    selected = np.zeros(len(index), dtype=bool)
    serialized = []
    for period_start, period_end in periods:
        start, end = pd.Timestamp(period_start), pd.Timestamp(period_end)
        if pd.isna(start) or pd.isna(end) or start > end:
            message = "Fitting periods require valid start <= end"
            raise ValueError(message)
        selected |= (index >= start) & (index <= end)
        serialized.append([start.isoformat(), end.isoformat()])
    return selected, json.dumps(serialized)


def _fit_flow(observations, sensor_flow_column, total_flow_column, nput, *, allow_effective_equal_flow):
    if sensor_flow_column is not None:
        denominator_column = sensor_flow_column
        provenance = f"measured_individual:{sensor_flow_column}"
        divisor = 1
    else:
        if not allow_effective_equal_flow or not isinstance(nput, int) or isinstance(nput, bool) or nput < 1:
            message = "No individual sensor flow; declare effective Q/nput calibration and a positive integer nput"
            raise ValueError(message)
        denominator_column = total_flow_column
        provenance = f"effective_equal_flow:{total_flow_column}/nput={nput}"
        divisor = nput
    if denominator_column not in observations:
        message = f"Missing declared filter observation channel: {denominator_column}"
        raise ValueError(message)
    return observations[denominator_column].to_numpy(dtype=float) / divisor, provenance


def fit_normalized_filter(
    observations: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    temperature_column: str,
    sensor_id: str,
    head_provenance: str,
    upstream_fit_dependencies: list[dict],
    sensor_flow_column: str | None = None,
    total_flow_column: str = "Q",
    nput: int | None = None,
    allow_effective_equal_flow: bool = False,
    inside_head_column: str = "gws0",
    outside_head_column: str = "gws1",
    reference_temperature: float = 12.0,
    fit_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Fit shared clogging slope and cleaning offsets at reference temperature.

    Parameters
    ----------
    observations : DataFrame
        Native-resolution, timestamp-indexed heads in m, temperature in degC and
        signed flow in m3/h. No smoothing or averaging is performed. Nonfinite
        heads/flow and zero flow are excluded before temperature validation.
    dates : DatetimeIndex
        Initial coefficient datum followed by cleaning dates.
    temperature_column : str
        Explicitly identified extracted-water temperature channel, not canal
        or bed temperature. No inferred channel or interpolation fallback.
    sensor_id : str
        Identity of the well carrying the head sensors.
    head_provenance : {"direct", "reconstructed"}
        Origin of the outside-filter head. Reconstructed heads are not
        independent validation evidence.
    upstream_fit_dependencies : list of dict
        Explicit upstream fit identities and fitting periods, including filter
        parameters used to reconstruct heads. Pass [] only if there are none.
    sensor_flow_column : str or None, optional
        Measured flow of this individual well. If declared, its absence is an
        error and invalid samples never silently fall back to Q/nput.
    total_flow_column : str, default "Q"
        Measured total flow channel, used only in the explicit effective route.
    nput : int or None, optional
        Declared active well count for the effective Q/nput denominator.
    allow_effective_equal_flow : bool, default False
        Explicit permission to fit an effective coefficient using Q/nput.
        It does not establish that actual individual flows were equal.
    inside_head_column, outside_head_column : str, optional
        Internal and external filter head channels. The stored coefficient is
        signed: (inside head - outside head) / flow / viscosity ratio.
    reference_temperature : float, default 12
        Reference temperature in degC for both fitted slope and offsets.
    fit_periods : list of (Timestamp, Timestamp) or None, optional
        Inclusive periods to fit and persist. None means the supplied span.

    Returns
    -------
    DataFrame
        Legacy-compatible datum/offset/slope columns and repeated, serializable
        metadata columns. The bounded linear least-squares objective uses raw
        residuals, not squared residuals passed to a least-squares optimizer.

    Notes
    -----
    With an effective denominator, R_effective/R_true = nput*q_sensor/Q.
    Viscosity normalization does not remove this bias. Temperature-corrected
    calibration is opt-in and does not use a configured legacy put_a_slope.
    """
    index = pd.DatetimeIndex(observations.index)
    dates = pd.DatetimeIndex(dates)
    if index.empty or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        message = "Native observations require a nonempty, unique, increasing timestamp index"
        raise ValueError(message)
    if dates.empty or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        message = "Filter fitting dates must be nonempty and strictly increasing"
        raise ValueError(message)
    if not sensor_id or head_provenance not in {"direct", "reconstructed"}:
        message = "Declare sensor_id and direct/reconstructed head_provenance"
        raise ValueError(message)
    if not isinstance(upstream_fit_dependencies, list):
        message = "Declare upstream_fit_dependencies explicitly as a list"
        raise TypeError(message)
    dependencies = json.dumps(upstream_fit_dependencies)
    if temperature_column not in observations:
        message = f"Missing extracted-water temperature channel: {temperature_column}"
        raise ExtractedWaterTemperatureError(message)
    flow, provenance = _fit_flow(
        observations, sensor_flow_column, total_flow_column, nput, allow_effective_equal_flow=allow_effective_equal_flow
    )
    missing = {inside_head_column, outside_head_column}.difference(observations.columns)
    if missing:
        message = f"Missing declared filter observation channels: {sorted(missing)}"
        raise ValueError(message)
    inside = observations[inside_head_column].to_numpy(dtype=float)
    outside = observations[outside_head_column].to_numpy(dtype=float)
    selected, periods_json = _fit_period_selection(index, fit_periods)
    selected &= np.isfinite(flow) & (flow != 0) & np.isfinite(inside) & np.isfinite(outside)
    selected_index = index[selected]
    if selected_index.empty or selected_index[0] < dates[0]:
        message = "Filter fit needs usable observations on or after the initial coefficient datum"
        raise ValueError(message)
    viscosity = extracted_water_viscosity_ratio(
        observations.loc[selected, temperature_column], selected_index, reference_temperature=reference_temperature
    )
    a_reference = (inside[selected] - outside[selected]) / (flow[selected] * viscosity)
    period = dates.searchsorted(selected_index, side="right") - 1
    days = np.asarray((selected_index - dates.take(period)) / pd.Timedelta(days=1))
    design = np.column_stack((days, np.eye(len(dates))[period]))
    scale = np.linalg.norm(design, axis=0)
    if np.any(scale == 0) or np.linalg.matrix_rank(design / scale) != design.shape[1]:
        message = "Filter slope and cleaning offsets are not identifiable in the selected fitting periods"
        raise ValueError(message)
    result = lsq_linear(design / scale, a_reference, bounds=(-np.inf, 0), tol=1e-14)
    if not result.success or not np.isfinite(result.x).all():
        message = f"Normalized filter fit failed: {result.message}"
        raise ValueError(message)
    parameters = result.x / scale
    return pd.DataFrame({
        "datum": dates,
        "offset": parameters[1:],
        "slope": parameters[0],
        "temperature_semantics": "reference_normalized",
        "temperature_ref_degC": reference_temperature,
        "temperature_channel": temperature_column,
        "temperature_policy": "native_no_interpolation",
        "coefficient_units": "h/m2",
        "slope_units": "h/m2/day",
        "sensor_id": sensor_id,
        "head_provenance": head_provenance,
        "inside_head_channel": inside_head_column,
        "outside_head_channel": outside_head_column,
        "flow_denominator_provenance": provenance,
        "fit_periods": periods_json,
        "upstream_fit_dependencies": dependencies,
        "fit_method": "bounded_linear_least_squares_reference_space",
        "observation_support": "native_instantaneous",
        "n_observations": len(selected_index),
    })
