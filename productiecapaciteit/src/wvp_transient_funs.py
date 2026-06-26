import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import fftconvolve
from scipy.special import exp1, k0

logger = logging.getLogger(__name__)

_INV_4PI = 1.0 / (4.0 * np.pi)
# Controls for the kd_grid integration method. The variable-kD convolution factors
# the leakage decay exp(-beta^2 t) out of the kernel, which makes the source amplitude
# grow like exp(+beta^2 t). When beta^2 * span is small the whole series fits one FFT;
# otherwise the work is split into blocks referenced to their own end time so the
# factor stays <= 1. _KD_GRID_BLOCK_SPAN caps beta^2 * (block span) (single-FFT
# threshold and block size). _KD_GRID_MEMORY_DECAY sets how far back the leakage decay
# stays non-negligible (exp(-decay)); beyond it the kernel is truncated. For very leaky
# aquifers (huge beta) the memory shrinks to a handful of cells and a direct banded
# near-window (no FFT, no blocking) is used instead, up to _KD_GRID_BAND_CAP cells.
_KD_GRID_BLOCK_SPAN = 17.0
_KD_GRID_MEMORY_DECAY = 22.0
_KD_GRID_BAND_CAP = 1000


def _kd_antiderivative_well_function(kappa, alpha2):
    """Antiderivative of g(w) = exp(-alpha^2 / w) / (4 pi w): (1 / 4 pi) E1(alpha^2 / w).

    This is the cumulative-transmissivity (kappa = integral of kD over time) form of
    the variable-kD Hantush kernel. ``E1`` is the exponential integral, finite for
    kappa > 0 and zero in the limit kappa -> 0.
    """
    kappa = np.asarray(kappa, dtype=float)
    out = np.zeros_like(kappa)
    positive = kappa > 0.0
    out[positive] = exp1(alpha2 / kappa[positive]) * _INV_4PI
    return out


def get_temp(index, mean, delta, time_offset, return_series=False):
    index_datetime = pd.DatetimeIndex(index)
    year = pd.Categorical(index_datetime.year, ordered=True)
    start_year = year.rename_categories(pd.to_datetime(year.categories, format="%Y"))
    end_year = year.rename_categories(pd.to_datetime(year.categories.astype(str) + "1231", format="%Y%m%d"))
    nday_year = end_year.map(lambda x: x.dayofyear, na_action="ignore").astype(float)
    dt_year = index_datetime - start_year.to_numpy()
    temp_data = delta * np.sin((dt_year / pd.Timedelta("1D") - time_offset) * 2 * np.pi / nday_year) + mean
    if return_series:
        return pd.Series(data=temp_data, index=index_datetime, name="wvp_model_temp")
    return temp_data.values


def visc_ratio(temp, temp_ref=10.0):
    visc_ref = (1 + 0.0155 * (temp_ref - 20.0)) ** -1.572  # / 1000  removed the division because we re taking a ratio.
    visc = (1 + 0.0155 * (temp - 20.0)) ** -1.572  # / 1000
    return visc / visc_ref


def dis(dis1, dis2):
    return (dis1 * dis1 + dis2 * dis2) ** 0.5


def as_float_array(name, values, size=None):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        if size is None:
            return arr.reshape(1)
        return np.full(size, float(arr), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1D array, got shape {arr.shape}")
    if size is not None and arr.size != size:
        raise ValueError(f"{name} must have length {size}, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def as_positive_integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value}")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def infer_lower_timestep(index):
    index = pd.DatetimeIndex(index)
    if index.size < 2:
        raise ValueError("index must contain at least two timestamps")
    dt_days = np.diff(index) / pd.Timedelta(1, unit="D")
    dt_days = dt_days[np.isfinite(dt_days) & (dt_days > 0.0)]
    if dt_days.size == 0:
        raise ValueError("index must be strictly increasing")
    return pd.Timedelta(float(dt_days.min()), unit="D")


def build_multiwell_geometry(
    dx_put,
    dx_mirrorwell=None,
    nput=None,
    *,
    r_mirrorwel=None,
    target_well_index=None,
    distance_scale=1.0,
    include_self=True,
    self_distance=1.0,
):
    """Build finite-row real-well and image-well terms for the transient model.

    Parameters
    ----------
    dx_put : float
        Distance between neighboring real wells along the row, in meters.
    dx_mirrorwell, r_mirrorwel : iterable
        Boundary specs from the config as ``(multiplicity, boundary_distance_m)``.
        ``dx_mirrorwell`` is kept as a backward-compatible alias.
        The image well is placed at twice this boundary distance. Negative
        multiplicities represent opposite-sign image wells for constant-head
        canal boundaries.
    nput : int
        Number of real wells in the row.
    target_well_index : int, optional
        Zero-based target well index. If omitted, one of the center wells is used.
    distance_scale : float, default 1.0
        Factor applied to all physical distances. Use ``1 / well_radius`` when
        passing the result to ``objective(..., multiwell_contains_r_self=True)``.
    include_self : bool, default True
        Include the target well itself as the first term.
    self_distance : float, default 1.0
        Distance assigned to the target well when ``include_self`` is true.
        For normalized distances this is one well radius.

    Returns
    -------
    list[tuple[float, float]], dict
        Multiwell terms ``(multiplicity, scaled_distance)`` and diagnostic counts.
    """
    dx_put = float(dx_put)
    if not np.isfinite(dx_put) or dx_put <= 0.0:
        raise ValueError(f"dx_put must be positive, got {dx_put}")

    if nput is None:
        raise ValueError("nput must be provided")
    nput_float = float(nput)
    if not np.isfinite(nput_float):
        raise ValueError(f"nput must be finite, got {nput}")
    nput_int = int(round(nput_float))
    if nput_int < 1 or not np.isclose(nput_float, nput_int):
        raise ValueError(f"nput must be a positive integer, got {nput}")

    if target_well_index is None:
        target_well_index = nput_int // 2
    target_well_index = int(target_well_index)
    if target_well_index < 0 or target_well_index >= nput_int:
        raise ValueError(f"target_well_index must be in [0, {nput_int - 1}], got {target_well_index}")

    distance_scale = float(distance_scale)
    if not np.isfinite(distance_scale) or distance_scale <= 0.0:
        raise ValueError(f"distance_scale must be positive, got {distance_scale}")
    self_distance = float(self_distance)
    if not np.isfinite(self_distance) or self_distance <= 0.0:
        raise ValueError(f"self_distance must be positive, got {self_distance}")

    neighbor_counts = defaultdict(int)
    for well_index in range(nput_int):
        if well_index == target_well_index:
            continue
        row_distance = abs(well_index - target_well_index) * dx_put
        neighbor_counts[row_distance] += 1
    neighbor_items = sorted(neighbor_counts.items())

    if r_mirrorwel is not None:
        if dx_mirrorwell is not None:
            raise ValueError("Specify only one of dx_mirrorwell and r_mirrorwel")
        dx_mirrorwell = r_mirrorwel

    if dx_mirrorwell is None:
        image_specs = []
    else:
        image_arr = np.asarray(dx_mirrorwell, dtype=float)
        if image_arr.size == 0:
            image_specs = []
        else:
            image_arr = np.atleast_2d(image_arr)
            if image_arr.shape[1] != 2:
                raise ValueError("dx_mirrorwell must contain (multiplicity, boundary_distance_m) pairs")
            if not np.isfinite(image_arr).all():
                raise ValueError("dx_mirrorwell contains NaN or infinite values")
            image_specs = [(float(multi), float(distance)) for multi, distance in image_arr]

    multiwell = []
    if include_self:
        multiwell.append((1.0, float(self_distance)))

    for row_distance, count in neighbor_items:
        multiwell.append((float(count), row_distance * distance_scale))

    for image_multi, boundary_distance in image_specs:
        if boundary_distance <= 0.0:
            raise ValueError(f"Mirror-well boundary distance must be positive, got {boundary_distance}")
        image_distance = 2.0 * boundary_distance
        if include_self:
            multiwell.append((image_multi, image_distance * distance_scale))
        for row_distance, count in neighbor_items:
            multiwell.append((
                count * image_multi,
                dis(row_distance, image_distance) * distance_scale,
            ))

    mirrorwell_multiplicity = sum(abs(multi) for multi, _ in image_specs)
    counts = {
        "self_wells": int(include_self),
        "neighbor_well_terms": len(neighbor_items),
        "neighbor_wells": nput_int - 1,
        "self_mirrorwell_terms": len(image_specs) if include_self else 0,
        "self_mirrorwells": mirrorwell_multiplicity if include_self else 0,
        "neighbor_mirrorwell_terms": len(image_specs) * len(neighbor_items),
        "neighbor_mirrorwells": mirrorwell_multiplicity * (nput_int - 1),
        "target_well_index": target_well_index,
        "nput": nput_int,
    }
    return multiwell, counts


def steady_multiwell_resistance_from_kd(
    kD,
    multiwell,
    nput,
    leakage_resistance_d,
    well_radius_m,
):
    """Return steady drawdown coefficient for a total flow input in m3/h.

    The coefficient includes conversion from total row flow in m3/h to per-well
    flow in m3/d, so multiplying the return value by total m3/h gives meters of
    drawdown at the target well.
    """
    kD = np.asarray(kD, dtype=float)
    leakage_resistance_d = float(leakage_resistance_d)
    well_radius_m = float(well_radius_m)
    nput = float(nput)
    if not np.isfinite(kD).all():
        raise ValueError("kD contains NaN or infinite values")
    if not np.isfinite(leakage_resistance_d) or leakage_resistance_d <= 0.0:
        raise ValueError(f"leakage_resistance_d must be positive, got {leakage_resistance_d}")
    if not np.isfinite(well_radius_m) or well_radius_m <= 0.0:
        raise ValueError(f"well_radius_m must be positive, got {well_radius_m}")
    if not np.isfinite(nput) or nput <= 0.0:
        raise ValueError(f"nput must be positive, got {nput}")
    if np.any(kD <= 0.0):
        raise ValueError("kD must be positive")

    leakage_factor = np.sqrt(kD * leakage_resistance_d)
    well_function_sum = np.zeros_like(kD, dtype=float)
    for multiplicity, normalized_distance in multiwell:
        multiplicity = float(multiplicity)
        normalized_distance = float(normalized_distance)
        if not np.isfinite([multiplicity, normalized_distance]).all():
            raise ValueError("multiwell contains NaN or infinite values")
        distance = normalized_distance * well_radius_m
        if distance <= 0.0:
            raise ValueError(f"multiwell distance must be positive, got {distance}")
        well_function_sum += multiplicity * 2.0 * k0(distance / leakage_factor)
    return 24.0 / nput * well_function_sum / (4.0 * np.pi * kD)


def solve_steady_multiwell_kd(
    target_resistance,
    multiwell,
    nput,
    leakage_resistance_d,
    well_radius_m,
    *,
    kd_min=1.0,
    kd_max=1_000.0,
):
    """Solve kD that reproduces a target steady multiwell resistance."""

    def residual(kD):
        return float(
            steady_multiwell_resistance_from_kd(
                kD,
                multiwell,
                nput,
                leakage_resistance_d,
                well_radius_m,
            )
            - target_resistance
        )

    target_resistance = float(target_resistance)
    if not np.isfinite(target_resistance) or target_resistance <= 0.0:
        raise ValueError(f"target_resistance must be positive, got {target_resistance}")
    kd_min = float(kd_min)
    kd_max = float(kd_max)
    if not np.isfinite([kd_min, kd_max]).all() or kd_min <= 0.0 or kd_max <= kd_min:
        raise ValueError(f"Expected 0 < kd_min < kd_max, got {kd_min=} and {kd_max=}")

    residual_min = residual(kd_min)
    residual_max = residual(kd_max)
    if residual_min < 0.0:
        raise ValueError(
            "Steady multiwell resistance is below target at the lower kD bound: "
            f"kD={kd_min:g} m2/d, target={target_resistance:.4g}"
        )
    if residual_max > 0.0:
        raise ValueError(
            "Steady multiwell resistance is above target at the upper kD bound: "
            f"kD={kd_max:g} m2/d, target={target_resistance:.4g}"
        )
    return brentq(residual, kd_min, kd_max, xtol=1e-10, rtol=1e-10)


def objective(args, return_result=False, **pextra):
    """
    Multialpha =

    Parameters
    ----------
    args
    return_result

    Returns
    -------

    """
    if "kD" in pextra:
        if len(args) < 2:
            raise ValueError("objective expects at least alpha and beta when kD is supplied")
        alpha, beta = args[:2]
        arg_idx = 2
        s = f"{alpha}, {beta}, kD=from pextra"
        kD = as_float_array("kD", pextra["kD"], pd.DatetimeIndex(pextra["index"]).size)
        if np.any(kD <= 0.0):
            raise ValueError("kD must be positive")
    else:
        if len(args) < 5:
            raise ValueError("objective expects at least five parameters")
        alpha, beta, kD0, temp_delta, temp_time_offset = args[:5]
        arg_idx = 5
        s = f"{alpha}, {beta}, {kD0}, {temp_delta}, {temp_time_offset}"
        if kD0 <= 0.0:
            raise ValueError(f"kD0 must be positive, got {kD0}")
        if not np.isfinite([temp_delta, temp_time_offset]).all():
            raise ValueError("Temperature model parameters must be finite")
        temp = get_temp(
            pextra["index"],
            pextra["temp_ref"],
            temp_delta,
            temp_time_offset,
            return_series=False,
        )
        kD = kD0 / visc_ratio(temp, temp_ref=pextra["temp_ref"])

    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")

    multiwell_contains_r_self = pextra.get("multiwell_contains_r_self", False)
    alpha_multi = None
    if pextra.get("multiwell") and not multiwell_contains_r_self:
        if "alpha_multi" in pextra:
            alpha_multi = pextra["alpha_multi"]
        else:
            if len(args) <= arg_idx:
                raise ValueError(
                    "objective needs alpha_multi when multiwell distances are not normalized by the self-well radius"
                )
            alpha_multi = args[arg_idx]
            arg_idx += 1
        s += f", {alpha_multi}"
    elif not pextra.get("multiwell") and multiwell_contains_r_self:
        raise ValueError("Define multiwell when multiwell_contains_r_self is True")

    if "rain" in pextra:
        raise NotImplementedError("Rain response is not implemented in objective()")

    if len(args) != arg_idx:
        raise ValueError(f"objective received {len(args)} parameters but consumed {arg_idx}")

    hantush_pextra = {key: value for key, value in pextra.items() if key != "kD"}

    def multi_variable_kd(*args):
        multi, alpha, beta, kD = args
        return multi * hantush_variable_kd(alpha, beta, kD, **hantush_pextra)

    hantush_args = []

    if multiwell_contains_r_self:
        for multi, distance in pextra["multiwell"]:
            hantush_args.append((multi, distance * alpha, beta, kD))
    else:
        hantush_args = [(1, alpha, beta, kD)]

        if pextra.get("multiwell"):
            for multi, distance in pextra["multiwell"]:
                hantush_args.append((multi, distance * alpha_multi * alpha, beta, kD))

    # if "rain" in pextra:

    if pextra.get("log_multiwell", False):
        counts = pextra.get("multiwell_counts", {})
        self_wells = counts.get("self_wells", 1)
        neighbor_wells = counts.get("neighbor_wells", 0)
        self_mirrorwells = counts.get("self_mirrorwells", 0)
        neighbor_mirrorwells = counts.get("neighbor_mirrorwells", 0)
        total_wells = self_wells + neighbor_wells
        total_mirrorwells = self_mirrorwells + neighbor_mirrorwells
        logger.info("objective parameters: %s", s)
        logger.info(
            "multiwell setup: "
            f"{total_wells} wells "
            f"(self={self_wells}, neighboring={neighbor_wells}; "
            f"neighbor terms={counts.get('neighbor_well_terms', 0)}), "
            f"{total_mirrorwells} mirror wells "
            f"(self mirrors={self_mirrorwells}, neighbor mirrors={neighbor_mirrorwells}; "
            f"mirror terms={counts.get('self_mirrorwell_terms', 0) + counts.get('neighbor_mirrorwell_terms', 0)}), "
            f"{len(hantush_args)} variable-kD Hantush evaluations"
        )

    if hantush_pextra.get("integration_method", "gauss") == "kd_grid":
        # The kd_grid method superposes all wells inside one shared-grid convolution,
        # so route the whole multiwell term list through a single evaluation.
        alpha_terms = [(multi, eff_alpha) for multi, eff_alpha, _beta, _kd in hantush_args]
        drawdown_model = hantush_variable_kd(alpha, beta, kD, **{**hantush_pextra, "alpha_terms": alpha_terms})
    elif len(hantush_args) == 1:
        results = [multi_variable_kd(*hantush_args[0])]
        drawdown_model = np.stack(results).sum(axis=0)
    else:
        max_workers = pextra.get("max_workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda args: multi_variable_kd(*args), hantush_args))
        drawdown_model = np.stack(results).sum(axis=0)

    if return_result:
        return drawdown_model

    drawdown_obs = np.asarray(pextra["drawdown_obs"], dtype=float)
    if drawdown_obs.shape != drawdown_model.shape:
        raise ValueError(f"drawdown_obs shape {drawdown_obs.shape} does not match model shape {drawdown_model.shape}")
    valid_obs = np.isfinite(drawdown_obs)
    if valid_obs.sum() == 0:
        raise ValueError("drawdown_obs contains no finite values")
    return drawdown_model[valid_obs] - drawdown_obs[valid_obs]


def get_perr(res):
    if np.any(res.active_mask):
        print(f"{res.active_mask} True for params at bounds")
    U, s, Vh = linalg.svd(res.jac, full_matrices=False)
    tol = np.finfo(float).eps * s[0] * max(res.jac.shape)
    w = s > tol
    cov = (Vh[w].T / s[w] ** 2) @ Vh[w]  # robust covariance matrix
    chi2dof = np.sum(res.fun**2) / (res.fun.size - res.x.size)
    cov *= chi2dof
    perr = np.sqrt(np.diag(cov))
    perr_rel = perr / res.x

    sl = []
    for xi, perr_ri in zip(res.x, perr_rel, strict=False):
        sl.append(f"{xi} +/- {perr_ri * 100:.1f}%")

    print("\n".join(sl))
    return perr


def hantush_variable_kd(alpha, beta, kD, **pextra):
    """Compute Hantush drawdown for spatially uniform, time-varying kD.

    This evaluates the variable-coefficient impulse response with
    ``Delta K = integral(kD(t), dt)`` between source and target times.

    ``Q_obs`` is interpreted as a piecewise-constant rate. By default,
    ``Q_obs[i]`` applies on ``[index[i], index[i + 1])``. Set
    ``flow_label="right"`` when ``Q_obs[i + 1]`` should apply on that interval.
    ``integration_method="gauss"`` is the default and uses ``n_gauss=32`` with
    ``max_gauss_step_days=0.5``. ``integration_method="quad"`` uses adaptive
    SciPy quadrature as a slower reference path. ``integration_method="kd_grid"``
    is the fast path: it convolves in cumulative-kD coordinates and interpolates
    back to time, scaling as O(nt log nt) instead of O(nt^2). ``n_per_step``
    (default 8) sets the grid resolution and ``near_steps`` (default 3) the width
    of the exactly-integrated near-diagonal window.
    """
    index = pd.DatetimeIndex(pextra["index"])
    nt = index.size
    if nt < 2:
        raise ValueError("index must contain at least two timestamps")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("index must be strictly increasing and unique")

    alpha = float(alpha)
    beta = float(beta)
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")

    q_obs = as_float_array("Q_obs", pextra["Q_obs"], nt)
    kD = as_float_array("kD", kD, nt)
    if np.any(kD <= 0.0):
        raise ValueError("kD must be positive")

    integration_method = pextra.get("integration_method", "gauss")
    if integration_method not in ("quad", "gauss", "kd_grid"):
        raise ValueError("integration_method must be 'quad', 'gauss', or 'kd_grid'")
    if pextra.get("tmax_days_cap") is not None:
        raise NotImplementedError(
            "tmax_days_cap is not supported for hantush_variable_kd because it "
            "would truncate the variable-kD convolution without a tail correction"
        )

    flow_label = pextra.get("flow_label", "left")
    if flow_label == "left":
        q_interval = q_obs[:-1]
    elif flow_label == "right":
        q_interval = q_obs[1:]
    else:
        raise ValueError("flow_label must be 'left' or 'right'")

    initial_condition = pextra.get("initial_condition", "steady")
    if initial_condition == "steady":
        initial_q = q_obs[0]
    elif initial_condition == "zero":
        initial_q = 0.0
    else:
        initial_q = float(initial_condition)
        if not np.isfinite(initial_q):
            raise ValueError("initial_condition must be 'steady', 'zero', or a finite number")

    time_days = (index - index[0]) / pd.Timedelta(1.0, unit="D")
    time_days = np.asarray(time_days, dtype=float)

    kd_fun = PchipInterpolator(time_days, kD, extrapolate=False)
    cumulative_kd_fun = kd_fun.antiderivative()
    cumulative_kd_offset = float(cumulative_kd_fun(time_days[0]))
    cumulative_kd = cumulative_kd_fun(time_days) - cumulative_kd_offset

    quad_epsabs = float(pextra.get("quad_epsabs", 1e-10))
    quad_epsrel = float(pextra.get("quad_epsrel", 1e-8))
    drawdown = np.zeros(nt, dtype=float)

    if integration_method == "kd_grid":
        # Multiwell superposition terms (multiplicity, effective alpha). Default to the
        # single self well when not called through the multiwell objective path.
        alpha_terms = pextra.get("alpha_terms")
        if alpha_terms is None:
            alpha_terms = [(1.0, alpha)]
        alpha2_terms = [(float(mult), float(a) * float(a)) for mult, a in alpha_terms]

        n_per_step = as_positive_integer("n_per_step", pextra.get("n_per_step", 8))
        near_steps = as_positive_integer("near_steps", pextra.get("near_steps", 3))
        drawdown += _variable_kd_rate_drawdown_kd_grid(
            alpha2_terms,
            beta,
            q_interval,
            time_days,
            cumulative_kd_fun,
            cumulative_kd_offset,
            cumulative_kd,
            n_per_step=n_per_step,
            near_steps=near_steps,
        )
        if initial_q != 0.0:
            for mult, a in alpha_terms:
                drawdown += mult * _variable_kd_initial_drawdown(
                    a,
                    beta,
                    float(kD[0]),
                    float(initial_q),
                    time_days,
                    cumulative_kd,
                    epsabs=quad_epsabs,
                    epsrel=quad_epsrel,
                )
        return drawdown

    if initial_q != 0.0:
        drawdown += _variable_kd_initial_drawdown(
            alpha,
            beta,
            float(kD[0]),
            float(initial_q),
            time_days,
            cumulative_kd,
            epsabs=quad_epsabs,
            epsrel=quad_epsrel,
        )

    if integration_method == "quad":
        drawdown += _variable_kd_rate_drawdown_quad(
            alpha,
            beta,
            q_interval,
            time_days,
            cumulative_kd_fun,
            cumulative_kd_offset,
            cumulative_kd,
            epsabs=quad_epsabs,
            epsrel=quad_epsrel,
        )
        return drawdown

    n_gauss = as_positive_integer("n_gauss", pextra.get("n_gauss", 32))
    max_gauss_step_days = float(pextra.get("max_gauss_step_days", 0.5))
    if not np.isfinite(max_gauss_step_days) or max_gauss_step_days <= 0.0:
        raise ValueError(f"max_gauss_step_days must be positive, got {max_gauss_step_days}")
    drawdown += _variable_kd_rate_drawdown_gauss(
        alpha,
        beta,
        q_interval,
        time_days,
        cumulative_kd_fun,
        cumulative_kd_offset,
        cumulative_kd,
        n_gauss=n_gauss,
        max_gauss_step_days=max_gauss_step_days,
        epsabs=quad_epsabs,
        epsrel=quad_epsrel,
    )
    return drawdown


def _variable_kd_rate_drawdown_quad(
    alpha,
    beta,
    q_interval,
    time_days,
    cumulative_kd_fun,
    cumulative_kd_offset,
    cumulative_kd,
    *,
    epsabs,
    epsrel,
):
    alpha2 = alpha * alpha
    beta2 = beta * beta
    drawdown = np.zeros(time_days.size, dtype=float)

    for target_idx in range(1, time_days.size):
        if not np.any(q_interval[:target_idx]):
            continue

        target_time = time_days[target_idx]
        target_cumulative_kd = cumulative_kd[target_idx]

        def integrand(source_time):
            source_idx = np.searchsorted(time_days, source_time, side="right") - 1
            source_idx = np.clip(source_idx, 0, q_interval.size - 1)
            d_k = target_cumulative_kd - (float(cumulative_kd_fun(source_time)) - cumulative_kd_offset)
            if d_k <= 0.0:
                return 0.0
            lag = target_time - source_time
            return q_interval[source_idx] * np.exp(-alpha2 / d_k - beta2 * lag) / (4.0 * np.pi * d_k)

        drawdown[target_idx] = quad(
            integrand,
            time_days[0],
            target_time,
            points=time_days[1:target_idx].tolist(),
            epsabs=epsabs,
            epsrel=epsrel,
            limit=max(50, 2 * target_idx),
        )[0]

    return drawdown


def _variable_kd_rate_drawdown_gauss(
    alpha,
    beta,
    q_interval,
    time_days,
    cumulative_kd_fun,
    cumulative_kd_offset,
    cumulative_kd,
    *,
    n_gauss,
    max_gauss_step_days,
    epsabs,
    epsrel,
):
    x_gauss, w_gauss = np.polynomial.legendre.leggauss(n_gauss)
    tau_blocks = []
    weight_blocks = []
    q_blocks = []
    interval_idx_blocks = []
    for interval_idx, (interval_left, interval_right) in enumerate(zip(time_days[:-1], time_days[1:])):
        n_substeps = max(
            int(np.ceil((interval_right - interval_left) / max_gauss_step_days)),
            1,
        )
        substep_edges = np.linspace(interval_left, interval_right, n_substeps + 1)
        substep_left = substep_edges[:-1]
        substep_right = substep_edges[1:]
        substep_mid = 0.5 * (substep_left + substep_right)
        substep_half_width = 0.5 * (substep_right - substep_left)
        substep_tau = substep_mid[:, None] + substep_half_width[:, None] * x_gauss
        substep_weights = substep_half_width[:, None] * w_gauss
        tau_blocks.append(substep_tau.ravel())
        weight_blocks.append(substep_weights.ravel())
        q_blocks.append(np.full(substep_tau.size, q_interval[interval_idx]))
        interval_idx_blocks.append(np.full(substep_tau.size, interval_idx))

    tau = np.concatenate(tau_blocks)
    weights = np.concatenate(weight_blocks)
    q_tau = np.concatenate(q_blocks)
    interval_idx = np.concatenate(interval_idx_blocks)
    cumulative_kd_tau = cumulative_kd_fun(tau) - cumulative_kd_offset

    alpha2 = alpha * alpha
    beta2 = beta * beta
    drawdown = np.zeros(time_days.size, dtype=float)
    for target_idx in range(1, time_days.size):
        source_end = np.searchsorted(interval_idx, target_idx - 1, side="left")
        if source_end > 0:
            d_k = cumulative_kd[target_idx] - cumulative_kd_tau[:source_end]
            lag = time_days[target_idx] - tau[:source_end]
            kernel = np.exp(-alpha2 / d_k - beta2 * lag) / (4.0 * np.pi * d_k)
            drawdown[target_idx] = np.sum(q_tau[:source_end] * kernel * weights[:source_end])

        interval_q = q_interval[target_idx - 1]
        if interval_q == 0.0:
            continue

        target_time = time_days[target_idx]
        target_cumulative_kd = cumulative_kd[target_idx]

        def recent_interval_integrand(source_time):
            d_k = target_cumulative_kd - (float(cumulative_kd_fun(source_time)) - cumulative_kd_offset)
            if d_k <= 0.0:
                return 0.0
            lag = target_time - source_time
            return interval_q * np.exp(-alpha2 / d_k - beta2 * lag) / (4.0 * np.pi * d_k)

        drawdown[target_idx] += quad(
            recent_interval_integrand,
            time_days[target_idx - 1],
            target_time,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=50,
        )[0]

    return drawdown


def _variable_kd_rate_drawdown_kd_grid(
    alpha2_terms,
    beta,
    q_interval,
    time_days,
    cumulative_kd_fun,
    cumulative_kd_offset,
    cumulative_kd,
    *,
    n_per_step,
    near_steps,
):
    """Rate-part variable-kD multiwell drawdown via convolution in cumulative-kD coordinates.

    Substituting ``kappa = K(t) = integral(kD)`` turns the Hantush kernel's
    transmissivity term into a convolution ``g(kappa_target - kappa_source)`` that is
    evaluated on a uniform ``kappa`` grid and interpolated back to the observation
    times. The leakage decay ``exp(-beta^2 (t - tau))`` is factored out of the kernel
    and handled per block (see :data:`_KD_GRID_BLOCK_SPAN`).

    ``alpha2_terms`` is an ``(n_terms, 2)`` array of ``(multiplicity, alpha^2)`` for the
    multiwell superposition (self well, neighbours and image wells). By linearity the
    whole superposition is a single convolution with the multiplicity-weighted sum of
    the per-term kernels, so all wells share one grid and one FFT. Only terms whose
    kernel is steep near the diagonal -- the small-``alpha^2`` finite-radius self well
    and immediate neighbours -- need the exact near-window integral on the data nodes;
    distant point-source wells are smooth and enter through the combined far kernel only.
    This is O(nt log nt) instead of the O(nt^2) ``gauss``/``quad`` paths.
    """
    alpha2_terms = np.atleast_2d(np.asarray(alpha2_terms, dtype=float))
    nt = time_days.size
    beta2 = beta * beta
    kd_fun = cumulative_kd_fun.derivative()
    t0 = time_days[0]
    t_last = time_days[-1]

    kappa_nodes = cumulative_kd
    kappa_max = float(kappa_nodes[-1])
    min_step = float(np.diff(kappa_nodes).min())
    if not np.isfinite(min_step) or min_step <= 0.0:
        raise ValueError("cumulative_kd must be strictly increasing for the kd_grid method")
    dk = min_step / float(n_per_step)
    n_grid = int(np.ceil(kappa_max / dk)) + 1
    node = np.arange(n_grid) * dk

    # Source density q / kD per grid cell [m*dk, (m+1)*dk], sampled at the midpoint.
    cell_mid = (np.arange(n_grid - 1) + 0.5) * dk
    t_mid = np.interp(cell_mid, kappa_nodes, time_days)
    j_mid = np.clip(np.searchsorted(time_days, t_mid, side="right") - 1, 0, nt - 2)
    rho = q_interval[j_mid] / kd_fun(np.clip(t_mid, t0, t_last))

    span = t_last - t0
    near_base = int(near_steps) * int(n_per_step)

    # Memory window: number of kappa cells over which the leakage decay exp(-beta^2 dt)
    # stays non-negligible. Beyond it the far kernel is truncated / the source is dropped.
    kd_max = float(np.max(np.diff(cumulative_kd) / np.diff(time_days)))
    if beta2 == 0.0:
        mem_cells = n_grid
    else:
        mem_cells = int(np.ceil(kd_max * _KD_GRID_MEMORY_DECAY / (beta2 * dk)))
    mem_cells = max(1, mem_cells)

    long_memory = beta2 == 0.0 or beta2 * span <= _KD_GRID_BLOCK_SPAN
    banded = (not long_memory) and mem_cells <= _KD_GRID_BAND_CAP

    if banded:
        # Very leaky aquifer: memory is a handful of cells. Cover the whole memory with
        # the exact direct near window below and skip the grid convolution entirely.
        near_cells = min(n_grid - 1, max(near_base, mem_cells))
    else:
        near_cells = near_base
    w_near = near_cells * dk

    # Steep (small-alpha^2) terms get the exact near window; smooth point-source terms
    # ride entirely on the combined far kernel. With no far convolution (banded) every
    # term uses the near window.
    if banded:
        near_term_mask = np.ones(alpha2_terms.shape[0], dtype=bool)
    else:
        near_term_mask = alpha2_terms[:, 1] <= w_near

    mults = alpha2_terms[:, 0]
    alpha2s = alpha2_terms[:, 1]

    far = np.zeros(nt)
    if not banded:
        # Combined edge-aligned cell-integrated kernel = multiplicity-weighted sum over
        # terms of (1/4pi) E1(alpha^2 / w), differenced across cells. Built with a single
        # batched E1 over (grid, terms) rather than one call per well. Truncated past the
        # leakage memory (blocked path). The near-treated terms then have their near-window
        # lags removed here (they are added back exactly in the near window below).
        kernel_len = n_grid if long_memory else min(n_grid, near_cells + mem_cells + 2)
        kernel_arg = np.arange(kernel_len + 1) * dk
        with np.errstate(divide="ignore"):
            ratio = alpha2s[None, :] / kernel_arg[:, None]  # arg 0 -> inf -> E1 = 0
        well_sum = (mults[None, :] * exp1(ratio)).sum(axis=1) * _INV_4PI
        far_kernel = np.diff(well_sum)
        clip_cells = min(near_cells, far_kernel.size)
        if near_term_mask.any() and clip_cells > 0:
            near_arg = np.arange(clip_cells + 1) * dk
            with np.errstate(divide="ignore"):
                near_ratio = alpha2s[near_term_mask][None, :] / near_arg[:, None]
            near_well_sum = (mults[near_term_mask][None, :] * exp1(near_ratio)).sum(axis=1) * _INV_4PI
            far_kernel[:clip_cells] -= np.diff(near_well_sum)

        if long_memory:
            t_ref = t_last
            source = rho * np.exp(beta2 * (t_mid - t_ref))
            conv = fftconvolve(source, far_kernel)[: n_grid - 1]
            c_far = np.empty(n_grid)
            c_far[0] = 0.0
            c_far[1:] = conv
            far = np.exp(-beta2 * (time_days - t_ref)) * np.interp(kappa_nodes, node, c_far)
        else:
            n_blocks = int(np.ceil(beta2 * span / _KD_GRID_BLOCK_SPAN))
            block_edges = np.linspace(t0, t_last, n_blocks + 1)
            memory_days = _KD_GRID_MEMORY_DECAY / beta2
            for b in range(n_blocks):
                lo_t, hi_t = block_edges[b], block_edges[b + 1]
                if b == 0:
                    target_mask = time_days <= hi_t + 1e-9
                else:
                    target_mask = (time_days > lo_t - 1e-9) & (time_days <= hi_t + 1e-9)
                if not target_mask.any():
                    continue
                t_ref = hi_t
                s0 = max(0, int(np.searchsorted(t_mid, lo_t - memory_days, side="left")))
                s1 = int(np.searchsorted(t_mid, hi_t, side="right"))
                if s1 <= s0:
                    continue
                source = rho[s0:s1] * np.exp(beta2 * (t_mid[s0:s1] - t_ref))
                conv = fftconvolve(source, far_kernel)
                kappa_t = kappa_nodes[target_mask]
                n_floor = np.clip(np.floor(kappa_t / dk).astype(np.int64), 0, n_grid - 1)
                n_ceil = np.clip(n_floor + 1, 0, n_grid - 1)
                frac = (kappa_t - node[n_floor]) / dk

                def _conv_at(node_idx, conv=conv, s0=s0, kappa_t=kappa_t):
                    flat_idx = node_idx - 1 - s0
                    out = np.zeros_like(kappa_t)
                    ok = (flat_idx >= 0) & (flat_idx < conv.size)
                    out[ok] = conv[flat_idx[ok]]
                    return out

                c_far = _conv_at(n_floor) * (1.0 - frac) + _conv_at(n_ceil) * frac
                far[target_mask] = np.exp(-beta2 * (time_days[target_mask] - t_ref)) * c_far

    # Near window: exact cumulative-kD integral over (K_i - w_near, K_i] on the data nodes
    # for the steep (near-treated) terms. The cell geometry and source are shared; only
    # the per-term well function differs, so it is summed with the term multiplicities.
    near = np.zeros(nt)
    if near_term_mask.any():
        top_node = np.floor(kappa_nodes / dk).astype(np.int64)
        near_lags = np.arange(near_cells + 1)
        cell = top_node[:, None] - near_lags[None, :]
        # There are n_grid - 1 cells (indices 0 .. n_grid - 2). When K_i lands on the
        # top node, top_node can equal n_grid - 1, whose cell is out of range; mark it
        # invalid instead of clipping it onto a real cell (which would double-count).
        valid = (cell >= 0) & (cell <= n_grid - 2)
        cell_clipped = np.clip(cell, 0, n_grid - 2)
        cell_lo = cell_clipped * dk
        cell_hi = (cell_clipped + 1) * dk
        kappa_target = kappa_nodes[:, None]
        seg_lo = np.maximum(cell_lo, kappa_target - w_near)
        seg_hi = np.minimum(cell_hi, kappa_target)
        has_segment = seg_hi > seg_lo
        arg_lo = kappa_target - seg_lo
        arg_hi = kappa_target - seg_hi
        d_well = np.zeros_like(arg_lo)
        for mult, alpha2 in alpha2_terms[near_term_mask]:
            d_well += mult * (
                _kd_antiderivative_well_function(arg_lo, alpha2)
                - _kd_antiderivative_well_function(arg_hi, alpha2)
            )
        seg_mid = (0.5 * (seg_lo + seg_hi)).ravel()
        t_seg = np.interp(seg_mid, kappa_nodes, time_days)
        j_seg = np.clip(np.searchsorted(time_days, t_seg, side="right") - 1, 0, nt - 2)
        rho_seg = (q_interval[j_seg] / kd_fun(np.clip(t_seg, t0, t_last))).reshape(cell.shape)
        t_seg = t_seg.reshape(cell.shape)
        contribution = rho_seg * np.exp(-beta2 * (time_days[:, None] - t_seg)) * d_well
        near = np.where(valid & has_segment, contribution, 0.0).sum(axis=1)

    drawdown = near + far
    drawdown[0] = 0.0
    return drawdown


def _variable_kd_initial_drawdown(
    alpha,
    beta,
    kD0,
    initial_q,
    time_days,
    cumulative_kd,
    *,
    epsabs,
    epsrel,
):
    alpha2 = alpha * alpha
    beta2 = beta * beta
    rho0 = 2.0 * alpha * beta / np.sqrt(kD0)

    out = np.empty_like(time_days, dtype=float)
    out[0] = initial_q / (4.0 * np.pi * kD0) * 2.0 * k0(rho0)

    for target_idx in range(1, time_days.size):
        lower_cumulative_kd = cumulative_kd[target_idx]
        target_lag = time_days[target_idx] - time_days[0]

        def integrand(cumulative_kd_total):
            return (
                np.exp(-alpha2 / cumulative_kd_total - beta2 * (cumulative_kd_total - lower_cumulative_kd) / kD0)
                / cumulative_kd_total
            )

        integral = quad(
            integrand,
            lower_cumulative_kd,
            np.inf,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=100,
        )[0]
        out[target_idx] = initial_q * np.exp(-beta2 * target_lag) / (4.0 * np.pi * kD0) * integral

    return out


# def rain_exp(index, rainfall, rain_gain, rain_time_scale, **pextra):
#     dP = np.diff(rainfall, prepend=0.0)
#     tmax = pd.Timedelta(-rain_time_scale * np.log(1 - pextra["frac_step_max), unit="D")
#     nt_max = tmax / pd.Timedelta(pextra["dt_lower"], unit="D")
#     nt = index.size
#     # nt_max = min(nt_max, nt)
#
#     it_arr = np.arange(nt_max)[None, :] + np.arange(nt)[:, None]
#     exclude = it_arr > (nt - 1)
#     it_arr[exclude] = nt - 1
#     # time_arr = (index[it_arr] - index[:, None]) / pd.Timedelta(1.0, unit="D")


def exp_impulse(time, time_scale):
    return np.exp(-time / time_scale)


def exp_step(time, time_scale):
    return -np.exp(-time / time_scale) + 1
