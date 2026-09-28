import bisect
import logging
import math
from itertools import pairwise
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import fftconvolve
from scipy.special import exp1, k0, k1

logger = logging.getLogger(__name__)

_INV_4PI = 1.0 / (4.0 * np.pi)
# Controls for the kd_grid integration method. The variable-kD convolution factors
# the leakage decay exp(-beta^2 t) out of the kernel, which makes the source amplitude
# grow like exp(+beta^2 t). When beta^2 * span is small the whole series fits one FFT;
# otherwise the work is split into blocks referenced to their own end time so the
# factor stays <= 1. _KD_GRID_BLOCK_SPAN caps beta^2 * (block span) (single-FFT
# threshold and block size). _KD_GRID_MEMORY_DECAY sets how far back the leakage decay
# stays non-negligible (exp(-decay)); beyond it the kernel is truncated, so very leaky
# aquifers (huge beta) get many short blocks with a short kernel each.
_KD_GRID_BLOCK_SPAN = 17.0
_KD_GRID_MEMORY_DECAY = 22.0
# Safety caps for the kd_grid method. The far-grid resolution dk is tied to the smallest
# cumulative-kD step, so a single very short interval can blow n_grid up; reject such
# pathological grids with a clear error instead of silently allocating multi-GB arrays.
# _KD_GRID_NEAR_MAX_ELEMENTS bounds the near-window row-block so peak memory does not scale
# with nt times the number of near-window segments per target.
_KD_GRID_MAX_NODES = 20_000_000
_KD_GRID_NEAR_MAX_ELEMENTS = 4_000_000
# Blocks whose source or kernel is at most this long are convolved directly: at small c there
# are hundreds of short blocks where FFT overhead dominates, and the direct sum also avoids the
# FFT round-off that referencing a block to its end time amplifies (up to exp(17)).
_KD_GRID_DIRECT_CONV_MAX = 1000
# Time sub-steps per data interval in the exact near window. The window is exact up to the
# piecewise-linear cumulative kD and source density within a sub-step, a second-order error
# (~1e-5 with seasonal kD), so two sub-steps suffice (see _variable_kd_near_window_drawdown).
_KD_GRID_NEAR_SUBSTEPS = 2
# Hantush well-function series controls (see _kd_antiderivative_well_function). A term whose
# leaky argument rho = 2 alpha beta / sqrt(kD) exceeds _HANTUSH_RHO_MAX is dropped: its whole
# well function 2 K0(rho) is below 2e-12 (the near-well terms are O(10)). The cap also bounds
# the series ratio b <= rho / 2 < 13, so the loop ends long before _HANTUSH_SERIES_MAX_TERMS,
# which is only a guard.
_HANTUSH_RHO_MAX = 26.0
_HANTUSH_SERIES_TOL = 1e-18
_HANTUSH_SERIES_MAX_TERMS = 200
# Far-kernel moment expansion (see _kd_grid_point_source_kernel): a term is folded into the
# combined power series once alpha^2 / w <= _KD_GRID_MOMENT_X_MAX, and evaluated with E1
# directly closer to its peak. The series is truncated where x^n / (n n!) < 1e-18 at that bound.
_KD_GRID_MOMENT_X_MAX = 4.0
_KD_GRID_MOMENT_TERMS = 34


def _hantush_series_threshold(n):
    """Smallest series ratio ``b`` for which term ``n`` (``b^n / n!``) still exceeds the tolerance."""
    return math.exp((math.log(_HANTUSH_SERIES_TOL) + math.lgamma(n + 1.0)) / n)


def _kd_antiderivative_well_function(kappa, alpha2, leakage_rate):
    """Leaky well-function tails ``(F, G)`` of the variable-kD Hantush kernel on a constant-kD segment.

    ``F(kappa) = (1 / 4 pi) int_kappa^inf exp(-alpha^2 / w - c w) / w dw`` and
    ``G(kappa) = (1 / 4 pi) int_kappa^inf exp(-alpha^2 / w - c w) dw`` in the
    cumulative-transmissivity coordinate (``kappa = integral of kD over time``), where the
    leakage decay is ``exp(-c w)`` with ``c = beta^2 / kD`` per unit cumulative transmissivity.
    With ``u = c kappa`` and ``rho = 2 alpha sqrt(c)``, ``4 pi F`` is the Hantush leaky well
    function ``W(u, rho)`` and ``4 pi c G = I(u) = int_u^inf exp(-y - rho^2 / 4y) dy``. Both come
    from Hantush's series in ``x = rho^2 / (4 u) = alpha^2 / kappa``:
    ``W(u, rho) = sum_n (-x)^n / n! E_{n+1}(u)`` and
    ``I(u) = e^-u + u sum_{n>=1} (-x)^n / n! E_n(u)``, summed in the ordering
    ``a = max(u, x) >= b = min(u, x)`` so they never cancel catastrophically. When ``u < x``
    the reflections ``W(u, rho) = 2 K0(rho) - W(x, rho)`` and
    ``I(u) = rho K1(rho) - u sum_n (-u)^n / n! E_{n+2}(x)`` swap the roles (exact at
    ``kappa = 0``: ``2 K0(rho)`` and ``rho K1(rho)``). ``E_n`` follows the upward recurrence
    ``E_{n+1}(a) = (e^-a - a E_n(a)) / n``, whose absolute error stays ``~eps E_1(a) e^a <=
    eps / a``, so the results are accurate to ~1e-15 absolute over the whole plane. Terms with
    ``rho`` above :data:`_HANTUSH_RHO_MAX` return zero.

    ``leakage_rate`` broadcasts against ``kappa`` (one ``c`` per segment).
    """
    kappa = np.asarray(kappa, dtype=float)
    leakage_rate = np.broadcast_to(np.asarray(leakage_rate, dtype=float), kappa.shape)
    rho2 = 4.0 * leakage_rate * alpha2
    keep = (rho2 < _HANTUSH_RHO_MAX * _HANTUSH_RHO_MAX).ravel()
    kappa_flat = np.maximum(kappa.ravel()[keep], 0.0)
    c_flat = leakage_rate.ravel()[keep]
    u = c_flat * kappa_flat
    with np.errstate(divide="ignore"):
        x = np.where(kappa_flat > 0.0, alpha2 / kappa_flat, np.inf)
    reflect = u < x
    a = np.minimum(np.maximum(u, x), 1.0e300)
    b = np.minimum(u, x)

    # Alternating series with term_n = (-b)^n / n!: W uses E_{n+1}(a), the direct I uses
    # E_n(a) (n >= 1) and the reflected I uses E_{n+2}(a). The loop runs while any element's
    # b^n / n! exceeds the tolerance; the working set is compacted to those elements whenever
    # it has halved (the others keep adding sub-tolerance terms until then).
    exp_neg_a = np.exp(-a)
    e_cur = exp1(a)
    e_next = exp_neg_a - a * e_cur
    well = e_cur.copy()
    moment_direct = np.zeros_like(a)
    moment_reflect = e_next.copy()
    idx = np.arange(a.size)
    a_sub, b_sub, exp_sub, term = a, b, exp_neg_a, np.ones_like(a)
    for n in range(1, _HANTUSH_SERIES_MAX_TERMS + 1):
        active = b_sub > _hantush_series_threshold(n)
        n_active = int(active.sum())
        if n_active == 0:
            break
        if active.size > 2 * n_active:
            idx = idx[active]
            a_sub, b_sub, exp_sub, term = a_sub[active], b_sub[active], exp_sub[active], term[active]
            e_cur, e_next = e_cur[active], e_next[active]
        e_prev, e_cur = e_cur, e_next
        e_next = (exp_sub - a_sub * e_cur) / (n + 1)
        term *= -b_sub / n
        well[idx] += term * e_cur
        moment_direct[idx] += term * e_prev
        moment_reflect[idx] += term * e_next

    moment = exp_neg_a + a * moment_direct
    if reflect.any():
        rho = np.sqrt(rho2.ravel()[keep][reflect])
        well[reflect] = 2.0 * k0(rho) - well[reflect]
        moment[reflect] = rho * k1(rho) - b[reflect] * moment_reflect[reflect]
    well_out = np.zeros(kappa.size)
    moment_out = np.zeros(kappa.size)
    well_out[keep] = well
    moment_out[keep] = moment / c_flat
    return well_out.reshape(kappa.shape) * _INV_4PI, moment_out.reshape(kappa.shape) * _INV_4PI


def _kd_grid_point_source_kernel(mults, alpha2s, w):
    """Sum the point-source well functions ``mult_m E1(alpha_m^2 / w)`` over the terms at the nodes ``w``.

    Terms far from their peak (``alpha_m^2 / w <= _KD_GRID_MOMENT_X_MAX``) are summed through
    the power series of ``E1``, ``E1(x) = -gamma - ln x + sum_n (-1)^(n+1) x^n / (n n!)``, which
    collapses the whole subset into one series in the moments ``M_n = sum mult_m alpha_m^2n``:
    ``sum_m mult_m E1(x_m) = M_0 (ln w - gamma) - sum mult_m ln alpha_m^2 + sum_n (-1)^(n+1)
    M_n / (n n! w^n)``. Sorting the terms by ``alpha^2`` makes that subset a prefix, so the
    moments are prefix sums looked up per node. Only the few nodes near each term's peak
    need a direct ``E1``; with ``w`` sorted those nodes are a prefix per term, so the cost grows
    only with the number of direct (node, term) pairs. Both branches are exact to round-off;
    ``w = 0`` gives zero (``E1(inf) = 0``).
    """
    w = np.asarray(w, dtype=float)
    out = np.zeros_like(w)
    if alpha2s.size == 0:
        return out
    order = np.argsort(alpha2s)
    alpha2_sorted = alpha2s[order]
    mult_sorted = mults[order]
    n_moment = np.arange(_KD_GRID_MOMENT_TERMS + 1)
    # prefix_moments[n, p] = sum over the p smallest alpha^2 of mult * alpha^(2n).
    prefix_moments = np.zeros((n_moment.size, alpha2s.size + 1))
    prefix_moments[:, 1:] = np.cumsum(mult_sorted * alpha2_sorted[None, :] ** n_moment[:, None], axis=1)
    prefix_log = np.r_[0.0, np.cumsum(mult_sorted * np.log(alpha2_sorted))]

    positive = w > 0.0
    w_order = np.argsort(w[positive], kind="stable")
    w_pos = w[positive][w_order]
    prefix = np.searchsorted(alpha2_sorted, _KD_GRID_MOMENT_X_MAX * w_pos, side="right")
    series = prefix_moments[0, prefix] * (np.log(w_pos) - np.euler_gamma) - prefix_log[prefix]
    inv_w_power = np.ones_like(w_pos)
    for n in range(1, _KD_GRID_MOMENT_TERMS + 1):
        inv_w_power /= w_pos
        series += (-1.0) ** (n + 1) / (n * math.factorial(n)) * prefix_moments[n, prefix] * inv_w_power

    # Direct E1 for the (node, term) pairs before the term joins the series (w < alpha^2 / x_max).
    n_direct = np.searchsorted(w_pos, alpha2_sorted / _KD_GRID_MOMENT_X_MAX, side="left")
    for mult, alpha2, n in zip(mult_sorted.tolist(), alpha2_sorted.tolist(), n_direct.tolist(), strict=True):
        series[:n] += mult * exp1(alpha2 / w_pos[:n])
    series_unsorted = np.empty_like(series)
    series_unsorted[w_order] = series
    out[positive] = series_unsorted
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


def visc_ratio(temp, temp_ref=12.0):
    visc_ref = (1 + 0.0155 * (temp_ref - 20.0)) ** -1.572  # / 1000  removed the division because we re taking a ratio.
    visc = (1 + 0.0155 * (temp - 20.0)) ** -1.572  # / 1000
    return visc / visc_ref


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


def as_positive_float(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _as_nput(nput):
    """Well count from the config, which stores it as a float (e.g. ``5.0``)."""
    nput_float = float(nput)
    if not np.isfinite(nput_float):
        raise ValueError(f"nput must be finite, got {nput}")
    nput_int = int(round(nput_float))
    if nput_int < 1 or not np.isclose(nput_float, nput_int):
        raise ValueError(f"nput must be a positive integer, got {nput}")
    return nput_int


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
    dx_put = as_positive_float("dx_put", dx_put)
    if nput is None:
        raise ValueError("nput must be provided")
    nput_int = _as_nput(nput)

    if target_well_index is None:
        target_well_index = nput_int // 2
    target_well_index = int(target_well_index)
    if target_well_index < 0 or target_well_index >= nput_int:
        raise ValueError(f"target_well_index must be in [0, {nput_int - 1}], got {target_well_index}")

    distance_scale = as_positive_float("distance_scale", distance_scale)
    self_distance = as_positive_float("self_distance", self_distance)

    # Neighbours on either side at the same offset share one term (multiplicity 2).
    well_offsets = np.abs(np.arange(nput_int) - target_well_index)
    offsets, offset_counts = np.unique(well_offsets[well_offsets > 0], return_counts=True)
    neighbor_items = list(zip((offsets * dx_put).tolist(), offset_counts.tolist(), strict=True))

    if r_mirrorwel is not None:
        if dx_mirrorwell is not None:
            raise ValueError("Specify only one of dx_mirrorwell and r_mirrorwel")
        dx_mirrorwell = r_mirrorwel
    image_specs = _parse_image_specs(dx_mirrorwell)

    multiwell = []
    if include_self:
        multiwell.append((1.0, self_distance))

    for row_distance, count in neighbor_items:
        multiwell.append((float(count), row_distance * distance_scale))

    for image_multi, boundary_distance in image_specs:
        image_distance = 2.0 * boundary_distance
        if include_self:
            multiwell.append((image_multi, image_distance * distance_scale))
        for row_distance, count in neighbor_items:
            multiwell.append((
                count * image_multi,
                float(np.hypot(row_distance, image_distance)) * distance_scale,
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


def _parse_image_specs(r_mirrorwel):
    """Normalize boundary specs into a list of ``(multiplicity, boundary_distance_m)``."""
    if r_mirrorwel is None:
        return []
    image_arr = np.asarray(r_mirrorwel, dtype=float)
    if image_arr.size == 0:
        return []
    image_arr = np.atleast_2d(image_arr)
    if image_arr.shape[1] != 2:
        raise ValueError("r_mirrorwel must contain (multiplicity, boundary_distance_m) pairs")
    if not np.isfinite(image_arr).all():
        raise ValueError("r_mirrorwel contains NaN or infinite values")
    for distance in image_arr[:, 1]:
        if distance <= 0.0:
            raise ValueError(f"Mirror-well boundary distance must be positive, got {distance}")
    return [(float(multi), float(distance)) for multi, distance in image_arr]


def crosssection_observation_points(
    nput,
    dx_put,
    distances,
    *,
    start="center",
    orientation="perpendicular",
):
    """Observation-point coordinates for a drawdown cross-section.

    The well row lies on the x-axis, well ``j`` at ``(j * dx_put, 0)``. A boundary
    (``r_mirrorwel``) is a line parallel to the row on the ``+y`` side; the
    cross-section is measured from the start well outward along one direction.

    Parameters
    ----------
    nput : int
        Number of real wells in the row.
    dx_put : float
        Spacing between neighboring wells, in meters.
    distances : array_like
        Nonnegative distances (m) from the start well at which to sample.
    start : {"center", "end"}
        Start the section at the center well or at the last (end) well.
    orientation : {"perpendicular", "along"}
        Direction of the section. ``"center"`` only allows ``"perpendicular"``
        (running away from the row, toward the ``+y`` boundary side). At the end
        well, ``"perpendicular"`` runs toward the boundary side and ``"along"``
        runs outward along the row axis, away from the well field.

    Returns
    -------
    (px, py, well_xs, start_index)
        ``px``/``py`` are the observation coordinates (one per distance),
        ``well_xs`` the real-well x positions, ``start_index`` the start well.
    """
    dx_put = as_positive_float("dx_put", dx_put)
    nput_int = _as_nput(nput)

    distances = as_float_array("distances", distances)
    if np.any(distances < 0.0):
        raise ValueError("cross-section distances must be nonnegative")

    well_xs = np.arange(nput_int, dtype=float) * dx_put

    if start == "center":
        if orientation != "perpendicular":
            raise ValueError("center cross-section must be perpendicular to the well row")
        start_index = nput_int // 2
        px = np.full(distances.shape, well_xs[start_index])
        py = distances.copy()
    elif start == "end":
        start_index = nput_int - 1
        if orientation == "perpendicular":
            px = np.full(distances.shape, well_xs[start_index])
            py = distances.copy()
        elif orientation == "along":
            px = well_xs[start_index] + distances
            py = np.zeros_like(distances)
        else:
            raise ValueError("end cross-section orientation must be 'perpendicular' or 'along'")
    else:
        raise ValueError("start must be 'center' or 'end'")

    return px, py, well_xs, start_index


def crosssection_image_offsets(r_mirrorwel, boundary_perp_offsets=None):
    """Resolve boundary specs into signed perpendicular canal offsets for a cross-section.

    ``r_mirrorwel`` stores only ``(multiplicity, distance)`` and discards which *side* of
    the well row each boundary sits on. That is lossless on the well axis (where
    ``dp_model``/``dp_steady`` live and the ``+b`` / ``-b`` images are equidistant) but a
    perpendicular cross-section samples off-axis, where the side matters. This resolves the
    side from the multiplicity:

    - ``(mult, b)`` with ``|mult| == 1`` -> a single canal; only allowed when it is the
      sole boundary, placed on ``+b`` (the section runs toward it).
    - ``(mult, b)`` with ``|mult| == 2`` -> two opposite-side canals at ``+b`` and ``-b``,
      each of strength ``sign(mult)`` (the dune-infiltration "wells between two panden"
      layout that the collapsed ``(-2, b)`` config entries encode).
    - anything else (mixed distances such as ``[(-1, 250), (-1, 82)]``, or ``|mult| > 2``)
      -> the side is ambiguous; raise ``NotImplementedError`` asking for explicit
      ``boundary_perp_offsets``.

    ``boundary_perp_offsets``, when given, is a list of ``(strength, signed_offset_m)``
    used directly (overriding ``r_mirrorwel``), so asymmetric strangen and "run away from
    the canal" sections stay expressible. The image well of a real well sits at
    ``y = 2 * signed_offset_m``.

    Returns a list of ``(strength, signed_offset_m)`` boundary-line offsets.
    """
    if boundary_perp_offsets is not None:
        offsets = []
        for strength, signed_offset in boundary_perp_offsets:
            strength = float(strength)
            signed_offset = float(signed_offset)
            if not np.isfinite([strength, signed_offset]).all():
                raise ValueError("boundary_perp_offsets contains NaN or infinite values")
            if signed_offset == 0.0:
                raise ValueError("boundary_perp_offsets entries must have a nonzero offset")
            offsets.append((strength, signed_offset))
        return offsets

    explicit_hint = "pass boundary_perp_offsets=[(strength, signed_offset_m), ...] explicitly"
    specs = _parse_image_specs(r_mirrorwel)
    offsets = []
    has_single = False
    for multi, boundary in specs:
        magnitude = int(round(abs(multi)))
        if magnitude not in {1, 2} or not np.isclose(abs(multi), magnitude):
            raise NotImplementedError(
                f"cross-section cannot infer canal sides for multiplicity {multi}; {explicit_hint}"
            )
        sign = 1.0 if multi > 0 else -1.0
        offsets.append((sign, boundary))
        if magnitude == 2:
            offsets.append((sign, -boundary))
        has_single |= magnitude == 1

    # A lone single-sided canal runs the section toward it; a single-sided canal that
    # coexists with any other boundary has an unknown side and must be made explicit.
    if has_single and len(specs) > 1:
        raise NotImplementedError(
            f"cross-section cannot infer canal sides for r_mirrorwel={r_mirrorwel!r}; {explicit_hint}"
        )
    return offsets


def build_crosssection_multiwell(px, py, well_xs, image_offsets, well_radius_m):
    """Multiwell terms for the drawdown at a single observation point ``(px, py)``.

    Returns ``(multiplicity, distance / well_radius)`` terms for every real well (on the
    row at ``y = 0``) and every image well (one per resolved boundary offset, placed at
    ``y = 2 * signed_offset``). ``image_offsets`` is the resolved
    ``(strength, signed_offset_m)`` list from :func:`crosssection_image_offsets`. Distances
    are clipped at one well radius so a point coinciding with a well reproduces that well's
    own drawdown rather than a singularity. On the well axis (``py == 0``) the ``+b`` /
    ``-b`` images are equidistant, so the output matches :func:`build_multiwell_geometry`
    and feeds the same ``objective``/``steady`` machinery.
    """
    well_xs = np.asarray(well_xs, dtype=float)
    well_radius_m = as_positive_float("well_radius_m", well_radius_m)

    # Row 0 is the real well row (y = 0); each image row sits at y = 2 * signed_offset.
    strengths = np.array([1.0] + [float(strength) for strength, _ in image_offsets])
    row_ys = np.array([0.0] + [2.0 * float(offset) for _, offset in image_offsets])
    distances = np.maximum(np.hypot(px - well_xs, py - row_ys[:, None]), well_radius_m) * (1.0 / well_radius_m)
    return list(zip(np.repeat(strengths, well_xs.size).tolist(), distances.ravel().tolist(), strict=True))


def steady_multiwell_resistance_from_kd(
    kD,
    multiwell,
    nput,
    leakage_resistance_d,
    well_radius_m,
):
    """Return steady drawdown coefficient for a total flow input in m3/h.

    Units: the returned coefficient has dimension meters per (m3/h). It already folds
    in the ``24 / nput`` conversion from total row flow in m3/h to per-well flow in
    m3/d (the De Glee well function is evaluated per well, then driven by the per-well
    rate), so ``coefficient * total_flow_m3h`` is meters of drawdown at the target well.
    """
    kD = np.asarray(kD, dtype=float)
    leakage_resistance_d = as_positive_float("leakage_resistance_d", leakage_resistance_d)
    well_radius_m = as_positive_float("well_radius_m", well_radius_m)
    nput = as_positive_float("nput", nput)
    if not np.isfinite(kD).all():
        raise ValueError("kD contains NaN or infinite values")
    if np.any(kD <= 0.0):
        raise ValueError("kD must be positive")

    terms = np.asarray(multiwell, dtype=float).reshape(-1, 2)
    if not np.isfinite(terms).all():
        raise ValueError("multiwell contains NaN or infinite values")
    multiplicities = terms[:, 0]
    distances = terms[:, 1] * well_radius_m
    if np.any(distances <= 0.0):
        raise ValueError(f"multiwell distance must be positive, got {distances.min()}")

    leakage_factor = np.sqrt(kD * leakage_resistance_d)
    well_function = 2.0 * k0(distances.reshape(-1, *([1] * kD.ndim)) / leakage_factor)
    well_function_sum = np.tensordot(multiplicities, well_function, axes=1)
    return 24.0 / nput * well_function_sum * _INV_4PI / kD


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

    target_resistance = as_positive_float("target_resistance", target_resistance)
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
    """Multiwell variable-kD Hantush drawdown, or its residual against ``drawdown_obs``.

    Parameters
    ----------
    args : sequence of float
        ``(alpha, beta)`` when ``pextra["kD"]`` is given, otherwise
        ``(alpha, beta, kD0, temp_delta, temp_time_offset)`` with kD from the seasonal
        temperature model. ``alpha_multi`` is appended when ``multiwell`` distances are
        not normalized by the self-well radius and ``pextra`` does not supply it.
    return_result : bool, default False
        Return the modelled drawdown instead of the residual.
    **pextra
        ``index``, ``Q_obs`` and the ``multiwell`` geometry, plus options forwarded to
        :func:`hantush_variable_kd`.

    Returns
    -------
    ndarray
        Modelled drawdown (``return_result=True``) or residual at the finite observations.
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

    # Multiwell superposition terms (multiplicity, effective alpha = distance * alpha).
    if multiwell_contains_r_self:
        alpha_terms = [(multi, distance * alpha) for multi, distance in pextra["multiwell"]]
    else:
        alpha_terms = [(1, alpha)]
        alpha_terms += [(multi, distance * alpha_multi * alpha) for multi, distance in pextra.get("multiwell") or []]

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
            f"{len(alpha_terms)} variable-kD Hantush terms"
        )

    hantush_pextra = {key: value for key, value in pextra.items() if key != "kD"}
    drawdown_model = hantush_variable_kd(alpha, beta, kD, **{**hantush_pextra, "alpha_terms": alpha_terms})

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
        logger.warning("%s True for params at bounds", res.active_mask)
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

    logger.info("%s", "\n".join(sl))
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

    ``alpha_terms``, a list of ``(multiplicity, alpha)``, superposes several wells (the
    multiwell geometry from :func:`objective`); ``alpha`` then only marks the finite-radius
    target well for the kd_grid near window. It defaults to the single well ``[(1, alpha)]``.
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

    time_days = np.asarray((index - index[0]) / pd.Timedelta(1.0, unit="D"), dtype=float)

    # PPoly.antiderivative() is zero at the first breakpoint, so cumulative_kd[0] == 0.
    kd_fun = PchipInterpolator(time_days, kD, extrapolate=False)
    cumulative_kd_fun = kd_fun.antiderivative()
    cumulative_kd = cumulative_kd_fun(time_days)

    # Multiwell superposition terms (multiplicity, effective alpha). Default to the
    # single self well when not called through the multiwell objective path.
    alpha_terms = pextra.get("alpha_terms")
    if alpha_terms is None:
        alpha_terms = [(1.0, alpha)]
    alpha_terms = np.asarray(alpha_terms, dtype=float).reshape(-1, 2)
    mults = alpha_terms[:, 0]
    alphas = alpha_terms[:, 1]
    alpha2s = alphas * alphas

    quad_epsabs = float(pextra.get("quad_epsabs", 1e-10))
    quad_epsrel = float(pextra.get("quad_epsrel", 1e-8))
    drawdown = np.zeros(nt, dtype=float)
    if initial_q != 0.0:
        drawdown += _variable_kd_initial_drawdown(
            mults,
            alphas,
            beta,
            float(kD[0]),
            float(initial_q),
            time_days,
            cumulative_kd,
            epsabs=quad_epsabs,
            epsrel=quad_epsrel,
        )

    if integration_method == "kd_grid":
        n_per_step = as_positive_integer("n_per_step", pextra.get("n_per_step", 8))
        near_steps = as_positive_integer("near_steps", pextra.get("near_steps", 3))
        drawdown += _variable_kd_rate_drawdown_kd_grid(
            mults,
            alpha2s,
            beta,
            q_interval,
            time_days,
            kd_fun,
            cumulative_kd_fun,
            cumulative_kd,
            finite_radius_alpha2=alpha * alpha,
            n_per_step=n_per_step,
            near_steps=near_steps,
        )
    elif integration_method == "quad":
        drawdown += _variable_kd_rate_drawdown_quad(
            mults,
            alpha2s,
            beta,
            q_interval,
            time_days,
            cumulative_kd_fun,
            cumulative_kd,
            epsabs=quad_epsabs,
            epsrel=quad_epsrel,
        )
    else:
        n_gauss = as_positive_integer("n_gauss", pextra.get("n_gauss", 32))
        max_gauss_step_days = as_positive_float("max_gauss_step_days", pextra.get("max_gauss_step_days", 0.5))
        drawdown += _variable_kd_rate_drawdown_gauss(
            mults,
            alpha2s,
            beta,
            q_interval,
            time_days,
            cumulative_kd_fun,
            cumulative_kd,
            n_gauss=n_gauss,
            max_gauss_step_days=max_gauss_step_days,
            epsabs=quad_epsabs,
            epsrel=quad_epsrel,
        )
    return drawdown


def _scalar_ppoly(ppoly):
    """Scalar evaluator of a 1-D ``PPoly`` without its per-call overhead.

    Uses the same per-interval power sum as SciPy. ``quad`` integrands evaluate the
    cumulative kD at every quadrature node, where ``PPoly.__call__`` (~40 us) dominates.
    """
    breaks = ppoly.x.tolist()
    coefs = ppoly.c[::-1].T.tolist()  # per interval, lowest order first
    last = len(coefs) - 1

    def evaluate(x):
        i = min(max(bisect.bisect_right(breaks, x) - 1, 0), last)
        s = x - breaks[i]
        result, power = 0.0, 1.0
        for c in coefs[i]:
            result += c * power
            power *= s
        return result

    return evaluate


def _rate_kernel(mults, alpha2s, beta2, d_k, lag):
    """Multiwell variable-kD Hantush impulse response ``sum_m mult_m exp(-alpha_m^2 / dK - beta^2 lag) / (4 pi dK)``."""
    well_sum = sum(
        mult * np.exp(-alpha2 / d_k - beta2 * lag)
        for mult, alpha2 in zip(mults.tolist(), alpha2s.tolist(), strict=True)
    )
    return well_sum * _INV_4PI / d_k


def _scalar_rate_kernel(mults, alpha2s, beta2):
    """Scalar :func:`_rate_kernel` for ``quad`` integrands, zero for ``dK <= 0``."""
    if mults.size == 1:
        mult, alpha2 = float(mults[0]), float(alpha2s[0])

        def kernel(d_k, lag):
            if d_k <= 0.0:
                return 0.0
            return mult * math.exp(-alpha2 / d_k - beta2 * lag) * _INV_4PI / d_k

    else:

        def kernel(d_k, lag):
            if d_k <= 0.0:
                return 0.0
            return float(mults @ np.exp(-alpha2s / d_k - beta2 * lag)) * _INV_4PI / d_k

    return kernel


def _variable_kd_rate_drawdown_quad(
    mults,
    alpha2s,
    beta,
    q_interval,
    time_days,
    cumulative_kd_fun,
    cumulative_kd,
    *,
    epsabs,
    epsrel,
):
    kernel = _scalar_rate_kernel(mults, alpha2s, beta * beta)
    cumulative_kd_at = _scalar_ppoly(cumulative_kd_fun)
    breaks = time_days.tolist()
    q_list = q_interval.tolist()
    last_interval = len(q_list) - 1
    has_flow = np.logical_or.accumulate(q_interval != 0.0)
    drawdown = np.zeros(time_days.size, dtype=float)

    for target_idx in range(1, time_days.size):
        if not has_flow[target_idx - 1]:
            continue

        target_time = breaks[target_idx]
        target_cumulative_kd = float(cumulative_kd[target_idx])

        def integrand(source_time, target_time=target_time, target_cumulative_kd=target_cumulative_kd):
            source_idx = min(max(bisect.bisect_right(breaks, source_time) - 1, 0), last_interval)
            d_k = target_cumulative_kd - cumulative_kd_at(source_time)
            return q_list[source_idx] * kernel(d_k, target_time - source_time)

        drawdown[target_idx] = quad(
            integrand,
            breaks[0],
            target_time,
            points=breaks[1:target_idx],
            epsabs=epsabs,
            epsrel=epsrel,
            limit=max(50, 2 * target_idx),
        )[0]

    return drawdown


def _variable_kd_rate_drawdown_gauss(
    mults,
    alpha2s,
    beta,
    q_interval,
    time_days,
    cumulative_kd_fun,
    cumulative_kd,
    *,
    n_gauss,
    max_gauss_step_days,
    epsabs,
    epsrel,
):
    # Gauss-Legendre nodes on every interval, split into equal substeps of at most
    # max_gauss_step_days (edges as np.linspace would place them).
    x_gauss, w_gauss = np.polynomial.legendre.leggauss(n_gauss)
    left, right = time_days[:-1], time_days[1:]
    n_substeps = np.maximum(np.ceil((right - left) / max_gauss_step_days).astype(np.int64), 1)
    substep_interval = np.repeat(np.arange(left.size), n_substeps)
    substep_rank = np.arange(substep_interval.size) - np.repeat(np.cumsum(n_substeps) - n_substeps, n_substeps)
    substep_width = ((right - left) / n_substeps)[substep_interval]
    substep_left = substep_rank * substep_width + left[substep_interval]
    substep_right = np.where(
        substep_rank + 1 == n_substeps[substep_interval],
        right[substep_interval],
        (substep_rank + 1) * substep_width + left[substep_interval],
    )
    substep_mid = 0.5 * (substep_left + substep_right)
    substep_half_width = 0.5 * (substep_right - substep_left)
    tau = (substep_mid[:, None] + substep_half_width[:, None] * x_gauss).ravel()
    q_weights = (q_interval[substep_interval, None] * (substep_half_width[:, None] * w_gauss)).ravel()
    cumulative_kd_tau = cumulative_kd_fun(tau)
    # Number of nodes in the intervals that have fully elapsed at each target; the
    # interval ending at the target is integrated adaptively below.
    source_end = np.r_[0, np.cumsum(n_substeps) * n_gauss]

    beta2 = beta * beta
    recent_kernel = _scalar_rate_kernel(mults, alpha2s, beta2)
    cumulative_kd_at = _scalar_ppoly(cumulative_kd_fun)
    drawdown = np.zeros(time_days.size, dtype=float)
    for target_idx in range(1, time_days.size):
        end = source_end[target_idx - 1]
        if end > 0:
            d_k = cumulative_kd[target_idx] - cumulative_kd_tau[:end]
            lag = time_days[target_idx] - tau[:end]
            drawdown[target_idx] = q_weights[:end] @ _rate_kernel(mults, alpha2s, beta2, d_k, lag)

        interval_q = float(q_interval[target_idx - 1])
        if interval_q == 0.0:
            continue

        target_time = float(time_days[target_idx])
        target_cumulative_kd = float(cumulative_kd[target_idx])

        def recent_interval_integrand(
            source_time, interval_q=interval_q, target_time=target_time, target_cumulative_kd=target_cumulative_kd
        ):
            d_k = target_cumulative_kd - cumulative_kd_at(source_time)
            return interval_q * recent_kernel(d_k, target_time - source_time)

        drawdown[target_idx] += quad(
            recent_interval_integrand,
            time_days[target_idx - 1],
            target_time,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=50,
        )[0]

    return drawdown


def _kd_grid_regime(beta, time_days, cumulative_kd, n_per_step):
    """Classify which kd_grid regime the inputs select and return the grid sizing.

    Returns ``(regime, dk, n_grid, mem_cells)`` where ``regime`` is one of
    ``"long_memory"`` or ``"blocked"``, ``dk`` is the uniform
    cumulative-kD grid step, ``n_grid`` the node count and ``mem_cells`` the leakage
    memory length in cells. This is the single source of truth for the regime decision,
    so tests can assert which path runs without re-deriving the thresholds.
    """
    beta2 = float(beta) * float(beta)
    span = float(time_days[-1] - time_days[0])
    kappa_max = float(cumulative_kd[-1])
    min_step = float(np.diff(cumulative_kd).min())
    if not np.isfinite(min_step) or min_step <= 0.0:
        raise ValueError("cumulative_kd must be strictly increasing for the kd_grid method")
    dk = min_step / float(n_per_step)
    n_grid = int(np.ceil(kappa_max / dk)) + 1
    kd_max = float(np.max(np.diff(cumulative_kd) / np.diff(time_days)))
    mem_cells = max(1, int(np.ceil(kd_max * _KD_GRID_MEMORY_DECAY / (beta2 * dk))))
    regime = "long_memory" if beta2 * span <= _KD_GRID_BLOCK_SPAN else "blocked"
    return regime, dk, n_grid, mem_cells


def _variable_kd_rate_drawdown_kd_grid(
    mults,
    alpha2s,
    beta,
    q_interval,
    time_days,
    kd_fun,
    cumulative_kd_fun,
    cumulative_kd,
    *,
    finite_radius_alpha2,
    n_per_step,
    near_steps,
):
    """Rate-part variable-kD multiwell drawdown via convolution in cumulative-kD coordinates.

    Substituting ``kappa = K(t) = integral(kD)`` turns the Hantush kernel's
    transmissivity term into a convolution ``g(kappa_target - kappa_source)`` that is
    evaluated on a uniform ``kappa`` grid and interpolated back to the observation
    times. The leakage decay ``exp(-beta^2 (t - tau))`` is factored out of the kernel
    and handled per block (see :data:`_KD_GRID_BLOCK_SPAN`).

    ``mults`` and ``alpha2s`` hold the ``(multiplicity, alpha^2)`` of the multiwell
    superposition (self well, neighbours and image wells). By linearity the whole
    superposition is a single convolution with the multiplicity-weighted sum of the
    per-term kernels, so all wells share one grid and one FFT.

    Only the well at which the head is of interest carries a **finite well radius**: its
    term sits at ``alpha^2 == finite_radius_alpha2`` (= ``r_well^2 * S / 4``), the smallest
    and steepest kernel, and gets the exact near-window integral on time sub-steps that
    resolves its sub-grid diagonal peak. All other wells in the series and every mirror
    well are modelled with an **infinitely small well radius** (point sources): their
    kernels are smooth at the relevant distances and ride the combined far kernel only,
    which is what makes them cheap. This is O(nt log nt) instead of the O(nt^2)
    ``gauss``/``quad`` paths.

    Accuracy note: the near window integrates each segment in closed form (Hantush well
    function on time sub-steps, :func:`_variable_kd_near_window_drawdown`), so the
    finite-radius target term is accurate to ~1e-5 independent of ``n_per_step``. The far
    convolution uses the exact cell-averaged source (:func:`_leaky_volume`) and the exact
    cell-integrated kernel, and its remaining error is the cell-scale resolution of the
    point-source kernels near their peaks (``alpha^2 ~ dk`` for the nearest neighbours),
    which is second order in ``dk``: ~1e-3 at ``n_per_step=8`` and ~1e-4 at 16 for a 15 m
    well spacing on 12-hourly data, for any leakage.
    """
    nt = time_days.size
    beta2 = beta * beta
    t0 = time_days[0]
    t_last = time_days[-1]

    kappa_nodes = cumulative_kd
    # Regime, grid step, node count and leakage-memory length (single source of truth).
    _regime, dk, n_grid, mem_cells = _kd_grid_regime(beta, time_days, kappa_nodes, n_per_step)
    if n_grid > _KD_GRID_MAX_NODES:
        raise ValueError(
            f"kd_grid would allocate {n_grid} grid nodes (cap {_KD_GRID_MAX_NODES}). The grid step is "
            "set by the smallest cumulative-kD interval, so a single very short time step forces a huge "
            "grid; resample to a more regular index before calling the kd_grid method."
        )
    node = np.arange(n_grid) * dk
    node_time = _invert_cumulative_kd(node, kappa_nodes, time_days, kd_fun, cumulative_kd_fun)
    # Target i sits between grid nodes n_floor and n_floor + 1 at fraction frac.
    n_floor = np.clip(np.floor(kappa_nodes / dk).astype(np.int64), 0, n_grid - 1)
    n_ceil = np.minimum(n_floor + 1, n_grid - 1)
    frac = (kappa_nodes - node[n_floor]) / dk

    near_cells = near_steps * n_per_step
    # Only the finite-radius target term (the well of interest, at alpha^2 ==
    # finite_radius_alpha2 = r_well^2 * S / 4) gets the exact near window; every other
    # well in the series and every mirror well is an infinitely small point source and
    # rides the combined far kernel only. The <= comparison (with a tiny relative
    # tolerance for the squaring round-off) selects the term(s) clipped to the well
    # radius and nothing farther; an off-well observation point has no such term, so it
    # is a pure point-source superposition.
    near_term_mask = alpha2s <= finite_radius_alpha2 * (1.0 + 1e-9)

    # Combined edge-aligned cell-integrated kernel = multiplicity-weighted sum over
    # terms of (1/4pi) E1(alpha^2 / w), differenced across cells (moment expansion for the
    # smooth terms, direct E1 near each peak), truncated past the leakage memory. The
    # near-treated terms enter only beyond the near window, w >= w_near (the window itself
    # is added exactly below).
    far = np.zeros(nt)
    kernel_len = min(n_grid, near_cells + mem_cells + 2)
    kernel_nodes = np.arange(kernel_len + 1) * dk
    point_mults, point_alpha2s = mults[~near_term_mask], alpha2s[~near_term_mask]
    near_mults, near_alpha2s = mults[near_term_mask], alpha2s[near_term_mask]
    w_near = near_cells * dk
    far_kernel = np.diff(_kd_grid_point_source_kernel(point_mults, point_alpha2s, kernel_nodes)) * _INV_4PI
    near = np.zeros(nt)
    if near_term_mask.any():
        near_e1 = _kd_grid_point_source_kernel(near_mults, near_alpha2s, np.maximum(kernel_nodes, w_near))
        far_kernel += np.diff(near_e1) * _INV_4PI
        boundary_time = _invert_cumulative_kd(
            np.maximum(kappa_nodes - w_near, 0.0), kappa_nodes, time_days, kd_fun, cumulative_kd_fun
        )
        far += _kd_grid_boundary_cell_correction(
            near_mults,
            near_alpha2s,
            q_interval,
            time_days,
            beta2,
            node_time,
            boundary_time,
            n_floor,
            frac,
            dk=dk,
            near_cells=near_cells,
        )
        # Near window: exact integral over (K_i - w_near, K_i] for the near-treated terms.
        near = _variable_kd_near_window_drawdown(
            near_mults,
            near_alpha2s,
            beta2,
            q_interval,
            time_days,
            kd_fun,
            cumulative_kd_fun,
            kappa_nodes,
            w_near=w_near,
            n_sub=_KD_GRID_NEAR_SUBSTEPS,
            near_steps=near_steps,
        )

    # Exact cell-averaged source q / kD exp(beta^2 (tau - t_ref)) over each cell, referenced to
    # the cell's own end so no factor overflows. dk is at most the smallest cumulative-kD step,
    # so a cell holds at most one data node and the leaky volume has at most two pieces.
    j_node = np.clip(np.searchsorted(time_days, node_time, side="right") - 1, 0, nt - 2)
    j_lo, j_hi = j_node[:-1], j_node[1:]
    t_lo, t_hi = node_time[:-1], node_time[1:]
    split = j_hi > j_lo
    t_split = np.where(split, time_days[j_hi], t_hi)
    cell_volume = q_interval[j_lo] * np.exp(beta2 * (t_lo - t_hi)) * np.expm1(beta2 * (t_split - t_lo))
    cell_volume += np.where(
        split, q_interval[j_hi] * np.exp(beta2 * (t_split - t_hi)) * np.expm1(beta2 * (t_hi - t_split)), 0.0
    )
    cell_volume /= beta2 * dk

    # Blocks of at most _KD_GRID_BLOCK_SPAN / beta^2 days (a single block in the long-memory
    # regime): each re-references its sources to the block end with a decaying factor,
    # convolves them with the far kernel and interpolates to its targets between grid nodes.
    n_blocks = int(np.ceil(beta2 * (t_last - t0) / _KD_GRID_BLOCK_SPAN))
    block_edges = np.linspace(t0, t_last, n_blocks + 1)
    memory_days = _KD_GRID_MEMORY_DECAY / beta2
    # Each target belongs to exactly one block, (lo_t, hi_t] up to round-off; targets are
    # sorted, so each block's targets are a slice.
    block_of = np.clip(np.searchsorted(block_edges, time_days - 1e-9, side="left") - 1, 0, n_blocks - 1)
    block_first = np.searchsorted(block_of, np.arange(n_blocks + 1))
    # Source cells ending after the memory start and starting before the block end.
    source_first = np.maximum(np.searchsorted(node_time, block_edges[:-1] - memory_days, side="right") - 1, 0)
    source_end = np.minimum(np.searchsorted(node_time, block_edges[1:], side="left"), n_grid - 1)
    # conv[k] holds grid node k + 1 + s0; nodes outside it get no far part.
    node_pair = np.stack([n_floor, n_ceil]) - 1
    c_pair = np.zeros((2, nt))
    for b in range(n_blocks):
        lo, hi = block_first[b], block_first[b + 1]
        s0, s1 = int(source_first[b]), int(source_end[b])
        if lo == hi or s1 <= s0:
            continue
        source = cell_volume[s0:s1] * np.exp(beta2 * (node_time[s0 + 1 : s1 + 1] - block_edges[b + 1]))
        if min(source.size, far_kernel.size) <= _KD_GRID_DIRECT_CONV_MAX:
            conv = np.convolve(source, far_kernel)
        else:
            conv = fftconvolve(source, far_kernel)
        flat_idx = node_pair[:, lo:hi] - s0
        inside = (flat_idx >= 0) & (flat_idx < conv.size)
        c_pair[:, lo:hi] = np.where(inside, conv[np.clip(flat_idx, 0, conv.size - 1)], 0.0)
    block_decay = np.exp(-beta2 * (time_days - block_edges[1:][block_of]))
    far += block_decay * (c_pair[0] * (1.0 - frac) + c_pair[1] * frac)

    drawdown = near + far
    drawdown[0] = 0.0
    return drawdown


def _invert_cumulative_kd(kappa, kappa_nodes, time_days, kd_fun, cumulative_kd_fun):
    """Times at which the cumulative transmissivity ``K(t)`` reaches ``kappa``.

    Two Newton steps on the PCHIP antiderivative from the linear-interpolation guess (off by
    ``O(dt^2 kD' / kD)``) reach round-off. Values past the last data node map to its time.
    """
    t0, t_last = time_days[0], time_days[-1]
    t = np.interp(kappa, kappa_nodes, time_days)
    for _ in range(2):
        t = np.clip(t - (cumulative_kd_fun(t) - kappa) / kd_fun(t), t0, t_last)
    return t


def _leaky_volume(t_lo, t_hi, t_ref, time_days, q_interval, beta2):
    """``int_{t_lo}^{t_hi} q(t) exp(beta^2 (t - t_ref)) dt``, exact for piecewise-constant ``q``.

    The pumped volume weighted by the leakage growth factor. Cell integrals of the source
    density ``q / kD exp(beta^2 (t - t_ref))`` over cumulative kD are exactly this
    (``d kappa / kD = d tau``), so data-interval boundaries inside a cell are honoured.
    Arrays broadcast; the pieces are summed per overlapped data interval. Callers keep
    ``t_hi`` at or below ``t_ref`` plus one step so the growth factor stays bounded.
    """
    nt = time_days.size
    t_lo, t_hi, t_ref = np.broadcast_arrays(t_lo, t_hi, t_ref)
    j_lo = np.clip(np.searchsorted(time_days, t_lo, side="right") - 1, 0, nt - 2)
    j_hi = np.clip(np.searchsorted(time_days, t_hi, side="right") - 1, 0, nt - 2)
    j = j_lo[..., None] + np.arange(int((j_hi - j_lo).max()) + 1)
    covered = j <= j_hi[..., None]
    j = np.minimum(j, nt - 2)
    piece_lo = np.maximum(time_days[j], t_lo[..., None])
    piece_hi = np.minimum(time_days[j + 1], t_hi[..., None])
    length = np.where(covered, np.maximum(piece_hi - piece_lo, 0.0), 0.0)
    return (q_interval[j] * np.exp(beta2 * (piece_lo - t_ref[..., None])) * np.expm1(beta2 * length)).sum(-1) / beta2


def _kd_grid_boundary_cell_correction(
    near_mults,
    near_alpha2s,
    q_interval,
    time_days,
    beta2,
    node_time,
    boundary_time,
    n_floor,
    frac,
    *,
    dk,
    near_cells,
):
    """Exact-minus-interpolated far part of the target term's near-window boundary cell, per target.

    The far part at ``K_i`` is interpolated between the two enclosing grid nodes. For the
    target term that weights the boundary cell ``[m_b dk, (m_b + 1) dk]``
    (``m_b = n_floor - near_cells``) by ``frac * kernel[near_cells]`` with its full-cell mean
    source, whereas the far part of a target at ``K_i`` covers only ``[m_b dk, K_i - w_near]``
    of it (ending at ``boundary_time``), with that part's own mean source; a pumping-rate jump
    inside the cell makes the two differ at first order in ``dk``. Both are closed forms
    (:func:`_leaky_volume`), so the interpolated piece is replaced by the exact one. Referenced
    to ``t_i`` the correction is block-independent. Every other cell of the target term is
    smooth on the cell scale (``w >= w_near``), so plain interpolation is second order there.
    """
    w_near = near_cells * dk
    width = frac * dk
    boundary_cell = n_floor - near_cells
    has_boundary = (boundary_cell >= 0) & (width > 0.0)
    # boundary_cell <= n_grid - 1 - near_cells, so only the lower clamp can bind.
    cell_lo = np.maximum(boundary_cell, 0)
    lo_time = node_time[cell_lo]
    hi_time = node_time[cell_lo + 1]
    safe_width = np.where(width > 0.0, width, 1.0)
    part_mean = _leaky_volume(lo_time, boundary_time, time_days, time_days, q_interval, beta2) / safe_width
    cell_mean = _leaky_volume(lo_time, hi_time, time_days, time_days, q_interval, beta2) / dk
    near_kernel_edges = _kd_grid_point_source_kernel(near_mults, near_alpha2s, np.array([w_near, w_near + dk]))
    part_kernel = _kd_grid_point_source_kernel(near_mults, near_alpha2s, w_near + width) - near_kernel_edges[0]
    cell_kernel = near_kernel_edges[1] - near_kernel_edges[0]
    return np.where(has_boundary, part_mean * part_kernel - frac * cell_mean * cell_kernel, 0.0) * _INV_4PI


def _variable_kd_near_window_drawdown(
    mults,
    alpha2s,
    beta2,
    q_interval,
    time_days,
    kd_fun,
    cumulative_kd_fun,
    kappa_nodes,
    *,
    w_near,
    n_sub,
    near_steps,
):
    """Near-window drawdown of the near-treated terms over the sources with ``K_i - K(tau) < w_near``.

    Computes ``sum_m mult_m int q exp(-alpha_m^2 / dK - beta^2 lag) / (4 pi dK) dtau`` exactly per
    segment. Every data interval is split into ``n_sub`` time sub-steps; on each the pumping rate is
    constant, the cumulative transmissivity is taken linear (``kD_seg`` = secant slope, ``K``
    exact at the sub-step edges) and the source density ``1 / kD`` linear between its edge
    values. With ``w = K_i - K(tau)`` the lag is ``t_i - tau = (t_i - tau_hi) + (w - w_lo) /
    kD_seg``, so the segment integral is the closed form
    ``q exp(-beta^2 (t_i - tau_hi) + c w_lo) [rho_hi (F(w_lo) - F(w_hi)) + rho' M1]`` with
    ``c = beta^2 / kD_seg``, ``rho_hi = 1 / kD(tau_hi)``, ``rho'`` the density slope in ``w``,
    ``F`` the leaky well-function tail (Hantush ``W``) and
    ``M1 = G(w_lo) - G(w_hi) - w_lo (F(w_lo) - F(w_hi))`` its first moment, both from
    :func:`_kd_antiderivative_well_function`. Clipping ``w_hi`` at ``w_near`` keeps the same
    linearisation. The remaining error is second order in the sub-step (curvature of ``K``
    and of ``1 / kD``), ~1e-5 with seasonal kD at two sub-steps. The near window spans at
    most ``near_steps`` data intervals because ``w_near`` is ``near_steps`` times the smallest
    cumulative-kD step.
    """
    nt = time_days.size
    n_lags = near_steps * n_sub
    # Global sub-step edges: data node j at index j * n_sub, so K and the lag are exact there.
    dt = np.diff(time_days)
    tau_sub = np.append(
        (time_days[:-1, None] + dt[:, None] * (np.arange(n_sub) / n_sub)[None, :]).ravel(),
        time_days[-1],
    )
    kappa_sub = cumulative_kd_fun(tau_sub)
    density_sub = 1.0 / kd_fun(tau_sub)
    n_sub_total = tau_sub.size - 1
    d_kappa_sub = np.diff(kappa_sub)
    density_slope = -np.diff(density_sub) / d_kappa_sub  # d(1/kD)/dw, w running backwards
    q_sub = np.repeat(q_interval, n_sub)
    leakage_sub = beta2 / (d_kappa_sub / np.diff(tau_sub))

    near = np.zeros(nt)
    lags = np.arange(n_lags)
    block_rows = max(1, _KD_GRID_NEAR_MAX_ELEMENTS // (2 * n_lags))
    for lo in range(1, nt, block_rows):
        hi = min(lo + block_rows, nt)
        rows = np.arange(lo, hi)
        seg = rows[:, None] * n_sub - 1 - lags[None, :]
        valid = seg >= 0
        seg = np.clip(seg, 0, n_sub_total - 1)
        kappa_target = kappa_nodes[lo:hi, None]
        w_lo = kappa_target - kappa_sub[seg + 1]
        w_hi = np.minimum(kappa_target - kappa_sub[seg], w_near)
        valid &= w_lo < w_near
        leakage = leakage_sub[seg]
        prefactor = q_sub[seg] * np.exp(-beta2 * (time_days[lo:hi, None] - tau_sub[seg + 1]) + leakage * w_lo)
        density_hi = density_sub[seg + 1]
        slope = density_slope[seg]
        edges_w = np.stack([w_lo, w_hi])
        d_well = np.zeros(seg.shape)
        for mult, alpha2 in zip(mults.tolist(), alpha2s.tolist(), strict=True):
            (well_lo, well_hi), (moment_lo, moment_hi) = _kd_antiderivative_well_function(edges_w, alpha2, leakage)
            d_well_term = well_lo - well_hi
            d_moment = moment_lo - moment_hi - w_lo * d_well_term
            d_well += mult * (density_hi * d_well_term + slope * d_moment)
        near[lo:hi] = np.where(valid, prefactor * d_well, 0.0).sum(axis=1)
    return near


def _variable_kd_initial_drawdown(
    mults,
    alphas,
    beta,
    kD0,
    initial_q,
    time_days,
    cumulative_kd,
    *,
    epsabs,
    epsrel,
    n_gauss=16,
):
    """Decaying drawdown of the steady pre-period flow ``initial_q`` (pumped at ``kD0``).

    At ``t0`` this is the De Glee steady state. For each later target ``i`` it is
    ``initial_q exp(-beta^2 t_i) / (4 pi kD0) * J_i`` with
    ``J_i = integral_{K_i}^inf G(k) exp(-b (k - K_i)) dk``, ``b = beta^2 / kD0`` and
    ``G(k) = sum_m mult_m exp(-alpha_m^2 / k) / k``. Splitting the integral at the data
    nodes gives the backward recursion ``J_i = c_i + exp(-b (K_{i+1} - K_i)) J_{i+1}``: the
    pieces ``c_i`` over ``[K_i, K_{i+1}]`` are vectorized Gauss-Legendre sums and only the
    tail beyond the last node needs adaptive quadrature.
    """
    beta2 = beta * beta
    b = beta2 / kD0
    nt = time_days.size
    out = np.empty(nt, dtype=float)
    out[0] = initial_q * _INV_4PI / kD0 * (mults @ (2.0 * k0(2.0 * alphas * beta / np.sqrt(kD0))))

    alpha2s = alphas * alphas
    terms = list(zip(mults.tolist(), alpha2s.tolist(), strict=True))
    kappa = cumulative_kd[1:]
    kappa_end = float(kappa[-1])

    def tail_integrand(k):
        return sum(mult * math.exp(-alpha2 / k - b * (k - kappa_end)) for mult, alpha2 in terms) / k

    # A distant well's integrand peaks near k = alpha^2, possibly far beyond kappa_end, where
    # one adaptive quad over [kappa_end, inf) can miss it. Split at doubling breakpoints up
    # to past the last peak and a long decay (exp(-40)), then close with the infinite tail.
    tail_span_end = kappa_end + float(alpha2s.max()) + 40.0 / b
    n_tail_edges = max(2, int(np.ceil(np.log2(tail_span_end / kappa_end))) + 1)
    tail_edges = np.append(np.geomspace(kappa_end, tail_span_end, n_tail_edges), np.inf)
    tail = sum(
        quad(tail_integrand, lo, hi, epsabs=epsabs, epsrel=epsrel, limit=100)[0] for lo, hi in pairwise(tail_edges)
    )

    # Gauss-Legendre over [K_i, K_{i+1}] for i = 1 .. nt - 2, subdivided so each piece is
    # short relative to both its distance from the 1/k singularity at k = 0 and the decay
    # length 1/b, which keeps the fixed rule at machine precision.
    lower, width = kappa[:-1], np.diff(kappa)
    n_pieces = np.maximum(np.ceil(np.maximum(b * width, width / lower)).astype(np.int64), 1)
    piece_interval = np.repeat(np.arange(lower.size), n_pieces)
    piece_rank = np.arange(piece_interval.size) - np.repeat(np.cumsum(n_pieces) - n_pieces, n_pieces)
    piece_width = (width / n_pieces)[piece_interval]
    x_gauss, w_gauss = np.polynomial.legendre.leggauss(n_gauss)
    k = (lower[piece_interval] + piece_rank * piece_width)[:, None] + 0.5 * piece_width[:, None] * (x_gauss + 1.0)
    well_sum = sum(mult * np.exp(-alpha2 / k) for mult, alpha2 in terms)
    integrand = well_sum / k * np.exp(-b * (k - lower[piece_interval, None]))
    pieces = np.bincount(piece_interval, weights=0.5 * piece_width * (integrand @ w_gauss), minlength=lower.size)

    # Backward recursion J_i = c_i + exp(-b (K_{i+1} - K_i)) J_{i+1}, starting from the tail.
    decay = np.exp(-b * width).tolist()
    pieces = pieces.tolist()
    j_integral = [0.0] * (nt - 1)
    j_integral[-1] = tail
    for i in range(nt - 3, -1, -1):
        j_integral[i] = pieces[i] + decay[i] * j_integral[i + 1]

    out[1:] = initial_q * np.exp(-beta2 * time_days[1:]) * _INV_4PI / kD0 * np.asarray(j_integral)
    return out
