"""Flow-driven, linear common-head Robin wells coupled to spatial shore sinks.

All source rates use m3/day; filter resistance uses day/m2, bed resistance
day/m. Positive shore amplitudes represent infiltration; well crossflow remains
signed. Hard well and total-flow constraints are never penalty equations.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd

from productiecapaciteit.src.wvp_transient_funs import IntervalResponse

_HARD_CLOSURE_RTOL = 1e-10
_MIN_RADIAL_NODES = 4


@dataclass
class BoundaryState:
    """One endpoint boundary solution, with unscaled physical residuals."""

    well_flow: np.ndarray
    shore_amplitude: np.ndarray
    common_head: float
    hard_residual: np.ndarray
    shore_residual: np.ndarray
    rank: tuple
    condition: tuple


def solve_boundary(kw, kc, p, weights, history_w, history_c, background, rf, rbed, total_flow) -> BoundaryState:
    """Solve exact Robin/flow constraints and arc-length-weighted shore LS.

    ``kw``/``kc`` multiply ``[well flows, positive infiltration amplitudes]``;
    their shore columns therefore contain negative Green responses. An SVD of
    row/column-scaled hard constraints supplies a particular solution and an
    orthonormal null space; a second SVD solves the reduced spatial objective.
    No normal equations or explicit inverse are formed.
    """
    kw, kc, p = (np.asarray(value, dtype=float) for value in (kw, kc, p))
    nw, ns = kw.shape
    na = ns - nw
    history_w = np.broadcast_to(np.asarray(history_w, float), (nw,))
    background = np.broadcast_to(np.asarray(background, float), (nw,))
    history_c = np.broadcast_to(np.asarray(history_c, float), (len(kc),))
    weights = np.asarray(weights, float)
    if (
        nw < 1
        or na < 0
        or kc.shape != (len(p), ns)
        or p.shape != (len(kc), na)
        or weights.shape != (len(kc),)
        or np.any(weights <= 0)
        or rf < 0
        or rbed < 0
    ):
        message = "invalid boundary dimensions, weights or negative resistance"
        raise ValueError(message)
    values = [kw.ravel(), kc.ravel(), p.ravel(), weights, history_w, history_c, background, [rf, rbed, total_flow]]
    if not np.isfinite(np.concatenate(values)).all():
        message = "boundary inputs must be finite"
        raise ValueError(message)
    c = np.zeros((nw + 1, ns + 1))
    c[:nw, :ns] = kw
    c[:nw, :nw] += rf * np.eye(nw)
    c[:nw, -1] = 1
    c[-1, :nw] = 1
    d = np.r_[background - history_w, total_flow]
    s = np.column_stack([kc, np.zeros(len(kc))])
    s[:, nw:ns] -= rbed * p
    f = -history_c
    # Explicit physical head/flow normalization followed by equilibration.
    flow_scale = max(abs(total_flow), 1)
    head_scale = max(np.max(np.abs(d[:-1])), (np.max(np.abs(kw)) + rf) * flow_scale, 1)
    row = np.r_[np.full(nw, 1 / head_scale), 1 / flow_scale]
    shore_scale = np.sqrt(weights / weights.sum()) / head_scale if len(weights) else weights
    cs, ss = c * row[:, None], s * shore_scale[:, None]
    column = np.linalg.norm(np.vstack([cs, ss]), axis=0)
    if np.any(column == 0):
        message = "rank deficient boundary column"
        raise np.linalg.LinAlgError(message)
    variable = 1 / column
    cs, ss = cs * variable, ss * variable
    equil = 1 / np.linalg.norm(cs, axis=1)
    cs, ds = cs * equil[:, None], d * row * equil
    u, sigma, vt = svd(cs, full_matrices=True)
    cutoff = np.finfo(float).eps * max(cs.shape) * sigma[0] * 10
    rank = int(np.sum(sigma > cutoff))
    if rank != len(c):
        message = "rank deficient hard well/flow constraints"
        raise np.linalg.LinAlgError(message)
    x0 = vt[:rank].T @ ((u.T @ ds) / sigma)
    z = vt[rank:].T
    reduced_condition = 1.0
    reduced_rank = 0
    if z.shape[1]:
        reduced = ss @ z
        ur, sr, vr = svd(reduced, full_matrices=False)
        reduced_rank = int(np.sum(sr > np.finfo(float).eps * max(reduced.shape) * sr[0] * 10)) if len(sr) else 0
        if reduced_rank != z.shape[1]:
            message = "rank deficient reduced shore problem; no unique boundary solution"
            raise np.linalg.LinAlgError(message)
        x0 += z @ (vr.T @ ((ur.T @ (f * shore_scale - ss @ x0)) / sr))
        reduced_condition = float(sr[0] / sr[-1])
    # One residual-refinement step restores hard closure lost to roundoff when
    # the absolute datum or filter loss is large. It does not relax constraints.
    correction = (d - c @ (x0 * variable)) * row * equil
    x0 += vt[:rank].T @ ((u.T @ correction) / sigma)
    x = x0 * variable
    hard = c @ x - d
    if (
        not np.isfinite(x).all()
        or np.max(np.abs(hard) / np.r_[np.full(nw, head_scale), flow_scale]) > _HARD_CLOSURE_RTOL
    ):
        message = "nonfinite solution or failed unscaled hard boundary closure"
        raise np.linalg.LinAlgError(message)
    return BoundaryState(
        x[:nw],
        x[nw:ns],
        float(x[-1]),
        hard,
        s @ x - f,
        (rank, reduced_rank),
        (float(sigma[0] / sigma[-1]), reduced_condition),
    )


def solve_common_head(a, h_free, total_flow: float, resistance: float) -> BoundaryState:
    """Solve a supplied aquifer matrix with no shores; retain signed flows."""
    a = np.asarray(a, float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:  # noqa: PLR2004 -- matrix dimensionality
        message = "aquifer matrix must be square"
        raise ValueError(message)
    return solve_boundary(a, np.empty((0, len(a))), np.empty((0, 0)), [], 0, [], h_free, resistance, 0, total_flow)


@dataclass
class BoundaryGeometry:
    """Geometry-only radial interpolation, separate from coefficient histories."""

    wells: np.ndarray
    controls: np.ndarray
    control_basis: np.ndarray
    weights: np.ndarray
    sources: np.ndarray
    source_basis: np.ndarray
    kernel: np.ndarray
    sinks: np.ndarray
    orders: np.ndarray
    well_radius: float
    radii: np.ndarray
    node_piece: np.ndarray

    def field_kernel(self, points) -> np.ndarray:
        """Build a geometry-only map operator using the shared log-radius spline."""
        from productiecapaciteit.reports.report_wvpweerstand_transient import radial_kernel  # noqa: PLC0415

        return radial_kernel(points, self.sources, self.source_basis, self.radii, self.well_radius)


def boundary_geometry(
    wells,
    shores,
    radii,
    well_radius: float,
    *,
    sink_offset_m: float = 0.5,
    order_spacing_m: float = 80,
    max_order: int = 2,
) -> BoundaryGeometry:
    """Reuse shore Legendre quadrature and log-radius interpolation, not time shapes."""
    import shapely  # noqa: PLC0415 -- geometry is optional for the pure algebra

    from productiecapaciteit.reports.report_wvpweerstand_transient import (  # noqa: PLC0415
        _legendre_basis,
        legendre_orders,
        linesink_nodes,
        radial_kernel,
    )

    wells = np.asarray(wells, float).reshape(-1, 2)
    if len(wells) < 1 or not np.isfinite(wells).all() or len(np.unique(wells, axis=0)) != len(wells):
        message = "wells require distinct finite coordinates"
        raise ValueError(message)
    if sink_offset_m <= 0 or well_radius <= 0:
        message = "sink offset and well radius must be positive"
        raise ValueError(message)
    shores = np.asarray(shores)
    radii = np.asarray(radii, float)
    if len(radii) < _MIN_RADIAL_NODES or radii[0] != well_radius:
        message = "at least four radial nodes starting at the well radius are required"
        raise ValueError(message)
    lengths = shapely.length(shores)
    orders = legendre_orders(lengths, order_spacing_m=order_spacing_m, max_order=max_order)
    sinks = shapely.offset_curve(shores, sink_offset_m)
    if len(shores):
        if np.any(shapely.is_empty(sinks)) or np.any(lengths <= 0):
            message = "empty offset sink or zero-length shore"
            raise ValueError(message)
        nodes, piece, node_basis = linesink_nodes(sinks, orders, sink_offset_m / 2)
        counts = 3 * (orders + 1)
        fractions = [(1 - np.cos(np.pi * np.arange(n) / (n - 1))) / 2 for n in counts]
        controls, _, p = _legendre_basis(shores, orders, fractions)
        weights = np.concatenate([
            length * np.r_[np.diff(f)[0] / 2, (f[2:] - f[:-2]) / 2, np.diff(f)[-1] / 2]
            for length, f in zip(lengths, fractions, strict=True)
        ])
    else:
        nodes, piece, node_basis = np.empty((0, 2)), np.empty(0, int), np.empty((0, 0))
        controls, p, weights = np.empty((0, 2)), np.empty((0, 0)), np.empty(0)
    nw, na = len(wells), node_basis.shape[1]
    sources = np.concatenate([wells, nodes])
    strengths = np.zeros((len(sources), nw + na))
    strengths[:nw, :nw] = np.eye(nw)
    strengths[nw:, nw:] = -node_basis
    kernel = radial_kernel(np.concatenate([wells, controls]), sources, strengths, radii, well_radius)
    return BoundaryGeometry(
        wells, controls, p, weights, sources, strengths, kernel, sinks, orders, well_radius, radii, piece
    )


def interval_mean(function, edges, *, order: int = 12) -> np.ndarray:
    """Gauss time-averages of a continuous prediction, not endpoint states.

    ``function`` accepts a 1D time array and returns time on its leading axis.
    Split ``edges`` at all forcing and cleaning discontinuities first.
    """
    edges = np.asarray(edges, float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0) or order < 2:  # noqa: PLR2004
        message = "ordered edges and quadrature order >= 2 required"
        raise ValueError(message)
    x, w = np.polynomial.legendre.leggauss(order)
    times = (edges[:-1, None] + edges[1:, None]) / 2 + np.diff(edges)[:, None] * x / 2
    values = np.asarray(function(times.ravel()))
    return np.einsum("ij...,j->i...", values.reshape(len(edges) - 1, order, *values.shape[1:]), w / 2)


@dataclass
class RobinSolution:
    """Compact source and endpoint boundary histories (no map-grid history)."""

    response: IntervalResponse
    geometry: BoundaryGeometry
    rates: np.ndarray
    initial_rates: np.ndarray
    initial_common_head: float
    common_head: np.ndarray
    well_drawdown: np.ndarray
    hard_residual: np.ndarray
    shore_residual: np.ndarray
    ranks: np.ndarray
    conditions: np.ndarray
    history_settings: tuple
    history_counts: tuple
    boundary_inputs: tuple

    @property
    def well_flow(self) -> np.ndarray:
        """Signed individual well rates in m3/day."""
        return self.rates[:, : len(self.geometry.wells)]

    @property
    def shore_amplitude(self) -> np.ndarray:
        """Positive infiltration amplitudes in m2/day."""
        return self.rates[:, len(self.geometry.wells) :]

    def radial_at(self, targets) -> np.ndarray:
        """Evaluate source histories at ordered arbitrary times in days."""
        targets = np.atleast_1d(targets).astype(float)
        if np.any(np.diff(targets) < 0) or np.any(targets > self.response.days[len(self.rates)]):
            message = "field targets must be ordered and within the solved history"
            raise ValueError(message)
        initial = self.response.initial_response(targets)[:, :, None] * self.initial_rates
        history = self.response.history(
            self.rates.shape[1], rtol=self.history_settings[0], atol=self.history_settings[1]
        )
        committed = 0
        for i, t in enumerate(targets):
            while committed < len(self.rates) and self.response.days[committed] < t:
                history.append(committed, self.rates[committed])
                committed += 1
            initial[i] += history.evaluate(t)
        return initial

    def drawdown(self, points, targets, *, chunk_points: int = 128) -> np.ndarray:
        """Chunk field evaluation, never a full space-time response tensor."""
        points = np.asarray(points, float).reshape(-1, 2)
        radial = self.radial_at(targets)
        out = np.empty((len(radial), len(points)))
        for start in range(0, len(points), chunk_points):
            kernel = self.geometry.field_kernel(points[start : start + chunk_points])
            out[:, start : start + chunk_points] = np.einsum("trs,prs->tp", radial, kernel, optimize=True)
        return out


def solve_transient(  # noqa: C901 -- validation, initialization and chronological advancement are one solve
    response,
    geometry: BoundaryGeometry,
    total_flow,
    resistance,
    bed_resistance,
    background,
    *,
    initial: str = "zero",
    initial_flow: float | None = None,
    initial_resistance: float | None = None,
    initial_bed_resistance: float | None = None,
    initial_background: float | None = None,
    history_rtol: float = 1e-8,
    history_atol: float = 1e-11,
    stop_after: int | None = None,
    resume: RobinSolution | None = None,
) -> RobinSolution:
    """Advance a simultaneous well/shore boundary solve chronologically.

    Arrays have one value per interval (N-1), collocated at its right endpoint.
    Cleaning intervals must already be split and their endpoint filter coefficient
    evaluated on the left. Steady initialization solves the same boundary problem
    at explicitly supplied initial coefficients, then carries that field forward.
    ``stop_after``/``resume`` checkpoint the compact history on the *same*
    prescribed coefficient grid, avoiding a PCHIP change at a restart boundary.
    """
    n = len(response.days) - 1
    arrays = [np.broadcast_to(np.asarray(v, float), (n,)) for v in (total_flow, resistance, bed_resistance, background)]
    if any(not np.isfinite(a).all() for a in arrays) or np.any(arrays[1] < 0) or np.any(arrays[2] < 0):
        message = "finite forcing/background and nonnegative resistances required"
        raise ValueError(message)
    if initial not in {"zero", "steady"} or (initial == "steady" and initial_flow is None):
        message = "initial must be zero or steady with explicit initial_flow"
        raise ValueError(message)
    if not np.array_equal(response.radii, geometry.radii):
        message = "response and geometry radii must agree"
        raise ValueError(message)
    stop = n if stop_after is None else stop_after
    if not isinstance(stop, int) or not 1 <= stop <= n:
        message = "stop_after must identify an interval in the coefficient grid"
        raise ValueError(message)
    begin = 0 if resume is None else len(resume.rates)
    if begin > stop:
        message = "resume history extends beyond stop_after"
        raise ValueError(message)
    if resume is not None and (
        not np.array_equal(response.days, resume.response.days)
        or not np.array_equal(response.kd, resume.response.kd)
        or response.storage != resume.response.storage
        or response.leakage != resume.response.leakage
        or not np.array_equal(geometry.kernel, resume.geometry.kernel)
        or (history_rtol, history_atol) != resume.history_settings
        or any(not np.array_equal(a[:begin], b[:begin]) for a, b in zip(arrays, resume.boundary_inputs, strict=True))
    ):
        message = "resume requires unchanged geometry, coefficient grid, history settings and past forcing"
        raise ValueError(message)
    nw = len(geometry.wells)
    ns = geometry.source_basis.shape[1]
    rates = np.empty((n, ns))
    heads, drawdown = np.empty(n), np.empty((n, nw))
    hard, shore = np.empty((n, nw + 1)), np.empty((n, len(geometry.controls)))
    ranks, conditions = np.empty((n, 2), int), np.empty((n, 2))
    initial_rates = np.zeros(ns)
    initial_head = float(arrays[3][0] if initial_background is None else initial_background)

    def boundary(matrix, past, rf, rb, bg, flow):
        return solve_boundary(
            matrix[:nw], matrix[nw:], geometry.control_basis, geometry.weights, past[:nw], past[nw:], bg, rf, rb, flow
        )

    if resume is not None:
        initial_rates = resume.initial_rates.copy()
        initial_head = resume.initial_common_head
        rates[:begin], heads[:begin], drawdown[:begin] = resume.rates, resume.common_head, resume.well_drawdown
        hard[:begin], shore[:begin] = resume.hard_residual, resume.shore_residual
        ranks[:begin], conditions[:begin] = resume.ranks, resume.conditions
    elif initial == "steady":
        matrix = np.einsum("r,prs->ps", response.steady_response(), geometry.kernel, optimize=True)
        state = boundary(
            matrix,
            np.zeros(len(matrix)),
            arrays[1][0] if initial_resistance is None else initial_resistance,
            arrays[2][0] if initial_bed_resistance is None else initial_bed_resistance,
            initial_head,
            initial_flow,
        )
        initial_rates = np.r_[state.well_flow, state.shore_amplitude]
        initial_head = state.common_head
    initial_field = response.initial_response()
    history = response.history(ns, rtol=history_rtol, atol=history_atol)
    for j in range(begin):
        history.append(j, rates[j])
    for j in range(begin, stop):
        t = response.days[j + 1]
        radial_past = history.evaluate(t) + initial_field[j + 1, :, None] * initial_rates
        past = np.einsum("rs,prs->p", radial_past, geometry.kernel, optimize=True)
        current = response.interval_weights(t, [j])[:, 0]
        matrix = np.einsum("r,prs->ps", current, geometry.kernel, optimize=True)
        state = boundary(matrix, past, arrays[1][j], arrays[2][j], arrays[3][j], arrays[0][j])
        rates[j] = np.r_[state.well_flow, state.shore_amplitude]
        heads[j], hard[j], shore[j] = state.common_head, state.hard_residual, state.shore_residual
        ranks[j], conditions[j] = state.rank, state.condition
        drawdown[j] = past[:nw] + matrix[:nw] @ rates[j]
        history.append(j, rates[j])
    return RobinSolution(
        response,
        geometry,
        rates[:stop],
        initial_rates,
        initial_head,
        heads[:stop],
        drawdown[:stop],
        hard[:stop],
        shore[:stop],
        ranks[:stop],
        conditions[:stop],
        (history_rtol, history_atol),
        (history.accepted_blocks, history.direct_intervals),
        tuple(a.copy() for a in arrays),
    )
