"""Tests for the 2D head map (transient line-sinks) in report_wvpweerstand_transient."""

import numpy as np
import pandas as pd
import pytest
import shapely
from numpy.polynomial import legendre
from scipy.special import k0

from productiecapaciteit.reports.report_wvpweerstand_transient import (
    _legendre_basis,
    bed_resistance,
    default_transient_coefficients,
    distance_table,
    drawdown_from_kernel,
    facing_shores,
    infiltration_by_body_and_side,
    lagged_histories,
    linesink_sources,
    map_sources,
    prepare_water,
    radial_kernel,
    row_segments,
    solve_transient_linesinks,
    split_shores,
    strip_bed_resistance,
    time_shapes,
    well_row_normals,
)
from productiecapaciteit.src.wvp_transient_funs import objective

NPUT = 21
DX = 15.0
FLOW_FIRST_WEEK_MEAN = "first_week_mean"
WELL_RADIUS = 0.3
SINK_OFFSET = 0.5


def _visc_ratio(temperature, temperature_ref=12.0):
    return ((1 + 0.0155 * (temperature - 20.0)) / (1 + 0.0155 * (temperature_ref - 20.0))) ** -1.572


def _row(n, dx, y=0.0):
    return np.column_stack([np.arange(n) * dx, np.full(n, y)])


def _k0_tables(radii, kd, leakage_factor, shapes):
    """Quasi-steady tables: every shape's drawdown is its current flow times the De Glee response."""
    response = k0(np.asarray(radii) / leakage_factor) / (2.0 * np.pi * kd)
    return response[None, :, None] * np.asarray(shapes, dtype=float)[:, None, :]


def _field(solution, tables, radii, wells_xy, points, spacing=SINK_OFFSET / 2.0):
    """Drawdown at points and all times through the map path: wells pump shape 0, nodes their strengths."""
    sources, strengths = map_sources(solution, wells_xy, spacing)
    return drawdown_from_kernel(tables, radial_kernel(points, sources, strengths, radii, WELL_RADIUS))


def _inflow_along(solution, shapes, piece, fractions):
    """Infiltration per metre q' at arc-length fractions of one shore piece (m2/d)."""
    per_piece = [np.asarray(fractions if p == piece else [], dtype=float) for p in range(solution["orders"].size)]
    _, _, basis = _legendre_basis(solution["sinks"], solution["orders"], per_piece)
    return -shapes @ (basis @ solution["coefficients"].T).T


# --------------------------------------------------------------------------- #
# Distance tables and the radial kernel
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def seasonal_coefficients():
    """Coefficients with a seasonal kD, as all calibrated strangen have."""
    return default_transient_coefficients(
        temperature_method="sin", temperature_delta_degc=5.0, temperature_time_offset_d=40.0
    )


@pytest.fixture(scope="module")
def gappy_index():
    """31 twelve-hourly steps with gaps, like the calibration dataframe."""
    return pd.date_range("2021-01-01", periods=36, freq="12h").delete([5, 6, 7, 20, 21])


@pytest.fixture(scope="module")
def flow_m3h(gappy_index):
    """Total strang flow that changes every step."""
    return np.random.default_rng(42).uniform(20.0, 120.0, gappy_index.size)


@pytest.mark.parametrize("initial_condition", ["zero", FLOW_FIRST_WEEK_MEAN])
def test_distance_table_matches_gauss_near_and_far(seasonal_coefficients, gappy_index, flow_m3h, initial_condition):
    # The point-source kd_grid path is off by decimeters within meters of the well under
    # seasonal kD and step-changing flow; each column must carry its own near window.
    q_per_well = flow_m3h / NPUT * 24.0
    ic = float(q_per_well[:14].mean()) if initial_condition == FLOW_FIRST_WEEK_MEAN else initial_condition
    well_radius = seasonal_coefficients.wvpt.well_radius_m
    radii = np.array([well_radius, 1.03 * well_radius, 1.0, 5.0, 15.0, 60.0])

    table = distance_table(seasonal_coefficients, gappy_index, q_per_well, radii, initial_condition=ic)

    wvpt = seasonal_coefficients.wvpt
    reference = np.column_stack([
        objective(
            [wvpt.alpha, wvpt.beta],
            return_result=True,
            index=gappy_index,
            Q_obs=q_per_well,
            kD=wvpt.kD_model(gappy_index).to_numpy(),
            multiwell=[(1.0, radius / well_radius)],
            multiwell_contains_r_self=True,
            initial_condition=ic,
            integration_method="gauss",
            flow_label="right",
        )
        for radius in radii
    ])
    np.testing.assert_allclose(table, reference, rtol=0.0, atol=5e-5 * np.abs(reference).max())


def test_distance_table_steady_limit(gappy_index):
    # Constant flow and kD with a steady initial condition stay at the De Glee drawdown.
    kd, leakage_resistance = 100.0, 200.0
    coefficients = default_transient_coefficients(kd_ref_m2_per_d=kd, leakage_resistance_d=leakage_resistance)
    q_per_well = np.full(gappy_index.size, 240.0)
    lam = np.sqrt(kd * leakage_resistance)
    radii = np.geomspace(coefficients.wvpt.well_radius_m, 5.0 * lam, 12)

    table = distance_table(coefficients, gappy_index, q_per_well, radii, initial_condition="steady")

    expected = q_per_well[0] * k0(radii / lam) / (2.0 * np.pi * kd)
    np.testing.assert_allclose(table, np.broadcast_to(expected, table.shape), rtol=2e-6)


def _cubic_in_log_r(r, c):
    u = np.log(r)
    return c[0] - c[1] * u + c[2] * u**2 - c[3] * u**3


def test_radial_kernel_equals_direct_spline_per_time_and_shape():
    # A table cubic in ln r is reproduced exactly between its nodes by the not-a-knot spline in
    # ln r, so the kernel product must equal the direct sum over sources and shapes. A different
    # cubic per (time, shape) and n_t != K != m catch any mix-up of the time, shape and source axes.
    rng = np.random.default_rng(0)
    radii = np.geomspace(0.3, 500.0, 15)
    n_times, n_shapes, n_sources = 4, 3, 5
    sources = rng.uniform(-50.0, 50.0, (n_sources, 2))
    strengths = rng.normal(size=(n_sources, n_shapes))
    coef = rng.uniform(0.5, 1.5, (n_times, n_shapes, 4)) * np.array([2.0, 0.4, 0.05, 0.004])
    tables = np.stack([
        np.stack([_cubic_in_log_r(radii, coef[t, k]) for k in range(n_shapes)], axis=-1) for t in range(n_times)
    ])
    gx, gy = np.meshgrid(np.linspace(-90.0, 130.0, 7), np.linspace(-60.0, 80.0, 5))
    points = np.vstack([np.column_stack([gx.ravel(), gy.ravel()]), sources[:1]])

    kernel = radial_kernel(points, sources, strengths, radii, 0.3, max_elements=n_sources * radii.size * 6)
    field = tables.reshape(n_times, -1) @ kernel.reshape(len(points), -1).T

    distance = np.maximum(np.hypot(*(points[:, None, :] - sources[None]).transpose(2, 0, 1)), 0.3)
    expected = np.stack([
        sum(_cubic_in_log_r(distance, coef[t, k]) @ strengths[:, k] for k in range(n_shapes)) for t in range(n_times)
    ])
    np.testing.assert_allclose(field, expected, rtol=0.0, atol=1e-12)


def test_radial_kernel_rejects_distances_beyond_table():
    radii = np.geomspace(0.3, 100.0, 10)
    with pytest.raises(ValueError, match="exceeds the table range"):
        radial_kernel([[150.0, 0.0]], [[0.0, 0.0]], [1.0], radii, 0.3)


# --------------------------------------------------------------------------- #
# Time shapes and bed resistance
# --------------------------------------------------------------------------- #
def test_lagged_histories_step_response_across_gaps():
    # A step from the steady pre-period value y0 to q1 decays exactly as y0 + (q1 - y0)(1 - e^{-t/T}),
    # also across missing times.
    index = pd.date_range("2021-01-01", periods=30, freq="12h").delete([4, 5, 6, 17])
    elapsed = (index - index[0]) / pd.Timedelta("1D")
    time_constants = np.array([0.3, 2.0, 40.0])

    lagged = lagged_histories(index, np.full(index.size, 5.0), time_constants, 2.0)

    expected = 2.0 + 3.0 * (1.0 - np.exp(-np.asarray(elapsed)[:, None] / time_constants))
    np.testing.assert_allclose(lagged, expected, rtol=1e-13)


def test_time_shapes_modulated_copies_and_steady_initial_values():
    index = pd.date_range("2021-01-01", periods=5, freq="D")
    q = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    kd = np.array([150.0, 120.0, 120.0, 120.0, 90.0])  # mean 120
    m = np.array([2.0, 1.0, 1.0, 1.0, 0.0])  # mean 1

    shapes, initial = time_shapes(index, q, 3.5, [1.0, 10.0], [kd, m])

    assert shapes.shape == (5, 9)
    np.testing.assert_array_equal(shapes[:, 0], q)
    np.testing.assert_allclose(shapes[:, 3:6], shapes[:, :3] * (kd / 120.0 - 1.0)[:, None], rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(shapes[:, 6:], shapes[:, :3] * (m - 1.0)[:, None], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(initial, 3.5 * np.array([1, 1, 1, 0.25, 0.25, 0.25, 1, 1, 1]), rtol=1e-15)
    np.testing.assert_array_equal(shapes[0, 1:3], [3.5, 3.5])


def test_time_shapes_skip_constant_modulations(constant_kd_coefficients):
    # A constant kD or bed resistance (fixed head, constant T_bodem) would add all-zero shapes.
    index = pd.date_range("2021-01-01", periods=5, freq="D")
    q = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    kd = np.array([150.0, 120.0, 120.0, 120.0, 90.0])
    constant = [bed_resistance(constant_kd_coefficients, index, np.full(5, 7.0), 0.1), np.zeros(5), np.full(5, 120.0)]

    shapes, initial = time_shapes(index, q, 3.5, [1.0, 10.0], [constant[0], kd, *constant[1:]])

    expected_shapes, expected_initial = time_shapes(index, q, 3.5, [1.0, 10.0], [kd])
    np.testing.assert_array_equal(shapes, expected_shapes)
    np.testing.assert_array_equal(initial, expected_initial)


@pytest.fixture(scope="module")
def constant_kd_coefficients():
    return default_transient_coefficients(kd_ref_m2_per_d=120.0, leakage_resistance_d=57.0)


def test_bed_resistance_is_defined_at_12_degc_whatever_the_sheet_reference():
    # R_bed_12C_d_per_m is the resistance at 12 degC, also when a WVPT sheet uses another reference.
    coefficients = default_transient_coefficients(temperature_ref_degc=10.0)
    index = pd.date_range("2021-01-01", periods=2, freq="D")

    resistance = bed_resistance(coefficients, index, np.array([12.0, 4.0]), 0.1)

    np.testing.assert_allclose(resistance, 0.1 * _visc_ratio(np.array([12.0, 4.0])), rtol=1e-14)


def test_bed_resistance_viscosity_gaps_and_fixed_head(constant_kd_coefficients):
    index = pd.date_range("2021-01-01", periods=7, freq="D")
    temperature = np.array([np.nan, 12.0, np.nan, 4.0, 20.0, np.nan, np.nan])

    with pytest.warns(UserWarning, match="3 leading or trailing"):
        resistance = bed_resistance(constant_kd_coefficients, index, temperature, 0.1)

    # 12 degC is the reference; the interior gap is interpolated in time, the trailing gap held.
    expected_temperature = np.array([12.0, 12.0, 8.0, 4.0, 20.0, 20.0, 20.0])
    np.testing.assert_allclose(resistance, 0.1 * _visc_ratio(expected_temperature), rtol=1e-14)
    assert resistance[0] == pytest.approx(0.1, rel=1e-15)
    np.testing.assert_array_equal(bed_resistance(constant_kd_coefficients, index, temperature, np.nan), 0.0)


@pytest.mark.parametrize("offsets", [[82.0], [82.0, -82.0]], ids=["one canal", "two canals"])
def test_strip_bed_resistance_matches_closed_form(offsets):
    kd, leakage_resistance, flow_per_m, drop = 122.0, 57.0, 17.3, 0.5
    lam = np.sqrt(kd * leakage_resistance)
    far = np.exp(-2 * 82.0 / lam) if len(offsets) == 2 else 0.0

    resistance = strip_bed_resistance(drop, flow_per_m, kd, leakage_resistance, offsets)

    expected = drop * (1 + far) / (flow_per_m * np.exp(-82.0 / lam) - 2 * kd * drop / lam)
    assert resistance == pytest.approx(expected, rel=1e-12)


def test_strip_bed_resistance_unequal_canals_and_unreachable_drop():
    kd, leakage_resistance, flow_per_m = 133.0, 150.0, 12.6
    lam = np.sqrt(kd * leakage_resistance)
    offsets = np.array([82.0, -250.0])
    resistance = strip_bed_resistance(0.5, flow_per_m, kd, leakage_resistance, offsets)

    # Forward 1D strip with that resistance: the near canal carries the 0.5 m drop.
    unit = lam / (2 * kd)
    coupling = unit * np.exp(-np.abs(offsets[:, None] - offsets) / lam) + resistance * np.eye(2)
    infiltration = np.linalg.solve(coupling, flow_per_m * unit * np.exp(-np.abs(offsets) / lam))
    assert resistance * infiltration[0] == pytest.approx(0.5, rel=1e-12)

    reachable = flow_per_m * unit * np.exp(-82.0 / lam)
    assert np.isnan(strip_bed_resistance(1.01 * reachable, flow_per_m, kd, leakage_resistance, offsets))


# --------------------------------------------------------------------------- #
# Steady line-sinks against closed forms
# --------------------------------------------------------------------------- #
KD_S, C_S = 100.0, 25.0
LAM_S = np.sqrt(KD_S * C_S)  # 50 m
Q_WELL = 240.0
B_S = 40.0
ROW_S = _row(101, 10.0)  # 1000 m = 20 leakage factors
RADII_S = np.geomspace(WELL_RADIUS, 3000.0, 40)
MID_S = 500.0


def _steady(shores, resistance, wells=ROW_S, n_times=1, **kwargs):
    shapes = np.full((n_times, 1), Q_WELL)
    tables = _k0_tables(RADII_S, KD_S, LAM_S, shapes)
    solution = solve_transient_linesinks(
        tables, RADII_S, wells, np.asarray(shores), shapes, np.full(n_times, resistance), WELL_RADIUS, **kwargs
    )
    return solution, tables, shapes


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_straight_canal_matches_whole_plane_robin_solution(constant_kd_coefficients, rho):
    # Long row, canal at b with the sink line delta into the water, bed resistance at a constant
    # 4 degC. Mid-row the canal is an infinite line-sink in a whole-plane aquifer:
    #   q' = Q' e^{-b/lam} / (e^{-delta/lam} + 2 R kD / lam)
    # and the field is the row plus a 1D line-sink of that strength. (A half-plane third-type
    # boundary, R kD / lam, differs by 7-25 % here.)
    index = pd.date_range("2021-01-01", periods=2, freq="D")
    r_12 = rho * LAM_S / KD_S
    resistance = bed_resistance(constant_kd_coefficients, index, np.full(2, 4.0), r_12)
    r_4 = r_12 * _visc_ratio(4.0)
    shore = shapely.LineString([(-100.0, B_S), (1100.0, B_S)])

    solution, tables, _ = _steady([shore], resistance[0], n_times=2, max_order=40, order_spacing_m=30.0)

    flow_per_m = Q_WELL / 10.0
    inflow = flow_per_m * np.exp(-B_S / LAM_S) / (np.exp(-SINK_OFFSET / LAM_S) + 2 * r_4 * KD_S / LAM_S)
    mid = np.argmin(np.abs(solution["controls"][:, 0] - MID_S))
    np.testing.assert_allclose(solution["inflow_per_m"][:, mid], inflow, rtol=5e-4)

    points = np.array([[MID_S + 3.7, y] for y in (0.3, 10.0, 25.0, 39.0, 60.0)])
    distance = np.maximum(np.hypot(*(points[:, None, :] - ROW_S[None]).transpose(2, 0, 1)), WELL_RADIUS)
    expected = Q_WELL * k0(distance / LAM_S).sum(axis=1) / (2 * np.pi * KD_S) - inflow * LAM_S / (2 * KD_S) * np.exp(
        -np.abs(points[:, 1] - B_S - SINK_OFFSET) / LAM_S
    )
    field = _field(solution, tables, RADII_S, ROW_S, points)
    np.testing.assert_allclose(
        field, np.broadcast_to(expected, field.shape), rtol=0.0, atol=5e-5 * np.abs(expected).max()
    )


def test_curved_shore_carries_the_bed_loss_between_controls():
    # Robin condition s = R q' also between the control points of a curved shore: a C-shaped canal
    # (240 degrees of a circle, water outside) around a short row.
    wells = _row(21, 10.0)
    angle = np.radians(np.linspace(210.0, -30.0, 241))
    shore = shapely.LineString(np.column_stack([100.0 + 160.0 * np.cos(angle), 160.0 * np.sin(angle)]))
    resistance = 0.2

    solution, tables, shapes = _steady([shore], resistance, wells=wells, max_order=40, order_spacing_m=10.0)

    fractions = np.linspace(0.005, 0.995, 199)
    points = shapely.get_coordinates(shapely.line_interpolate_point(shore, fractions, normalized=True))
    drawdown = _field(solution, tables, RADII_S, wells, points)[0]
    loss = resistance * _inflow_along(solution, shapes, 0, fractions)[0]
    assert loss.min() > 0.25 * loss.max() > 0.0
    np.testing.assert_allclose(drawdown, loss, rtol=0.0, atol=1e-3 * loss.max())


def _strip_images(points, wells, near, far, reflections=4):
    """Exact fixed-head strip between canals at y = near and y = far (opposite signs): image series."""
    width = abs(near - far)
    k = np.arange(-reflections, reflections + 1)
    offsets = np.concatenate([2 * k[k != 0] * width * np.sign(near), 2 * near + 2 * k * width * np.sign(near)])
    signs = np.concatenate([np.ones(2 * reflections), -np.ones(2 * reflections + 1)])
    sources = np.concatenate([wells, *[wells + np.array([0.0, offset]) for offset in offsets]])
    strengths = np.concatenate([np.ones(len(wells)), np.repeat(signs, len(wells))])
    distance = np.maximum(np.hypot(*(points[:, None, :] - sources[None]).transpose(2, 0, 1)), WELL_RADIUS)
    return Q_WELL * k0(distance / LAM_S) @ strengths / (2 * np.pi * KD_S)


def _canal(y, water_above, x0=-100.0, x1=1100.0):
    """Straight shore at y with the water on the left of its direction."""
    return shapely.LineString([(x0, y), (x1, y)] if water_above else [(x1, y), (x0, y)])


def test_fixed_head_strip_matches_reflected_images():
    # Canals on both sides, the near one on the right at 40 m and the far one on the left at 90 m:
    # between them the fixed-head line-sinks must reproduce the image series of a strip, which one
    # image per canal does not.
    sign = -1.0
    near, far = 40.0 * sign, -90.0 * sign
    shores = [_canal(near, sign > 0), _canal(far, sign < 0)]

    solution, tables, _ = _steady(shores, 0.0, max_order=40, order_spacing_m=30.0, sink_offset_m=2.0)

    points = np.array([[MID_S + 3.7, sign * y] for y in (-85.0, -60.0, -30.0, 0.3, 15.0, 35.0)])
    expected = _strip_images(points, ROW_S, near, far)
    field = _field(solution, tables, RADII_S, ROW_S, points, spacing=1.0)[0]
    np.testing.assert_allclose(field, expected, rtol=0.0, atol=1e-4 * expected.max())


def test_round_trip_two_pieces_two_shapes():
    # The map path (point sinks from the coefficients) reproduces the solve at the controls, the
    # node strengths integrate q' along each piece, and the canals take part of the pumping.
    wells = _row(41, 10.0)
    index = pd.date_range("2021-01-01", periods=40, freq="12h")
    q = np.random.default_rng(3).uniform(100.0, 400.0, index.size)
    shapes, _ = time_shapes(index, q, 250.0, [3.0])
    tables = _k0_tables(RADII_S, KD_S, LAM_S, shapes)
    resistance = np.linspace(0.05, 0.3, index.size)
    shores = np.array([_canal(40.0, True, -50.0, 449.9), _canal(-60.0, False, -37.3, 450.0)])  # no whole node counts

    solution = solve_transient_linesinks(tables, RADII_S, wells, shores, shapes, resistance, WELL_RADIUS)

    drawdown = _field(solution, tables, RADII_S, wells, solution["controls"])
    np.testing.assert_allclose(
        drawdown - resistance[:, None] * solution["inflow_per_m"],
        solution["shore_residual_m"],
        rtol=0.0,
        atol=1e-12 * drawdown.max(),
    )

    _, piece, strengths = linesink_sources(solution, SINK_OFFSET / 2.0)
    fractions, weights = legendre.leggauss(30)
    for p, shore in enumerate(shores):
        integral = shore.length * _inflow_along(solution, shapes, p, (fractions + 1) / 2) @ weights / 2
        np.testing.assert_allclose(-shapes @ strengths[piece == p].sum(axis=0), integral, rtol=1e-6)
    total = -shapes @ strengths.sum(axis=0)
    assert np.all(total > 0.0)
    assert np.all(total < q * len(wells))


@pytest.mark.parametrize("reverse", [False, True], ids=["row east", "row west"])
def test_infiltration_by_body_and_side(reverse):
    # The side follows the well numbering: with the row running east the canal at +40 m is left; with
    # the numbering reversed the same canal is right. Each canal is one body and carries its own nodes.
    wells = _row(21, 10.0)[::-1] if reverse else _row(21, 10.0)
    index = pd.date_range("2021-01-01", periods=3, freq="D")
    shapes = np.array([[200.0], [240.0], [300.0]])
    tables = _k0_tables(RADII_S, KD_S, LAM_S, shapes)
    shores = np.array([_canal(40.0, True, -50.0, 250.0), _canal(-60.0, False, -50.0, 250.0)])
    solution = solve_transient_linesinks(tables, RADII_S, wells, shores, shapes, np.full(3, 0.1), WELL_RADIUS)

    infiltration = infiltration_by_body_and_side(solution, shapes, [3, 7], wells, well_row_normals(wells), index, 1.0)

    near, far = ("right", "left") if reverse else ("left", "right")
    assert infiltration.columns.tolist() == sorted([(3, near), (7, far)])
    _, piece, strengths = linesink_sources(solution, 1.0)
    for body, p in ((3, 0), (7, 1)):
        side = near if body == 3 else far
        np.testing.assert_allclose(infiltration[(body, side)], -shapes @ strengths[piece == p].sum(axis=0), rtol=1e-14)
    assert (infiltration[(3, near)] > infiltration[(7, far)]).all()


def test_finite_canal_lies_between_no_canal_and_long_canal():
    # Comparison principle: more fixed-head boundary never raises the drawdown. A canal along
    # half the row must give drawdowns between those without canal and with a canal along all of
    # it, also around its ends, and take in water along its whole length.
    wells = _row(41, 10.0)
    points = np.column_stack([g.ravel() for g in np.meshgrid(np.linspace(-150, 550, 15), np.linspace(-150, 30, 5))])
    shapes = np.full((1, 1), Q_WELL)
    tables = _k0_tables(RADII_S, KD_S, LAM_S, shapes)
    no_canal = tables[0, :, 0] @ radial_kernel(points, wells, np.ones(len(wells)), RADII_S, WELL_RADIUS)[..., 0].T

    orders = {"max_order": 40, "order_spacing_m": 20.0}
    short, _, _ = _steady([_canal(B_S, True, 100.0, 300.0)], 0.0, wells=wells, **orders)
    long, _, _ = _steady([_canal(B_S, True, -100.0, 500.0)], 0.0, wells=wells, **orders)
    short_field = _field(short, tables, RADII_S, wells, points)[0]
    long_field = _field(long, tables, RADII_S, wells, points)[0]

    tolerance = 1e-3 * no_canal.max()
    assert np.all(short_field <= no_canal + tolerance)
    assert np.all(long_field <= short_field + tolerance)
    assert (no_canal - short_field).max() > 0.5
    assert np.all(short["inflow_per_m"] > 0.0)


# --------------------------------------------------------------------------- #
# Transient line-sinks
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def transient_canal():
    """Q300-like single fixed-head canal, 60 days on the steep flank of the seasonal kD, real tables."""
    coefficients = default_transient_coefficients(
        kd_ref_m2_per_d=122.0,
        leakage_resistance_d=57.0,
        temperature_method="sin",
        temperature_mean_degc=12.2,
        temperature_delta_degc=6.7,
        temperature_time_offset_d=174.0,
    )
    index = pd.date_range("2021-06-10", periods=120, freq="12h")
    kd = coefficients.wvpt.kD_model(index).to_numpy()
    lam = np.sqrt(kd.mean() * 57.0)
    wells = _row(int(np.ceil(20 * lam / DX)) + 1, DX)
    q = np.repeat(np.random.default_rng(7).uniform(150.0, 350.0, index.size // 4), 4)
    b, storage = 82.0, coefficients.wvpt.storage_coefficient
    time_constants = np.geomspace(b**2 * storage / (4 * kd.mean()), 3 * 57.0 * storage, 3)
    shapes, initial = time_shapes(index, q, q[:14].mean(), time_constants, [kd])
    radii = np.geomspace(WELL_RADIUS, 2500.0, 25)
    tables = np.stack(
        [
            distance_table(coefficients, index, shape, radii, initial_condition=v)
            for shape, v in zip(shapes.T, initial, strict=False)
        ],
        axis=-1,
    )
    shore = shapely.LineString([(-200.0, b), (wells[-1, 0] + 200.0, b)])
    solution = solve_transient_linesinks(tables, radii, wells, [shore], shapes, np.zeros(index.size), WELL_RADIUS)
    images = np.concatenate([wells, wells + np.array([0.0, 2 * b])])
    image_strengths = np.concatenate([np.ones(len(wells)), -np.ones(len(wells))])
    return {
        "kd": kd,
        "lam": lam,
        "wells": wells,
        "tables": tables,
        "radii": radii,
        "solution": solution,
        "images": images,
        "image_strengths": image_strengths,
    }


def test_transient_fixed_head_canal_matches_image_wells(transient_canal):
    # With a fixed head, one image well per real well is exact for a straight canal also under a
    # seasonal kD. The time shapes (q, three lags, kD-modulated copies) must reproduce it mid-row.
    # Measured: 4 mm at the row, 34 mm 7 m from the canal (0.6 % of the well drawdown) on flow steps
    # every 2 days; this is the time basis (8 lags: 1 and 5 mm).
    c = transient_canal
    x_mid = c["wells"][len(c["wells"]) // 2, 0] + 3.7
    points = np.array([[x_mid, y] for y in (0.3, 20.0, 40.0, 60.0, 75.0)])
    image_kernel = radial_kernel(points, c["images"], c["image_strengths"], c["radii"], WELL_RADIUS)[..., 0]
    expected = c["tables"][..., 0] @ image_kernel.T

    field = _field(c["solution"], c["tables"], c["radii"], c["wells"], points)

    np.testing.assert_allclose(field, expected, rtol=0.0, atol=8e-3 * expected.max())
    np.testing.assert_allclose(field[:, 0], expected[:, 0], rtol=0.0, atol=1e-3 * expected.max())


def test_transient_fixed_head_canal_inflow_28_day_mean(transient_canal):
    # Mid-row inflow of the image solution: q' = kD(t) s(b - eps) / eps (s is antisymmetric about
    # the canal, so this is a central difference), compared as 28-day means. Holding the shore at a
    # fixed head with the sink line delta further into the water takes e^{delta / lambda} more
    # inflow (steady line-sink); measured remainder 0.2 %, 0.8 % without the factor.
    c = transient_canal
    solution = c["solution"]
    mid = np.argmin(np.abs(solution["controls"][:, 0] - c["wells"][len(c["wells"]) // 2, 0]))
    eps = 0.25
    point = solution["controls"][mid] - [0.0, eps]
    image_kernel = radial_kernel(point, c["images"], c["image_strengths"], c["radii"], WELL_RADIUS)[0, :, 0]
    expected = c["kd"] * (c["tables"][..., 0] @ image_kernel) / eps * np.exp(SINK_OFFSET / c["lam"])

    window = np.arange(expected.size) // 56
    np.testing.assert_allclose(
        np.bincount(window, solution["inflow_per_m"][:, mid]) / np.bincount(window),
        np.bincount(window, expected) / np.bincount(window),
        rtol=4e-3,
    )


@pytest.fixture(scope="module")
def seasonal_bed():
    """Two years of daily steps with a gap, a seasonal T_bodem with gaps and a canal at 60 m."""
    coefficients = default_transient_coefficients(kd_ref_m2_per_d=120.0, leakage_resistance_d=57.0)
    index = pd.date_range("2021-01-01", periods=730, freq="D").delete(range(200, 215))
    day = np.asarray(index.dayofyear, dtype=float)
    t_bodem = 11.5 + 6.5 * np.sin(2 * np.pi * (day - 110.0) / 365.25)
    t_bodem[[50, 51, 400]] = np.nan
    t_bodem[-8:] = np.nan
    with pytest.warns(UserWarning, match="8 leading or trailing"):
        resistance = bed_resistance(coefficients, index, t_bodem, 0.12)
    return {
        "index": index,
        "resistance": resistance,
        "lam": np.sqrt(120.0 * 57.0),
        "radii": np.geomspace(WELL_RADIUS, 1500.0, 30),
        "wells": _row(41, 10.0),
        "shore": _canal(60.0, True, -100.0, 500.0),
    }


def test_quasi_steady_inflow_follows_seasonal_bed_resistance(seasonal_bed):
    # Without storage every date is a steady Robin problem with that date's R(t). The inflow varies
    # by 8 % over the season; the resistance-modulated shape copies follow it to second order in
    # R(t) / mean(R) - 1 (measured 0.3 %; without the copies 6 %).
    s = seasonal_bed
    q = np.full(s["index"].size, 240.0)
    kd = np.full(s["index"].size, 120.0)
    shapes, _ = time_shapes(s["index"], q, 240.0, [2.8, 34.0], [kd, s["resistance"]])
    tables = _k0_tables(s["radii"], 120.0, s["lam"], shapes)
    geometry = (s["radii"], s["wells"], [s["shore"]])

    solution = solve_transient_linesinks(tables, *geometry, shapes, s["resistance"], WELL_RADIUS)

    for i in np.unique([np.argmin(s["resistance"]), np.argmax(s["resistance"]), 150, 600]):
        steady = solve_transient_linesinks(
            tables[i : i + 1], *geometry, shapes[i : i + 1], s["resistance"][i : i + 1], WELL_RADIUS
        )
        np.testing.assert_allclose(solution["inflow_per_m"][i], steady["inflow_per_m"][0], rtol=5e-3)
    mid = solution["inflow_per_m"][:, solution["controls"].shape[0] // 2]
    assert mid.max() / mid.min() > 1.05


def test_time_compression_equals_full_least_squares(seasonal_bed):
    s = seasonal_bed
    q = np.random.default_rng(11).uniform(100.0, 400.0, s["index"].size)
    kd = 120.0 * (1.0 + 0.1 * np.sin(np.arange(s["index"].size) / 40.0))
    shapes, _ = time_shapes(s["index"], q, 240.0, [2.8, 34.0], [kd, s["resistance"]])
    tables = _k0_tables(s["radii"], 120.0, s["lam"], shapes) * (kd.mean() / kd)[:, None, None]
    args = (tables, s["radii"], s["wells"], [s["shore"]], shapes, s["resistance"], WELL_RADIUS)

    compressed = solve_transient_linesinks(*args)
    full = solve_transient_linesinks(*args, svd_rtol=None, max_elements=20_000)  # many QR blocks

    np.testing.assert_allclose(
        compressed["inflow_per_m"], full["inflow_per_m"], rtol=0.0, atol=1e-10 * np.abs(full["inflow_per_m"]).max()
    )


# --------------------------------------------------------------------------- #
# Geometry (in a rotated frame at RD coordinates)
# --------------------------------------------------------------------------- #
ORIGIN = np.array([102_000.0, 505_000.0])
ROTATION = np.array([[np.cos(0.65), -np.sin(0.65)], [np.sin(0.65), np.cos(0.65)]])
XY = _row(41, 15.0)  # 0..600 m at y = 0
CANAL = shapely.box(-50, 80, 650, 90)


def _shores(polygons, search_m=1000.0, *, prepare=False):
    """facing_shores in the rotated RD frame with 5 m edges; pieces back in the local frame."""
    in_rd = [shapely.transform(p, lambda c: c @ ROTATION.T + ORIGIN) for p in polygons]
    water = prepare_water(in_rd) if prepare else shapely.union_all(in_rd)
    shores, bodies = facing_shores(XY @ ROTATION.T + ORIGIN, water, search_m, edge_m=5.0)
    # The sink line lies in the water: the water is on the left of every piece.
    sinks = shapely.offset_curve(shores, SINK_OFFSET)
    np.testing.assert_allclose(shapely.length(shapely.intersection(sinks, water)), shapely.length(sinks), rtol=1e-4)
    return [shapely.transform(s, lambda c: (c - ORIGIN) @ ROTATION) for s in shores], bodies


def _on_line(shore, y):
    return np.allclose(shapely.get_coordinates(shore)[:, 1], y, atol=1e-6)


def test_facing_shores_keeps_the_near_bank_only():
    shores, _ = _shores([CANAL])
    assert len(shores) == 1
    assert _on_line(shores[0], 80.0)
    assert shores[0].length == pytest.approx(700.0, abs=1e-6)


def test_facing_shores_drops_banks_behind_the_own_water_body():
    # Near and far canal joined at one end: one body after the union, whose far arm faces the
    # wells but lies behind the near arm.
    shores, bodies = _shores([CANAL, shapely.box(-50, 150, 650, 160), shapely.box(640, 90, 650, 150)])
    assert len(shores) == 1
    assert _on_line(shores[0], 80.0)
    assert shores[0].length == pytest.approx(700.0, abs=1e-6)
    assert bodies.tolist() == [0]


def test_facing_shores_partial_shadow_has_the_seen_length():
    # A strip over x < 300 in front of the canal: a bank point x_e is seen (from any well) iff the
    # x = 600 well sees it: 0.375 * 600 + 0.625 * x_e > 300 -> x_e > 120. Kept [120, 650] = 530 m,
    # within one edge (midpoint test).
    shores, bodies = _shores([CANAL, shapely.box(-200, 30, 300, 50)])
    canal = [s for s in shores if _on_line(s, 80.0)]
    assert len(canal) == 1
    assert abs(canal[0].length - 530.0) <= 5.0
    assert len(set(bodies.tolist())) == 2


def test_facing_shores_bank_seen_from_other_wells_is_kept():
    # A pond in front of a few wells: the bank behind it is still seen from the other wells.
    shores, _ = _shores([CANAL, shapely.box(280, 30, 320, 50)])
    canal = [s for s in shores if _on_line(s, 80.0)]
    assert len(canal) == 1
    assert canal[0].length == pytest.approx(700.0, abs=1e-6)


def test_facing_shores_joins_a_run_across_the_ring_start():
    # The ring starts at a 3 m kink inside the near bank.
    ring = [(300, 77), (650, 80), (650, 90), (-50, 90), (-50, 80), (300, 77)]
    shores, _ = _shores([shapely.Polygon(ring)])
    assert len(shores) == 1
    assert shores[0].length == pytest.approx(2 * np.hypot(350.0, 3.0), abs=1e-6)


def test_facing_shores_row_inside_a_hole():
    # Water all around the row: the whole inner ring is one closed piece.
    water = shapely.difference(shapely.box(-100, -90, 700, 90), shapely.box(-50, -80, 650, 80))
    shores, _ = _shores([water])
    assert len(shores) == 1
    assert shores[0].is_closed
    assert shores[0].length == pytest.approx(2 * 700 + 2 * 160, abs=1e-6)


def test_facing_shores_clips_a_canal_at_the_search_distance():
    # The canal runs 3 km past the row end; bank edges count within 200 m of a well.
    shores, _ = _shores([shapely.box(-50, 80, 3600, 90)], 200.0)
    reach = 600.0 + np.sqrt(200.0**2 - 80.0**2)
    assert len(shores) == 1
    assert abs(shores[0].length - (reach + 50.0)) <= 5.0


def test_prepare_water_closes_a_bridge_gap():
    # A 5 m culvert gap splits the canal into two polygons; closed, it is one body and one shore.
    shores, bodies = _shores([shapely.box(-50, 80, 300, 90), shapely.box(305, 80, 650, 90)], prepare=True)
    assert len(shores) == 1
    assert bodies.tolist() == [0]
    assert shores[0].length == pytest.approx(700.0, abs=1.0)  # the closing cuts the corners by a few decimeters


def test_prepare_water_drops_slivers_without_room_for_a_sink_line():
    # A 0.3 m wide ditch facing the wells cannot hold a sink line 0.5 m into the water; the opening
    # removes it (the in-water check of _shores fails on a kept sliver) and keeps the canal.
    shores, _ = _shores([CANAL, shapely.box(100, -60.3, 500, -60)], prepare=True)
    assert len(shores) == 1
    assert shapely.get_coordinates(shores[0])[:, 1].min() > 79.0  # the canal bank, corners rounded by 0.5 m
    assert shores[0].length == pytest.approx(700.0, abs=1.0)


def test_split_shores_cuts_pieces_by_their_distance_to_the_wells():
    # Sub-pieces start where the previous one ends, are at most clip(scale * d, 20, 160) m long with d
    # the distance from their start to the nearest well (the last may absorb a short remainder), and
    # keep the length and direction of their parent.
    shores = np.array([shapely.LineString([(-900, 60), (1500, 60)]), shapely.LineString([(650, -30), (650, -300)])])

    pieces, parent = split_shores(shores, XY, scale=0.8)

    assert parent.tolist() == sorted(parent.tolist())
    assert set(parent.tolist()) == {0, 1}
    for p, shore in enumerate(shores):
        own = pieces[parent == p]
        coords = [shapely.get_coordinates(piece) for piece in own]
        np.testing.assert_allclose([c[0] for c in coords[1:]], [c[-1] for c in coords[:-1]], atol=1e-9)
        np.testing.assert_allclose(coords[0][0], shapely.get_coordinates(shore)[0], atol=1e-9)
        assert shapely.length(own).sum() == pytest.approx(shore.length, abs=1e-6)
        start = shapely.points([c[0] for c in coords])
        limit = np.clip(0.8 * shapely.distance(start, shapely.multipoints(XY)), 20.0, 160.0)
        assert np.all(shapely.length(own)[:-1] <= limit[:-1] + 1e-9)
        assert shapely.length(own)[-1] <= limit[-1] + 10.0
    assert shapely.length(pieces).min() >= 10.0
    assert shapely.length(pieces[parent == 0]).min() < 60.0  # near the wells
    assert shapely.length(pieces[parent == 0]).max() > 150.0  # far from the wells


# --------------------------------------------------------------------------- #
# Well rows
# --------------------------------------------------------------------------- #
def test_well_row_normals_on_bent_row():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 10.0], [20.0, 20.0]])
    normals = well_row_normals(xy)
    half_sqrt2 = np.sqrt(0.5)
    expected = np.array([[0.0, 1.0], [0.0, 1.0], [-half_sqrt2, half_sqrt2], [-1.0, 0.0], [-1.0, 0.0]])
    np.testing.assert_allclose(normals, expected, rtol=0.0, atol=1e-15)


def test_row_segments_orders_by_well_number_and_splits_gaps():
    xy = np.array([[30.0, 0.0], [0.0, 0.0], [300.0, 0.0], [15.0, 0.0], [315.0, 0.0]])
    segments = row_segments([2, 0, 3, 1, 4], xy)
    assert [segment.tolist() for segment in segments] == [[1, 3, 0], [2, 4]]


def test_row_segments_follows_well_number_around_a_hairpin():
    # Position order (x or principal axis) differs from well-number order on a hairpin row.
    xy = np.array([[15.0, 15.0], [0.0, 0.0], [30.0, 0.0], [0.0, 15.0], [15.0, 0.0], [30.0, 15.0]])
    assert [segment.tolist() for segment in row_segments([4, 0, 2, 5, 1, 3], xy)] == [[1, 4, 2, 5, 0, 3]]
