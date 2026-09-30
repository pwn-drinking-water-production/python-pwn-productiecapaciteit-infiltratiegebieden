"""Tests for the 2D head map in report_wvpweerstand_transient."""

import numpy as np
import pandas as pd
import pytest
import shapely
from scipy.special import k0

from productiecapaciteit.reports.report_wvpweerstand_transient import (
    canal_mask,
    canal_sides,
    default_transient_coefficients,
    distance_table,
    drawdown_field,
    facing_shores,
    image_offsets,
    image_wells,
    prepare_water,
    row_segments,
    well_row_normals,
)
from productiecapaciteit.src.wvp_transient_funs import objective

NPUT = 21
DX = 15.0
FLOW_FIRST_WEEK_MEAN = "first_week_mean"
SINGLE_CANAL = [(-1.0, 75.0, "left")]
TWO_CANALS = [(-1.0, 82.0, "left"), (-1.0, 82.0, "right")]
SINK_OFFSET = 0.5


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


def _initial_condition(name, q_per_well):
    return float(q_per_well[:14].mean()) if name == FLOW_FIRST_WEEK_MEAN else name


def _straight_row_wells():
    return np.column_stack([np.arange(NPUT) * DX, np.zeros(NPUT)])


def _synthetic_k0_table(radii, lam=150.0):
    return 2.0 * k0(np.asarray(radii) / lam)[None, :]


@pytest.mark.parametrize("initial_condition", ["zero", FLOW_FIRST_WEEK_MEAN])
def test_distance_table_matches_gauss_near_and_far(seasonal_coefficients, gappy_index, flow_m3h, initial_condition):
    # The point-source kd_grid path is off by decimeters within meters of the well under
    # seasonal kD and step-changing flow; each column must carry its own near window.
    q_per_well = flow_m3h / NPUT * 24.0
    ic = _initial_condition(initial_condition, q_per_well)
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


@pytest.mark.parametrize("initial_condition", ["zero", FLOW_FIRST_WEEK_MEAN])
@pytest.mark.parametrize("r_mirrorwel", [SINGLE_CANAL, TWO_CANALS])
def test_field_matches_crosssection_exactly(
    seasonal_coefficients, gappy_index, flow_m3h, r_mirrorwel, initial_condition
):
    # A straight row reproduces the tested 1D cross-section: same images, strengths, sides,
    # per-well flow and flow labelling. The table sits on the exact distances, so the spline
    # is evaluated at its knots. What remains is the adaptive quad of gauss's latest interval
    # (epsrel 1e-8), run per term here but on the summed kernel in the cross-section: ~1e-11 m,
    # whereas a geometry error (image at b or 4b, sign, side) is >= 0.7 m.
    q_per_well = flow_m3h / NPUT * 24.0
    ic = _initial_condition(initial_condition, q_per_well)
    well_radius = seasonal_coefficients.wvpt.well_radius_m
    xy = _straight_row_wells()
    sources, strengths = image_wells(xy, well_row_normals(xy), image_offsets(r_mirrorwel))
    distances = np.array([0.0, 10.0, 30.0, 60.0])
    gx = np.full(distances.size, xy[NPUT // 2, 0])
    gy = distances

    radii = np.unique(np.maximum(np.hypot(gx[:, None] - sources[:, 0], gy[:, None] - sources[:, 1]), well_radius))
    table = distance_table(
        seasonal_coefficients, gappy_index, q_per_well, radii, initial_condition=ic, integration_method="gauss"
    )
    field = drawdown_field(table, radii, sources, strengths, gx, gy, well_radius)

    crosssection = seasonal_coefficients.wvpt.dp_model_crosssection(
        gappy_index,
        flow_m3h,
        NPUT,
        DX,
        r_mirrorwel,
        distances,
        initial_condition=ic,
        integration_method="gauss",
        flow_label="right",
    )
    np.testing.assert_allclose(field, -crosssection.to_numpy(), rtol=0.0, atol=1e-9)


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


def test_single_canal_line_has_zero_drawdown():
    # Rotated straight row at RD-sized coordinates: every point on the canal line, also beyond
    # the row ends, is equidistant to each well and its image.
    angle = np.deg2rad(37.0)
    direction = np.array([np.cos(angle), np.sin(angle)])
    left = np.array([-np.sin(angle), np.cos(angle)])
    origin = np.array([102_000.0, 505_000.0])
    xy = origin + np.arange(NPUT)[:, None] * DX * direction
    sources, strengths = image_wells(xy, well_row_normals(xy), image_offsets(SINGLE_CANAL, n_reflections=5))
    along = np.linspace(-200.0, NPUT * DX + 200.0, 50)
    line = origin + along[:, None] * direction + 75.0 * left
    radii = np.geomspace(0.3, 2_000.0, 60)

    field = drawdown_field(_synthetic_k0_table(radii), radii, sources, strengths, line[:, 0], line[:, 1], 0.3)

    np.testing.assert_allclose(field, 0.0, rtol=0.0, atol=1e-9)


def test_two_sided_canal_line_equals_far_images():
    # Single reflection: on the +b canal the wells cancel against their +2b images; the -2b
    # images remain, so the canal is not at zero drawdown.
    xy = _straight_row_wells()
    sources, strengths = image_wells(xy, well_row_normals(xy), image_offsets(TWO_CANALS))
    line_x = np.linspace(-150.0, NPUT * DX + 150.0, 40)
    line_y = np.full(line_x.size, 82.0)
    radii = np.geomspace(0.3, 2_000.0, 60)
    table = _synthetic_k0_table(radii)

    field = drawdown_field(table, radii, sources, strengths, line_x, line_y, 0.3)

    far_images = xy - 2.0 * 82.0 * np.array([0.0, 1.0])
    expected = -drawdown_field(table, radii, far_images, np.ones(NPUT), line_x, line_y, 0.3)
    np.testing.assert_allclose(field, expected, rtol=0.0, atol=1e-9)


ASYMMETRIC_STRIP = [(-1.0, 82.0, "left"), (-1.0, 250.0, "right")]


def test_image_offsets_strip_series_matches_closed_form():
    # Strip between y = a (left) and y = -c (right), width L = a + c, both constant head: the
    # images are +1 at 2kL (k != 0) and -1 at 2a + 2kL. Two chains of four reflections give
    # k = +-1, +-2 for the positive images and 2a, 2a +- 2L, 2a - 4L for the negative ones.
    a, c = 82.0, 250.0
    width = a + c
    offsets = image_offsets(ASYMMETRIC_STRIP, n_reflections=4)
    positive = sorted(offset for weight, offset in offsets if weight > 0)
    negative = sorted(offset for weight, offset in offsets if weight < 0)
    np.testing.assert_allclose(positive, [-4 * width, -2 * width, 2 * width, 4 * width], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(negative, [2 * a - 4 * width, 2 * a - 2 * width, 2 * a, 2 * a + 2 * width], atol=1e-12)
    assert {abs(weight) for weight, _ in offsets} == {1.0}


def test_image_offsets_single_canal_ignores_reflections():
    assert image_offsets(SINGLE_CANAL, n_reflections=6) == [(-1.0, 150.0)]


@pytest.mark.parametrize(("n_reflections", "exact"), [(1, False), (12, True)])
def test_strip_reflections_hold_both_canals_at_zero_drawdown(n_reflections, exact):
    # Repeated reflections make both canals of an asymmetric strip exact constant-head lines;
    # the single image pair leaves a clear residual on them.
    xy = _straight_row_wells()
    sources, strengths = image_wells(xy, well_row_normals(xy), image_offsets(ASYMMETRIC_STRIP, n_reflections))
    line_x = np.tile(np.linspace(-150.0, NPUT * DX + 150.0, 30), 2)
    line_y = np.repeat([82.0, -250.0], 30)
    radii = np.geomspace(0.3, 20_000.0, 80)

    field = drawdown_field(_synthetic_k0_table(radii), radii, sources, strengths, line_x, line_y, 0.3)

    assert (np.abs(field).max() <= 1e-9) == exact
    assert exact or np.abs(field).max() > 1e-3


def test_drawdown_field_chunks_match_one_pass():
    xy = _straight_row_wells()
    sources, strengths = image_wells(xy, well_row_normals(xy), image_offsets(ASYMMETRIC_STRIP, n_reflections=3))
    gx, gy = np.meshgrid(np.linspace(-100.0, 400.0, 11), np.linspace(-300.0, 120.0, 9))
    radii = np.geomspace(0.3, 5_000.0, 60)
    table = np.vstack([_synthetic_k0_table(radii), 0.5 * _synthetic_k0_table(radii)])

    one_pass = drawdown_field(table, radii, sources, strengths, gx, gy, 0.3)
    chunked = drawdown_field(table, radii, sources, strengths, gx, gy, 0.3, max_elements=sources.shape[0] * 7)

    # Same values; only the BLAS summation order of the source sum differs per block size.
    np.testing.assert_allclose(chunked, one_pass, rtol=1e-13, atol=1e-13)


def _cubic_in_log_r(r):
    u = np.log(r)
    return 2.0 - 0.4 * u + 0.05 * u**2 - 0.004 * u**3


def test_field_interpolates_in_log_distance():
    # A response cubic in ln r is reproduced exactly between the table nodes by the not-a-knot
    # spline in ln r; interpolating linearly, or in r, is not.
    radii = np.geomspace(0.3, 500.0, 15)
    table = _cubic_in_log_r(radii)[None, :]
    sources = np.array([[0.0, 0.0], [40.0, 10.0]])
    strengths = np.array([1.0, -1.0])
    gx, gy = np.meshgrid(np.linspace(-90.0, 130.0, 7), np.linspace(-60.0, 80.0, 5))

    field = drawdown_field(table, radii, sources, strengths, gx, gy, 0.3)

    distance = np.maximum(np.hypot(gx[..., None] - sources[:, 0], gy[..., None] - sources[:, 1]), 0.3)
    expected = _cubic_in_log_r(distance) @ strengths
    np.testing.assert_allclose(field[0], expected, rtol=0.0, atol=1e-12)


def test_field_rejects_distances_beyond_table():
    radii = np.geomspace(0.3, 100.0, 10)
    with pytest.raises(ValueError, match="exceeds the table range"):
        drawdown_field(_synthetic_k0_table(radii), radii, [[0.0, 0.0]], [1.0], [150.0], [0.0], 0.3)


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


def test_canal_sides_follow_each_canals_side_in_order():
    # Q200/P200 layout: two single canals on opposite sides at different distances.
    normals = well_row_normals(_straight_row_wells())
    canals = canal_sides(normals, [(-1.0, 250.0, "right"), (-1.0, 82.0, "left")])
    assert [(canal[0], canal[1]) for canal in canals] == [(-1.0, 250.0), (-1.0, 82.0)]
    np.testing.assert_array_equal(canals[0][2], -normals)
    np.testing.assert_array_equal(canals[1][2], normals)


def test_bent_row_images_and_mask_use_each_wells_normal():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 10.0], [20.0, 20.0]])
    h = np.sqrt(0.5)
    left = np.array([[0.0, 1.0], [0.0, 1.0], [-h, h], [-1.0, 0.0], [-1.0, 0.0]])
    canals = canal_sides(well_row_normals(xy), [(-1.0, 3.0, "left")])

    sources, _ = image_wells(xy, well_row_normals(xy), image_offsets([(-1.0, 3.0, "left")]))
    mask = canal_mask(np.array([17.0 + 1e-6, 17.0 - 1e-6]), np.array([20.0, 20.0]), xy, canals)

    np.testing.assert_allclose(sources[5:], xy + 6.0 * left, rtol=0.0, atol=1e-13)
    assert mask.tolist() == [False, True]


@pytest.mark.parametrize(("r_mirrorwel", "masked_side"), [(SINGLE_CANAL, [True, False]), (TWO_CANALS, [True, True])])
def test_canal_mask_at_canal_distance(r_mirrorwel, masked_side):
    xy = _straight_row_wells()
    canals = canal_sides(well_row_normals(xy), r_mirrorwel)
    boundary = r_mirrorwel[0][1]
    eps = 1e-6
    gx = np.full(4, xy[NPUT // 2, 0])
    gy = np.array([boundary - eps, boundary + eps, -(boundary - eps), -(boundary + eps)])

    mask = canal_mask(gx, gy, xy, canals)

    assert mask.tolist() == [False, masked_side[0], False, masked_side[1]]


def _row(n, dx, y=0.0):
    return np.column_stack([np.arange(n) * dx, np.full(n, y)])


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
