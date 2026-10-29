"""Independent algebra, quadrature and physical limits for the causal solver."""

# ruff: noqa: D103, INP001, PLR2004

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import shapely
from scipy.integrate import quad
from scipy.special import k0

from productiecapaciteit.src.wvp_robin import (
    boundary_geometry,
    interval_mean,
    solve_boundary,
    solve_common_head,
    solve_transient,
)
from productiecapaciteit.src.wvp_transient_funs import IntervalResponse, hantush_variable_kd


def test_hand_derived_asymmetric_algebra():
    state = solve_common_head([[2, 0.5], [0.5, 1]], [30, 30], 12, 0.5)
    np.testing.assert_allclose(state.well_flow, [4, 8], rtol=2e-15)
    np.testing.assert_allclose(state.common_head, 16, rtol=2e-15)
    np.testing.assert_allclose(state.hard_residual, 0, atol=2e-14)


@pytest.mark.parametrize("resistance", [0, 0.5, 1e6])
def test_resistance_limits_with_prescribed_flow(resistance):
    a = np.array([[2, 0.5], [0.5, 1]])
    state = solve_common_head(a, [0, 0], 12, resistance)
    expected_q1 = 12 * (0.5 + resistance) / (2 + 2 * resistance)
    np.testing.assert_allclose(state.well_flow, [expected_q1, 12 - expected_q1], rtol=1e-10)
    if resistance == 0:
        np.testing.assert_allclose(a @ state.well_flow, [10.5, 10.5], atol=2e-14)


def test_one_well_units_and_signed_crossflow():
    state = solve_common_head([[2]], [30], 24, 24 / 24)
    np.testing.assert_allclose(state.common_head, -42, atol=1e-13)
    state = solve_common_head(np.eye(2), [4, 0], 0, 1)
    np.testing.assert_allclose(state.well_flow, [1, -1], atol=2e-15)
    np.testing.assert_allclose(state.common_head, 2, atol=2e-15)


def test_two_shore_constraints_and_weighted_spatial_objective():
    # Independent KKT oracle, used only in the test. Five controls, two shore modes.
    kw = np.array([[2, 0.5, -0.3, -0.1], [0.5, 1, -0.2, -0.4]])
    kc = np.array([[0.5, 0.2, -1, 0], [0.1, 0.4, 0, -1], [0.3, 0.3, -0.5, -0.5]])
    p = np.array([[1, 0], [0, 1], [0.5, 0.5]])
    weights = np.array([1, 3, 2])
    history_c = np.array([0.1, -0.3, 0.5])
    state = solve_boundary(kw, kc, p, weights, [0.2, -0.4], history_c, [10, 11], 0.5, 0.2, 12)
    c = np.vstack([
        np.column_stack([kw + np.column_stack([0.5 * np.eye(2), np.zeros((2, 2))]), np.ones(2)]),
        [1, 1, 0, 0, 0],
    ])
    s = np.column_stack([kc - np.column_stack([np.zeros((3, 2)), 0.2 * p]), np.zeros(3)])
    d = np.array([9.8, 11.4, 12])
    kkt = np.block([[s.T @ np.diag(weights) @ s, c.T], [c, np.zeros((3, 3))]])
    expected = np.linalg.solve(kkt, np.r_[-s.T @ (weights * history_c), d])[:5]
    np.testing.assert_allclose(np.r_[state.well_flow, state.shore_amplitude, state.common_head], expected, atol=5e-14)
    np.testing.assert_allclose(state.hard_residual, 0, atol=3e-14)
    changed = solve_boundary(kw, kc, p, weights, [0, 0], [0, 0, 0], [10, 10], 4, 0.2, 12)
    assert np.linalg.norm(changed.shore_amplitude - state.shore_amplitude) > 0.1


def test_rank_deficiency_is_not_an_equal_flow_fallback():
    with pytest.raises(np.linalg.LinAlgError, match="rank"):
        solve_common_head(np.zeros((2, 2)), [0, 0], 2, 0)


@pytest.fixture
def response():
    time = np.array([0, 0.1, 0.7, 1.4, 3])
    index = pd.Timestamp("2020") + pd.to_timedelta(time, unit="D")
    return IntervalResponse(index, [70, 71, 77, 84, 100], [0.3, 3, 30], 0.2, 100)


def test_interval_histories_match_independent_direct_quadrature(response):
    rates = np.array([[2, 4], [9, -2], [0, 3], [5, 1]])
    # kD(t)=70+10t: the integral is independently known.
    for i, t in enumerate(response.days[1:], 1):
        actual = response.interval_weights(t, np.arange(i)) @ rates[:i]
        expected = np.zeros((3, 2))
        for j in range(i):
            for r, radius in enumerate(response.radii):

                def kernel(tau, t=t, radius=radius):
                    dk = 70 * (t - tau) + 5 * (t * t - tau * tau)
                    return np.exp(-(radius**2) * 0.2 / (4 * dk) - (t - tau) / 20) / (4 * np.pi * dk)

                expected[r] += quad(kernel, response.days[j], response.days[j + 1], epsabs=1e-12)[0] * rates[j]
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-11)


def test_interval_superposition_matches_existing_quad_and_initial_field(response):
    q = np.array([7, 3, -2, 4, 0])
    for r, radius in enumerate(response.radii):
        expected = hantush_variable_kd(
            radius * np.sqrt(0.2 / 4),
            np.sqrt(1 / 20),
            [70, 71, 77, 84, 100],
            index=response.index,
            Q_obs=q,
            initial_condition=7,
            flow_label="right",
            integration_method="quad",
        )
        actual = response.initial_response()[:, r] * 7
        for i in range(1, len(q)):
            actual[i] += response.interval_weights(response.days[i], np.arange(i))[r] @ q[1 : i + 1]
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-10)


def test_interval_mean_is_not_endpoint():
    # C=L=q=1, interval length C/L=1: analytic reservoir solution.
    average = interval_mean(lambda t: 1 - np.exp(-t), np.array([0, 1]), order=12)
    np.testing.assert_allclose(average, np.exp(-1), rtol=3e-15)
    assert abs(average[0] - (1 - np.exp(-1))) > 0.2


def make_model(periods=9):
    index = pd.date_range("2020", periods=periods, freq="12h")
    response = IntervalResponse(index, 100, np.geomspace(0.3, 200, 45), 0.2, 100)
    wells = np.array([[0, 0], [15, 0]])
    shores = np.array([shapely.LineString([[-30, 20], [40, 20]]), shapely.LineString([[40, -30], [-30, -30]])])
    geometry = boundary_geometry(wells, shores, response.radii, 0.3, max_order=2, sink_offset_m=1)
    return response, geometry


def test_zero_start_causality_and_independent_well_histories():
    response, geometry = make_model()
    q = np.full(8, 240.0)
    first = solve_transient(response, geometry, q, 0.001, 0.1, 5, initial="zero")
    q[-2:] = 0
    second = solve_transient(response, geometry, q, 0.001, 0.1, 5, initial="zero")
    np.testing.assert_allclose(first.well_flow[:6], second.well_flow[:6], atol=0, rtol=0)
    np.testing.assert_allclose(second.well_flow.sum(axis=1), q, atol=1e-10)
    assert np.all(second.well_flow[-2:].min(axis=1) < 0)
    assert np.all(second.well_flow[-2:].max(axis=1) > 0)
    assert np.max(np.abs(first.well_flow[:, 0] - first.well_flow[:, 1])) > 0.01
    assert np.max(np.abs(first.hard_residual)) < 1e-10
    assert np.isfinite(first.shore_residual).all()


def test_steady_start_constant_coefficients_preserves_solved_field():
    response, _ = make_model()
    geometry = boundary_geometry([[0, 0], [15, 0], [45, 0]], [], response.radii, 0.3)
    solution = solve_transient(response, geometry, np.full(8, 240), 0.001, 0.1, 5, initial="steady", initial_flow=240)
    np.testing.assert_allclose(solution.well_flow, np.tile(solution.initial_rates[:3], (8, 1)), atol=3e-8)
    np.testing.assert_allclose(solution.common_head, solution.initial_common_head, atol=1e-9)
    assert np.ptp(solution.initial_rates) > 0.1


def test_no_shore_steady_leakage_conservation():
    index = pd.date_range("2020", periods=3, freq="D")
    radii = np.geomspace(0.3, 100, 30)
    response = IntervalResponse(index, 100, radii, 0.2, 100)
    geometry = boundary_geometry([[0, 0]], [], radii, 0.3)
    solution = solve_transient(response, geometry, [24, 24], 0.001, 0, 0, initial="steady", initial_flow=24)
    expected = 24 * k0(0.3 / 100) / (2 * np.pi * 100)
    np.testing.assert_allclose(solution.well_drawdown, expected, rtol=1e-10)
    # Integral of leakage s/c over the infinite plane: int_0^inf x K0(x) dx = 1.
    integrated = quad(lambda x: x * k0(x), 0, np.inf, epsabs=1e-11)[0] * 24
    np.testing.assert_allclose(integrated, solution.well_flow[:, 0], rtol=1e-11)


def test_accelerated_history_against_direct_with_variable_kd_and_signed_rates():
    index = pd.date_range("2020", periods=90, freq="6h")
    response = IntervalResponse(index, 100 + 20 * np.sin(np.arange(90) / 17), np.geomspace(0.3, 200, 20), 0.2, 100)
    rates = np.random.default_rng(42).normal(size=(89, 3)) * 100
    history = response.history(3, rtol=1e-8, atol=1e-10)
    for j in range(89):
        history.append(j, rates[j])
    actual = history.evaluate(response.days[-1])
    assert history.accepted_blocks > 0
    expected = response.interval_weights(response.days[-1], np.arange(89)) @ rates
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=3e-10)


def test_restart_equivalence_variable_kd_and_cleaning():
    response, geometry = make_model()
    response = IntervalResponse(response.index, np.linspace(80, 120, len(response.index)), response.radii, 0.2, 100)
    rf = np.r_[np.full(4, 0.003), np.full(4, 0.001)]
    full = solve_transient(response, geometry, np.arange(8) * 10 + 200, rf, 0.1, 5)
    partial = solve_transient(response, geometry, np.arange(8) * 10 + 200, rf, 0.1, 5, stop_after=4)
    restarted = solve_transient(response, geometry, np.arange(8) * 10 + 200, rf, 0.1, 5, resume=partial)
    np.testing.assert_allclose(restarted.rates, full.rates, rtol=5e-14, atol=5e-14)
    np.testing.assert_allclose(restarted.common_head, full.common_head, rtol=5e-14, atol=5e-14)


def test_fixed_head_and_finite_bed_limits_independent_dense_collocation():
    # One well and one shore degree: exact independent elimination a=kcw*q/(kcs+Rbed).
    for rbed in (0.0, 0.2, 1e6):
        state = solve_boundary([[2, -0.5]], [[0.5, -1.0]], [[1]], [3], [0], [0], [10], 0.1, rbed, 12)
        amplitude = 6 / (1 + rbed)
        np.testing.assert_allclose(state.shore_amplitude, amplitude, rtol=3e-15, atol=2e-14)
        np.testing.assert_allclose(state.common_head, 10 - 2.1 * 12 + 0.5 * amplitude, atol=2e-14)
        np.testing.assert_allclose(state.shore_residual, 0, atol=2e-14)


def test_cleaning_changes_algebra_not_instantaneous_aquifer_field():
    # At a discontinuity the parabolic aquifer field is continuous, while the
    # algebraic partition may jump. This oracle supplies that same free field.
    before = solve_common_head(np.zeros((2, 2)), [4, 0], 12, 1)
    after = solve_common_head(np.zeros((2, 2)), [4, 0], 12, 0.5)
    np.testing.assert_allclose(before.well_flow, [8, 4], atol=2e-14)
    np.testing.assert_allclose(after.well_flow, [10, 2], atol=2e-14)
    np.testing.assert_allclose(before.common_head, -4, atol=2e-14)
    np.testing.assert_allclose(after.common_head, -1, atol=2e-14)


def test_straight_fixed_head_canal_converges_to_independent_image_solution():
    points = np.array([[0, -4], [3, -2], [12, -6], [-8, -15]])
    actual_distance = np.maximum(np.linalg.norm(points - [0, -4], axis=1), 0.3)
    image_distance = np.linalg.norm(points - [0, 4], axis=1)
    expected = 24 / (2 * np.pi * 100) * (k0(actual_distance / 10) - k0(image_distance / 10))
    response = IntervalResponse(pd.date_range("2020", periods=3, freq="12h"), 100, np.geomspace(0.3, 1000, 120), 0.2, 1)
    errors = []
    for segment in (20.0, 10.0, 5.0):
        x = np.arange(-100, 100 + segment / 2, segment)
        shores = np.array([shapely.LineString([[a, 0], [b, 0]]) for a, b in pairwise(x)])
        geometry = boundary_geometry(
            [[0, -4]], shores, response.radii, 0.3, max_order=3, order_spacing_m=segment / 3, sink_offset_m=1
        )
        solution = solve_transient(response, geometry, 24, 0.002, 0, 0, initial="steady", initial_flow=24)
        errors.append(np.max(np.abs(solution.drawdown(points, [0])[0] - expected)))
    # Finite 200 m shore is ten leakage lengths each side; source discretization
    # dominates the coarse error. Require convergence, not exact image equality.
    assert errors[-1] < 2e-6
    assert errors[-1] < errors[0] / 20
