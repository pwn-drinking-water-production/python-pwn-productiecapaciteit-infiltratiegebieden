from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.special import exp1, k0, k1

import productiecapaciteit.src.wvp_transient_funs as funs
from productiecapaciteit.src.wvp_transient_funs import (
    _kd_grid_regime,
    build_multiwell_geometry,
    hantush_variable_kd,
    objective,
)


def _variable_kd_pextra(index, q_obs):
    return {
        "index": index,
        "Q_obs": q_obs,
        "dt_lower": pd.Timedelta("1D"),
        "frac_step_max": 0.999999,
        "initial_condition": "zero",
    }


@pytest.fixture
def hantush_case():
    radius = 12.0
    storage = 0.2
    leakage_resistance = 150.0
    kD = 100.0
    q = 2400.0
    index = pd.date_range("2020-01-01", periods=80, freq="1D")
    return {
        "radius": radius,
        "storage": storage,
        "leakage_resistance": leakage_resistance,
        "kD": kD,
        "q": q,
        "index": index,
        "alpha": (radius**2 * storage / 4.0) ** 0.5,
        "beta": (1.0 / (leakage_resistance * storage)) ** 0.5,
        "q_obs": np.full(index.size, q),
    }


def test_constant_transmissivity_matches_pastas_hantush_well_model(hantush_case):
    HantushWellModel = pytest.importorskip("pastas.rfunc").HantushWellModel
    actual = hantush_variable_kd(
        hantush_case["alpha"],
        hantush_case["beta"],
        np.full(hantush_case["index"].size, hantush_case["kD"]),
        **_variable_kd_pextra(hantush_case["index"], hantush_case["q_obs"]),
    )

    a = hantush_case["leakage_resistance"] * hantush_case["storage"]
    b = np.log(1.0 / (4.0 * hantush_case["leakage_resistance"] * hantush_case["kD"]))
    gain = hantush_case["q"] / (2.0 * np.pi * hantush_case["kD"])
    elapsed_days = np.arange(hantush_case["index"].size, dtype=float)
    expected = np.zeros(hantush_case["index"].size)
    expected[1:] = HantushWellModel.numpy_step(
        gain,
        a,
        b,
        hantush_case["radius"],
        elapsed_days[1:],
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-10)


def test_scalar_and_constant_array_transmissivity_are_identical(hantush_case):
    pextra = _variable_kd_pextra(hantush_case["index"], hantush_case["q_obs"])
    scalar_kd = hantush_variable_kd(
        hantush_case["alpha"],
        hantush_case["beta"],
        hantush_case["kD"],
        **pextra,
    )
    array_kd = hantush_variable_kd(
        hantush_case["alpha"],
        hantush_case["beta"],
        np.full(hantush_case["index"].size, hantush_case["kD"]),
        **pextra,
    )

    np.testing.assert_allclose(scalar_kd, array_kd, rtol=0.0, atol=0.0)


def test_objective_accepts_supplied_transmissivity(hantush_case):
    pextra = {
        **_variable_kd_pextra(hantush_case["index"], hantush_case["q_obs"]),
        "drawdown_obs": np.zeros(hantush_case["index"].size),
        "kD": np.full(hantush_case["index"].size, hantush_case["kD"]),
        "multiwell": [(1.0, 1.0)],
        "multiwell_contains_r_self": True,
    }

    actual = objective(
        [hantush_case["alpha"], hantush_case["beta"]],
        return_result=True,
        **pextra,
    )
    expected = hantush_variable_kd(
        hantush_case["alpha"],
        hantush_case["beta"],
        hantush_case["kD"],
        **_variable_kd_pextra(hantush_case["index"], hantush_case["q_obs"]),
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_variable_kd_defaults_to_fast_gauss_settings(hantush_case):
    q_obs = hantush_case["q_obs"].copy()
    q_obs[20:40] *= 0.6
    q_obs[40:] *= 1.25
    kD = np.full(hantush_case["index"].size, hantush_case["kD"])
    kD[30:] *= 1.4
    pextra = _variable_kd_pextra(hantush_case["index"], q_obs)

    actual = hantush_variable_kd(
        hantush_case["alpha"],
        hantush_case["beta"],
        kD,
        **pextra,
    )
    expected = hantush_variable_kd(
        hantush_case["alpha"],
        hantush_case["beta"],
        kD,
        **pextra,
        integration_method="gauss",
        n_gauss=32,
        max_gauss_step_days=0.5,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_variable_kd_quad_matches_linear_kd_reference_with_irregular_steps():
    time_days = np.array([0.0, 0.35, 1.4, 3.2, 6.0])
    index = pd.Timestamp("2020-01-01") + pd.to_timedelta(time_days, unit="D")
    alpha = 1.3
    beta = 0.08
    kd0 = 90.0
    kd_slope = 7.5
    kD = kd0 + kd_slope * time_days
    q_obs = np.array([900.0, 1400.0, 650.0, 1750.0, 1200.0])

    actual = hantush_variable_kd(
        alpha,
        beta,
        kD,
        index=index,
        Q_obs=q_obs,
        initial_condition="zero",
        integration_method="quad",
        quad_epsabs=1e-11,
        quad_epsrel=1e-10,
    )

    def cumulative_kd(t):
        return kd0 * t + 0.5 * kd_slope * t * t

    expected = np.zeros_like(time_days)
    for target_idx in range(1, time_days.size):
        for source_idx in range(target_idx):

            def integrand(source_time):
                d_k = cumulative_kd(time_days[target_idx]) - cumulative_kd(source_time)
                if d_k <= 0.0:
                    return 0.0
                lag = time_days[target_idx] - source_time
                return q_obs[source_idx] * np.exp(-alpha * alpha / d_k - beta * beta * lag) / (4.0 * np.pi * d_k)

            expected[target_idx] += quad(
                integrand,
                time_days[source_idx],
                time_days[source_idx + 1],
                epsabs=1e-11,
                epsrel=1e-10,
                limit=100,
            )[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_variable_kd_steady_initial_condition_matches_steady_limit(hantush_case):
    expected = (
        hantush_case["q"]
        / (4.0 * np.pi * hantush_case["kD"])
        * 2.0
        * k0(2.0 * hantush_case["alpha"] * hantush_case["beta"] / np.sqrt(hantush_case["kD"]))
    )

    actual = hantush_variable_kd(
        hantush_case["alpha"],
        hantush_case["beta"],
        hantush_case["kD"],
        index=hantush_case["index"][:12],
        Q_obs=hantush_case["q_obs"][:12],
        initial_condition="steady",
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def _leaky_tail_quad(alpha2, leakage_rate, lower, *, moment=False, shift=0.0):
    """Reference ``int_lower^inf exp(-alpha^2/w - c (w - shift)) / w^(1 - moment) dw`` by quadrature.

    Doubling breakpoints from ``lower`` past the peak (``w ~ alpha^2`` or ``alpha / sqrt(c)``)
    and the decay, so no piece spans more than a factor two of the ``1/w``-like integrand and a
    distant peak cannot be missed. The neglected tail beyond ``60 / c`` is below ``exp(-60)``.
    """
    span_end = lower + 100.0 * alpha2 + 10.0 * np.sqrt(alpha2 / leakage_rate) + 60.0 / leakage_rate
    edges = np.geomspace(lower, span_end, max(2, int(np.ceil(np.log2(span_end / lower))) + 1))

    def integrand(w):
        value = np.exp(-alpha2 / w - leakage_rate * (w - shift))
        return value if moment else value / w

    return sum(quad(integrand, lo, hi, epsabs=1e-16, epsrel=1e-12, limit=200)[0] for lo, hi in pairwise(edges))


@pytest.mark.parametrize("leakage_resistance", [200.0, 5000.0])
def test_variable_kd_initial_condition_resolves_distant_image_well(leakage_resistance):
    # A distant (image) well's pre-period drawdown integrand exp(-alpha^2/k - b (k - K_i)) / k
    # peaks near k = alpha^2, far above the lower limit K_i. A single adaptive quad over
    # [K_i, inf) can miss that peak; the reference splits it at doubling breakpoints.
    index = pd.date_range("2020-01-01", periods=60, freq="12h")
    time_days = np.asarray((index - index[0]) / pd.Timedelta("1D"), dtype=float)
    kD = 100.0 * (1.0 + 0.2 * np.sin(2.0 * np.pi * time_days / 365.0))
    storage = 0.2
    alpha = (800.0**2 * storage / 4.0) ** 0.5  # image well 800 m away
    beta = (1.0 / (leakage_resistance * storage)) ** 0.5
    initial_q = 1000.0

    actual = hantush_variable_kd(alpha, beta, kD, index=index, Q_obs=np.zeros(index.size), initial_condition=initial_q)

    cumulative_kd = PchipInterpolator(time_days, kD).antiderivative()(time_days)
    b = beta * beta / kD[0]
    expected = np.empty(index.size)
    expected[0] = initial_q / (4.0 * np.pi * kD[0]) * 2.0 * k0(2.0 * alpha * beta / np.sqrt(kD[0]))
    for i in range(1, index.size):
        integral = _leaky_tail_quad(alpha * alpha, b, cumulative_kd[i], shift=cumulative_kd[i])
        expected[i] = initial_q * np.exp(-beta * beta * time_days[i]) / (4.0 * np.pi * kD[0]) * integral

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=0.0)


def test_variable_kd_response_to_kd_step_is_not_instantaneous(hantush_case):
    index = hantush_case["index"][:60]
    q_obs = hantush_case["q_obs"][:60]
    step_idx = 30
    kD = np.full(index.size, hantush_case["kD"])
    kD[step_idx:] = 2.0 * hantush_case["kD"]

    actual = hantush_variable_kd(
        hantush_case["alpha"],
        hantush_case["beta"],
        kD,
        index=index,
        Q_obs=q_obs,
        initial_condition="steady",
    )

    rho_new = 2.0 * hantush_case["alpha"] * hantush_case["beta"] / np.sqrt(kD[step_idx])
    new_steady = q_obs[step_idx] / (4.0 * np.pi * kD[step_idx]) * 2.0 * k0(rho_new)

    assert actual[step_idx] > new_steady
    assert actual[-1] < actual[step_idx]


def test_variable_kd_right_labelled_flow_matches_shifted_left_label(hantush_case):
    index = hantush_case["index"][:10]
    q_interval = np.array([500.0, 1200.0, 800.0, 1600.0, 300.0, 900.0, 700.0, 1400.0, 600.0])
    q_left = np.r_[q_interval, 999.0]
    q_right = np.r_[111.0, q_interval]

    left = hantush_variable_kd(
        hantush_case["alpha"],
        hantush_case["beta"],
        hantush_case["kD"],
        index=index,
        Q_obs=q_left,
        initial_condition="zero",
        flow_label="left",
    )
    right = hantush_variable_kd(
        hantush_case["alpha"],
        hantush_case["beta"],
        hantush_case["kD"],
        index=index,
        Q_obs=q_right,
        initial_condition="zero",
        flow_label="right",
    )

    np.testing.assert_allclose(right, left, rtol=0.0, atol=0.0)


def test_objective_uses_variable_kd_hantush(hantush_case):
    index = hantush_case["index"][:24]
    q_obs = hantush_case["q_obs"][:24]
    kD = np.full(index.size, hantush_case["kD"])
    kD[12:] = 1.8 * hantush_case["kD"]
    pextra = {
        **_variable_kd_pextra(index, q_obs),
        "drawdown_obs": np.zeros(index.size),
        "kD": kD,
        "multiwell": [(1.0, 1.0)],
        "multiwell_contains_r_self": True,
        "initial_condition": "steady",
    }

    actual = objective(
        [hantush_case["alpha"], hantush_case["beta"]],
        return_result=True,
        **pextra,
    )
    expected_pextra = {
        **_variable_kd_pextra(index, q_obs),
        "initial_condition": "steady",
    }
    expected = hantush_variable_kd(
        hantush_case["alpha"],
        hantush_case["beta"],
        kD,
        **expected_pextra,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_variable_kd_rejects_invalid_flow_label(hantush_case):
    with pytest.raises(ValueError, match="flow_label"):
        hantush_variable_kd(
            hantush_case["alpha"],
            hantush_case["beta"],
            hantush_case["kD"],
            index=hantush_case["index"],
            Q_obs=hantush_case["q_obs"],
            initial_condition="zero",
            flow_label="center",
        )


@pytest.mark.parametrize("n_gauss", [0, 1.9, True])
def test_variable_kd_rejects_invalid_n_gauss(hantush_case, n_gauss):
    with pytest.raises(ValueError, match="n_gauss"):
        hantush_variable_kd(
            hantush_case["alpha"],
            hantush_case["beta"],
            hantush_case["kD"],
            index=hantush_case["index"],
            Q_obs=hantush_case["q_obs"],
            initial_condition="zero",
            n_gauss=n_gauss,
        )


def test_variable_kd_rejects_tmax_days_cap(hantush_case):
    with pytest.raises(NotImplementedError, match="tmax_days_cap"):
        hantush_variable_kd(
            hantush_case["alpha"],
            hantush_case["beta"],
            hantush_case["kD"],
            index=hantush_case["index"],
            Q_obs=hantush_case["q_obs"],
            initial_condition="zero",
            tmax_days_cap=10.0,
        )


def _variable_kd_synthetic_case(periods=240, freq_hours=12):
    """Variable kD (seasonal viscosity-like) and variable flow on a realistic grid."""
    index = pd.date_range("2018-01-01", periods=periods, freq=f"{freq_hours}h")
    n = np.arange(periods)
    days = n * (freq_hours / 24.0)
    kD = 100.0 + 20.0 * np.sin(2.0 * np.pi * days / 365.0) + 0.01 * days
    rng = np.random.default_rng(7)
    q_obs = 2400.0 + 200.0 * np.sin(2.0 * np.pi * days / 90.0) + rng.normal(0.0, 120.0, periods)
    return index, kD, q_obs


def test_kd_grid_matches_quad_rate_part_variable_kd_and_flow():
    index, kD, q_obs = _variable_kd_synthetic_case(periods=120)
    radius, storage, leakage = 0.2, 0.2, 200.0
    alpha = (radius**2 * storage / 4.0) ** 0.5
    beta = (1.0 / (leakage * storage)) ** 0.5

    fast = hantush_variable_kd(
        alpha,
        beta,
        kD,
        index=index,
        Q_obs=q_obs,
        initial_condition="zero",
        integration_method="kd_grid",
    )
    reference = hantush_variable_kd(
        alpha,
        beta,
        kD,
        index=index,
        Q_obs=q_obs,
        initial_condition="zero",
        integration_method="quad",
        quad_epsabs=1e-12,
        quad_epsrel=1e-11,
    )
    np.testing.assert_allclose(fast, reference, rtol=3e-3, atol=1e-6)


def test_kd_grid_refines_toward_quad_with_resolution():
    index, kD, q_obs = _variable_kd_synthetic_case(periods=120)
    alpha = (0.2**2 * 0.2 / 4.0) ** 0.5
    beta = (1.0 / (200.0 * 0.2)) ** 0.5
    reference = hantush_variable_kd(
        alpha,
        beta,
        kD,
        index=index,
        Q_obs=q_obs,
        initial_condition="zero",
        integration_method="quad",
        quad_epsabs=1e-12,
        quad_epsrel=1e-11,
    )

    def max_rel(n_per_step):
        fast = hantush_variable_kd(
            alpha,
            beta,
            kD,
            index=index,
            Q_obs=q_obs,
            initial_condition="zero",
            integration_method="kd_grid",
            n_per_step=n_per_step,
        )
        return np.max(np.abs(fast - reference) / np.maximum(np.abs(reference), 1e-9))

    assert max_rel(16) < max_rel(4)


@pytest.mark.parametrize("leakage_resistance", [25.0, 200.0, 2000.0, 1.0e5])
def test_kd_grid_matches_quad_across_leakage(leakage_resistance):
    index, kD, q_obs = _variable_kd_synthetic_case(periods=120)
    storage = 0.2
    alpha = (0.2**2 * storage / 4.0) ** 0.5
    beta = (1.0 / (leakage_resistance * storage)) ** 0.5

    fast = hantush_variable_kd(
        alpha,
        beta,
        kD,
        index=index,
        Q_obs=q_obs,
        initial_condition="zero",
        integration_method="kd_grid",
    )
    reference = hantush_variable_kd(
        alpha,
        beta,
        kD,
        index=index,
        Q_obs=q_obs,
        initial_condition="zero",
        integration_method="quad",
        quad_epsabs=1e-12,
        quad_epsrel=1e-11,
    )
    assert np.isfinite(fast).all()
    np.testing.assert_allclose(fast, reference, rtol=5e-3, atol=1e-6)


def test_kd_grid_blocking_stays_finite_and_accurate_over_long_span():
    # A large beta^2 * span would overflow a single exp(beta^2 t) factoring; the
    # blocked path must stay finite. Low leakage (leaky aquifer) drives beta up so
    # blocking is exercised on a modest grid (keeping the gauss reference cheap).
    index, kD, q_obs = _variable_kd_synthetic_case(periods=400)
    storage = 0.2
    leakage = 50.0
    alpha = (0.2**2 * storage / 4.0) ** 0.5
    beta = (1.0 / (leakage * storage)) ** 0.5
    span_days = (index[-1] - index[0]) / pd.Timedelta("1D")
    assert beta**2 * span_days > 17.0  # blocking is exercised

    fast = hantush_variable_kd(
        alpha,
        beta,
        kD,
        index=index,
        Q_obs=q_obs,
        initial_condition="zero",
        integration_method="kd_grid",
    )
    reference = hantush_variable_kd(
        alpha,
        beta,
        kD,
        index=index,
        Q_obs=q_obs,
        initial_condition="zero",
        integration_method="gauss",
    )
    assert np.isfinite(fast).all()
    np.testing.assert_allclose(fast, reference, rtol=5e-3, atol=1e-5)


def test_kd_grid_matches_pastas_hantush_well_model_constant_kd(hantush_case):
    HantushWellModel = pytest.importorskip("pastas.rfunc").HantushWellModel
    fast = hantush_variable_kd(
        hantush_case["alpha"],
        hantush_case["beta"],
        np.full(hantush_case["index"].size, hantush_case["kD"]),
        **_variable_kd_pextra(hantush_case["index"], hantush_case["q_obs"]),
        integration_method="kd_grid",
    )
    a = hantush_case["leakage_resistance"] * hantush_case["storage"]
    b = np.log(1.0 / (4.0 * hantush_case["leakage_resistance"] * hantush_case["kD"]))
    gain = hantush_case["q"] / (2.0 * np.pi * hantush_case["kD"])
    elapsed_days = np.arange(hantush_case["index"].size, dtype=float)
    expected = np.zeros(hantush_case["index"].size)
    expected[1:] = HantushWellModel.numpy_step(gain, a, b, hantush_case["radius"], elapsed_days[1:])
    np.testing.assert_allclose(fast, expected, rtol=5e-3, atol=1e-6)


@pytest.mark.parametrize(
    ("leakage_resistance", "n_per_step", "rtol"),
    [(5.0, 16, 1e-3), (3.0, 16, 1e-3), (8.0, 8, 5e-3)],
)
def test_kd_grid_very_leaky_matches_quad(leakage_resistance, n_per_step, rtol):
    # Very leaky aquifers drive beta high: the leakage memory is a few cells and kd_grid
    # runs many short blocks. The near window integrates the leakage decay exactly, so the
    # error is set by the far grid alone; this asserts the blocked path runs (via the public
    # regime classifier) AND bounds its error against quad.
    index, kD, q_obs = _variable_kd_synthetic_case(periods=120)
    storage = 0.2
    alpha = (0.2**2 * storage / 4.0) ** 0.5
    beta = (1.0 / (leakage_resistance * storage)) ** 0.5

    time_days = np.asarray((index - index[0]) / pd.Timedelta("1D"), dtype=float)
    cumulative_kd = PchipInterpolator(time_days, kD).antiderivative()(time_days)
    cumulative_kd -= cumulative_kd[0]
    regime, _dk, _n_grid, _mem_cells = _kd_grid_regime(beta, time_days, cumulative_kd, n_per_step)
    assert regime == "blocked"

    fast = hantush_variable_kd(
        alpha,
        beta,
        kD,
        index=index,
        Q_obs=q_obs,
        initial_condition="zero",
        integration_method="kd_grid",
        n_per_step=n_per_step,
    )
    reference = hantush_variable_kd(
        alpha,
        beta,
        kD,
        index=index,
        Q_obs=q_obs,
        initial_condition="zero",
        integration_method="quad",
        # The kd_grid error bounded here is >= 1e-3, so a 1e-9 reference is ample
        # and far cheaper than 1e-11.
        quad_epsabs=1e-10,
        quad_epsrel=1e-9,
    )
    assert np.isfinite(fast).all()
    np.testing.assert_allclose(fast, reference, rtol=rtol, atol=1e-6)


@pytest.mark.parametrize(("leakage", "expected_regime"), [(200.0, "long_memory"), (3.0, "blocked")])
def test_kd_grid_finite_radius_near_window_is_target_well_only(monkeypatch, leakage, expected_regime):
    # Only the well whose head is of interest carries a finite well radius: its term sits
    # at alpha^2 = r_well^2 * S / 4 and gets the exact near-window integral. Every other
    # well in the series and every mirror well is an infinitely small point source that
    # rides the far convolution only. Restricting the (expensive) near window to the target
    # is the whole point of the optimisation, but it is invisible in the output -- neighbour
    # terms are grid-resolved, so the near window and the far kernel agree on them to ~1e-4
    # (see test_dp_model_multiwell_kd_grid_matches_gauss). So this pins the *mechanism*: the
    # finite-radius near-window integrator is only ever evaluated with the target alpha^2.
    # Re-widening the near window to neighbours (the prior alpha^2 <= w_near rule) would make
    # neighbour alpha^2 values reach the integrator and fail this test.
    seen_alpha2 = []
    original = funs._kd_antiderivative_well_function

    def spy(kappa, alpha2, leakage_rate):
        seen_alpha2.append(float(alpha2))
        return original(kappa, alpha2, leakage_rate)

    monkeypatch.setattr(funs, "_kd_antiderivative_well_function", spy)

    storage, well_radius = 0.2, 0.2
    alpha = (well_radius**2 * storage / 4.0) ** 0.5
    beta = (1.0 / (leakage * storage)) ** 0.5
    target_alpha2 = alpha * alpha

    # A close neighbour at dx=15 m (eff alpha^2 = 11.25) plus an image well: under the old
    # alpha^2 <= w_near rule both would have been pulled into the near window.
    multiwell, _counts = build_multiwell_geometry(
        15.0, [(-1.0, 50.0)], 3, target_well_index=1, distance_scale=1.0 / well_radius
    )
    neighbour_alpha2 = sorted({(distance * alpha) ** 2 for _multi, distance in multiwell})
    assert neighbour_alpha2[1] > target_alpha2 * 100.0  # the nearest neighbour is far above the target

    index, kD, q_obs = _variable_kd_synthetic_case(periods=120)
    time_days = np.asarray((index - index[0]) / pd.Timedelta("1D"), dtype=float)
    cumulative_kd = PchipInterpolator(time_days, kD).antiderivative()(time_days)
    cumulative_kd -= cumulative_kd[0]
    regime, *_ = _kd_grid_regime(beta, time_days, cumulative_kd, 8)
    assert regime == expected_regime  # also holds for very leaky aquifers

    objective(
        [alpha, beta],
        return_result=True,
        index=index,
        Q_obs=q_obs,
        kD=kD,
        dt_lower=pd.Timedelta("1D"),
        multiwell=multiwell,
        multiwell_contains_r_self=True,
        initial_condition="zero",
        integration_method="kd_grid",
    )

    assert seen_alpha2, "the finite-radius near-window integrator was never exercised"
    assert max(seen_alpha2) <= target_alpha2 * (1.0 + 1e-9)


@pytest.mark.parametrize(
    ("alpha2", "leakage_rate"),
    [(a, c) for a in (4.5e-3, 11.25, 4.5e3) for c in (1e-5, 1e-3, 0.0455) if 4.0 * a * c < funs._HANTUSH_RHO_MAX**2],
)
def test_kd_antiderivative_well_function_matches_quadrature(alpha2, leakage_rate):
    # The exact leaky segment integrals of the near window: the tails F = (1/4pi) int_kappa^inf
    # exp(-alpha^2/w - c w)/w dw (the Hantush well function W(c kappa, 2 alpha sqrt(c))) and
    # G = (1/4pi) int_kappa^inf exp(-alpha^2/w - c w) dw (its first moment). Checked against
    # adaptive quadrature over both series branches (u >= x directly, u < x through the
    # reflections) and the kappa = 0 limits 2 K0(rho) and rho K1(rho) / c.
    kappa = np.array([0.0, 1e-6, 1e-3, 0.1, 1.0, 6.9, 55.0, 400.0, 2400.0])
    well, moment = funs._kd_antiderivative_well_function(kappa, alpha2, leakage_rate)
    rho = 2.0 * np.sqrt(leakage_rate * alpha2)
    expected_well = np.empty_like(kappa)
    expected_moment = np.empty_like(kappa)
    for i, k in enumerate(kappa):
        if k == 0.0:
            expected_well[i] = 2.0 * k0(rho)
            expected_moment[i] = rho * k1(rho) / leakage_rate
            continue
        expected_well[i] = _leaky_tail_quad(alpha2, leakage_rate, k)
        expected_moment[i] = _leaky_tail_quad(alpha2, leakage_rate, k, moment=True)
    np.testing.assert_allclose(well * 4.0 * np.pi, expected_well, rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(moment * 4.0 * np.pi, expected_moment, rtol=1e-9, atol=1e-13)


def test_kd_antiderivative_well_function_drops_negligible_terms():
    # rho = 2 sqrt(c alpha^2) above _HANTUSH_RHO_MAX: the whole well function 2 K0(rho) is below
    # the accuracy of the near-well terms, so the term is dropped (zero F and G).
    alpha2, leakage_rate = 4.5e3, 0.0455
    rho = 2.0 * np.sqrt(leakage_rate * alpha2)
    assert rho >= funs._HANTUSH_RHO_MAX
    assert 2.0 * k0(rho) < 1e-11
    well, moment = funs._kd_antiderivative_well_function(np.array([0.0, 1.0, 400.0]), alpha2, leakage_rate)
    np.testing.assert_array_equal(well, 0.0)
    np.testing.assert_array_equal(moment, 0.0)


def test_kd_grid_point_source_kernel_matches_direct_e1():
    # The far kernel sums E1(alpha^2 / w) over ~50 wells through a moment expansion
    # (terms far from their peak) plus direct E1 near each peak; must equal the plain sum.
    radius = 0.3
    alpha = (radius**2 * 0.2 / 4.0) ** 0.5
    multiwell, _ = build_multiwell_geometry(15.0, [(-2.0, 120.0), (1.0, 400.0)], 30, distance_scale=1.0 / radius)
    terms = np.asarray(multiwell, dtype=float)
    mults, alpha2s = terms[:, 0], (terms[:, 1] * alpha) ** 2
    w = np.arange(20001) * 6.9
    actual = funs._kd_grid_point_source_kernel(mults, alpha2s, w)
    with np.errstate(divide="ignore"):
        expected = exp1(alpha2s[None, :] / w[:, None]) @ mults
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)


def test_kd_grid_leakage_factoring_is_not_first_order():
    # Regression: factoring the leakage decay exp(-beta^2 lag) at cell midpoints made the
    # near window first-order in dk. The diagonal cell of the target well weights w ~ 1/w, so
    # its leakage centroid sits far from the midpoint: at c = 1 d, n_per_step = 8 this gave an
    # 8% error against quad (4% at n_per_step = 16). The exact Hantush-W segment integral
    # removes it: the single-well error is then set by the near window's sub-step
    # linearisation alone (~1e-5) at any n_per_step.
    index, kD, q_obs = _variable_kd_synthetic_case(periods=120)
    storage, leakage = 0.2, 1.0
    alpha = (0.2**2 * storage / 4.0) ** 0.5
    beta = (1.0 / (leakage * storage)) ** 0.5
    reference = hantush_variable_kd(
        alpha,
        beta,
        kD,
        index=index,
        Q_obs=q_obs,
        initial_condition="zero",
        integration_method="quad",
        quad_epsabs=1e-10,
        quad_epsrel=1e-9,
    )

    def max_rel(n_per_step):
        fast = hantush_variable_kd(
            alpha,
            beta,
            kD,
            index=index,
            Q_obs=q_obs,
            initial_condition="zero",
            integration_method="kd_grid",
            n_per_step=n_per_step,
        )
        return np.max(np.abs(fast - reference)) / np.max(np.abs(reference))

    assert max_rel(8) < 1e-4
    assert max_rel(16) < 1e-4
