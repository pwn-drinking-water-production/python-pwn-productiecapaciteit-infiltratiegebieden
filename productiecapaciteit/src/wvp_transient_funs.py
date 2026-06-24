from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.special import k0, lambertw
from scipy import linalg


def get_temp(index, mean, delta, time_offset, return_series=False):
    index_datetime = pd.DatetimeIndex(index)
    year = pd.Categorical(index_datetime.year, ordered=True)
    start_year = year.rename_categories(pd.to_datetime(year.categories, format="%Y"))
    end_year = year.rename_categories(
        pd.to_datetime(year.categories.astype(str) + "1231", format="%Y%m%d")
    )
    nday_year = end_year.map(lambda x: x.dayofyear, na_action="ignore").astype(float)
    dt_year = index_datetime - start_year.to_numpy()
    temp_data = (
        delta
        * np.sin((dt_year / pd.Timedelta("1D") - time_offset) * 2 * np.pi / nday_year)
        + mean
    )
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
    dx_mirrorwell,
    nput,
    *,
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
    dx_mirrorwell : iterable
        Image-well specs from the config as ``(multiplicity, canal_distance_m)``.
        Negative multiplicities represent opposite-sign image wells for
        constant-head canal boundaries.
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
    if dx_put <= 0.0:
        raise ValueError(f"dx_put must be positive, got {dx_put}")

    nput_float = float(nput)
    nput_int = int(round(nput_float))
    if nput_int < 1 or not np.isclose(nput_float, nput_int):
        raise ValueError(f"nput must be a positive integer, got {nput}")

    if target_well_index is None:
        target_well_index = nput_int // 2
    target_well_index = int(target_well_index)
    if target_well_index < 0 or target_well_index >= nput_int:
        raise ValueError(
            f"target_well_index must be in [0, {nput_int - 1}], got {target_well_index}"
        )

    distance_scale = float(distance_scale)
    if distance_scale <= 0.0:
        raise ValueError(f"distance_scale must be positive, got {distance_scale}")

    neighbor_counts = defaultdict(int)
    for well_index in range(nput_int):
        if well_index == target_well_index:
            continue
        row_distance = abs(well_index - target_well_index) * dx_put
        neighbor_counts[row_distance] += 1
    neighbor_items = sorted(neighbor_counts.items())

    if dx_mirrorwell is None:
        image_specs = []
    else:
        image_arr = np.asarray(dx_mirrorwell, dtype=float)
        if image_arr.size == 0:
            image_specs = []
        else:
            image_arr = np.atleast_2d(image_arr)
            if image_arr.shape[1] != 2:
                raise ValueError(
                    "dx_mirrorwell must contain (multiplicity, canal_distance_m) pairs"
                )
            image_specs = [(float(multi), float(distance)) for multi, distance in image_arr]

    multiwell = []
    if include_self:
        multiwell.append((1.0, float(self_distance)))

    for row_distance, count in neighbor_items:
        multiwell.append((float(count), row_distance * distance_scale))

    for image_multi, canal_distance in image_specs:
        if canal_distance <= 0.0:
            raise ValueError(f"Mirror-well canal distance must be positive, got {canal_distance}")
        image_distance = 2.0 * canal_distance
        if include_self:
            multiwell.append((image_multi, image_distance * distance_scale))
        for row_distance, count in neighbor_items:
            multiwell.append(
                (
                    count * image_multi,
                    dis(row_distance, image_distance) * distance_scale,
                )
            )

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


def objective(args, return_result=False, **pextra):
    """
    multialpha =

    Parameters
    ----------
    args
    return_result

    Returns
    -------

    """
    if len(args) < 5:
        raise ValueError("objective expects at least five parameters")

    alpha, beta, kD0, temp_delta, temp_time_offset = args[:5]
    arg_idx = 5
    s = f"{alpha}, {beta}, {kD0}, {temp_delta}, {temp_time_offset}"
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")
    if kD0 <= 0.0:
        raise ValueError(f"kD0 must be positive, got {kD0}")
    if not np.isfinite([temp_delta, temp_time_offset]).all():
        raise ValueError("Temperature model parameters must be finite")

    multiwell_contains_r_self = pextra.get("multiwell_contains_r_self", False)
    alpha_multi = None
    if pextra.get("multiwell") and not multiwell_contains_r_self:
        if "alpha_multi" in pextra:
            alpha_multi = pextra["alpha_multi"]
        else:
            if len(args) <= arg_idx:
                raise ValueError(
                    "objective needs alpha_multi when multiwell distances are not "
                    "normalized by the self-well radius"
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

    temp = get_temp(
        pextra["index"],
        pextra["temp_ref"],
        temp_delta,
        temp_time_offset,
        return_series=False,
    )
    kD = kD0 / visc_ratio(temp, temp_ref=pextra["temp_ref"])

    def multihantush(*args):
        multi, alpha, beta, kD = args
        return multi * hantush(alpha, beta, kD, **pextra)

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
        print(f"objective parameters: {s}")
        print(
            "multiwell setup: "
            f"{total_wells} wells "
            f"(self={self_wells}, neighboring={neighbor_wells}; "
            f"neighbor terms={counts.get('neighbor_well_terms', 0)}), "
            f"{total_mirrorwells} mirror wells "
            f"(self mirrors={self_mirrorwells}, neighbor mirrors={neighbor_mirrorwells}; "
            f"mirror terms={counts.get('self_mirrorwell_terms', 0) + counts.get('neighbor_mirrorwell_terms', 0)}), "
            f"{len(hantush_args)} Hantush evaluations"
        )

    if len(hantush_args) == 1:
        results = [multihantush(*hantush_args[0])]
    else:
        max_workers = pextra.get("max_workers", None)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda args: multihantush(*args), hantush_args))
    drawdown_model = np.stack(results).sum(axis=0)

    if return_result:
        return drawdown_model

    drawdown_obs = np.asarray(pextra["drawdown_obs"], dtype=float)
    if drawdown_obs.shape != drawdown_model.shape:
        raise ValueError(
            f"drawdown_obs shape {drawdown_obs.shape} does not match model shape "
            f"{drawdown_model.shape}"
        )
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


# @background
def hantush(alpha, beta, kD, **pextra):
    """Compute the drawdown for a single well using the Hantush well function.

    alpha = (r**2 * S / 4) ** 0.5
    beta = (1 / (c * S)) ** 0.5

    Parameters
    ----------
    index
    Q
    alpha
    beta
    kD
    nt_max

    Returns
    -------

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

    initial_condition = pextra.get("initial_condition", "steady")
    if initial_condition == "steady":
        initial_q = q_obs[0]
    elif initial_condition == "zero":
        initial_q = 0.0
    else:
        initial_q = float(initial_condition)
        if not np.isfinite(initial_q):
            raise ValueError("initial_condition must be 'steady', 'zero', or a finite number")

    dQ = np.diff(q_obs, prepend=initial_q)
    rho = 2.0 * alpha * beta / np.sqrt(kD)
    cS = 1.0 / np.square(beta)
    h_inf = 2.0 * k0(rho)

    frac_step_max = float(pextra.get("frac_step_max", 0.95))
    if not 0.0 < frac_step_max < 1.0:
        raise ValueError(f"frac_step_max must be between 0 and 1, got {frac_step_max}")

    k0_rho = k0(rho)
    valid_k0 = np.isfinite(k0_rho) & (k0_rho > 0.0)
    if valid_k0.any():
        tmax_val = (
            np.max(lambertw(1.0 / ((1.0 - frac_step_max) * k0_rho[valid_k0])).real)
            * cS
        )
    else:
        tmax_val = 0.0

    tmax_days_cap = pextra.get("tmax_days_cap", None)
    if tmax_days_cap is not None:
        tmax_val = min(tmax_val, float(tmax_days_cap))
    tmax_val = max(float(tmax_val), 0.0)

    dt_lower = pextra.get("dt_lower", None)
    if dt_lower is None:
        dt_lower = infer_lower_timestep(index)
    dt_lower_days = pd.Timedelta(dt_lower) / pd.Timedelta(1.0, unit="D")
    if dt_lower_days <= 0.0:
        raise ValueError(f"dt_lower must be positive, got {dt_lower}")

    nt_max = min(max(int(np.ceil(tmax_val / dt_lower_days)) + 1, 1), nt)

    it_arr = np.arange(nt_max)[None, :] + np.arange(nt)[:, None]
    exclude = it_arr > (nt - 1)
    it_arr[exclude] = nt - 1
    time_arr = (index.to_numpy()[it_arr] - index.to_numpy()[:, None]) / pd.Timedelta(
        1.0, unit="D"
    )

    tau_arr = np.log(
        (2.0 / rho[it_arr] * beta**2) * time_arr,
        where=time_arr > 0.0,
        out=np.full_like(time_arr, fill_value=-10.0),
    )
    dh, _ = Wh_approx(rho, tau_arr, it_arr=it_arr)
    dh[time_arr == 0.0] = 0.0

    ds = dQ[:, None] / (4.0 * np.pi * kD[it_arr]) * dh

    ds_flipped = np.fliplr(ds)
    drawdown = np.array(
        [np.trace(ds_flipped, i) for i in np.arange(nt_max - 1, -nt + nt_max - 1, -1)]
    )

    if initial_q != 0.0:
        drawdown += initial_q / (4.0 * np.pi * kD) * h_inf

    if nt_max < nt:
        steady_q = np.cumsum(dQ)[: nt - nt_max]
        drawdown[nt_max:] += steady_q / (4.0 * np.pi * kD[nt_max:]) * h_inf[nt_max:]

    return drawdown


def expint(u):
    # Fast approximation for scipy.special.exp1 according to equation 7a and 7b from Srivastava(1998)
    gamma = 0.57721566490153286060  # Euler-Macheroni constant

    out = np.zeros_like(u, dtype=float)
    show = u < 1.0
    out[show] = np.log(np.exp(-gamma) / u[show]) + 0.9653 * u[show] - 0.1690 * u[show] ** 2
    show2 = np.logical_and(~show, u < 20.0)
    out[show2] = 1.0 / (u[show2] * np.exp(u[show2])) * (u[show2] + 0.3575) / (u[show2] + 1.280)
    return out


def Wh_approx(rho, tau, it_arr=None):
    """

    Parameters
    ----------
    rho     vector
    tau     array
    it_arr

    Returns
    -------

    """
    tau = np.array(tau, copy=True)
    tau[tau > 100.0] = 100.0
    h_inf = k0(rho)
    expintrho = expint(rho)
    denominator = expintrho - expint(rho / 2)
    w = np.divide(
        expintrho - h_inf,
        denominator,
        out=np.zeros_like(rho, dtype=float),
        where=denominator != 0.0,
    )
    integral = (
        h_inf[it_arr]
        - w[it_arr] * expint(rho[it_arr] / 2 * np.exp(np.abs(tau)))
        + (w - 1)[it_arr] * expint(rho[it_arr] * np.cosh(tau))
    )
    return h_inf[it_arr] + np.sign(tau) * integral, 2 * h_inf


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
