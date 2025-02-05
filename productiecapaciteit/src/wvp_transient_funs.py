import asyncio

import numpy as np
import pandas as pd
from scipy.special import k0, lambertw


def get_temp(index, mean, delta, time_offset, return_series=False):
    index_datetime = pd.DatetimeIndex(index)
    year = pd.Categorical(index_datetime.year, ordered=True)
    start_year = year.rename_categories(pd.to_datetime(year.categories, format="%Y"))
    end_year = year.rename_categories(pd.to_datetime(year.categories.astype(str) + "1231", format="%Y%m%d"))
    nday_year = end_year.map(lambda x: x.dayofyear).astype(float)
    dt_year = index_datetime - start_year.to_numpy()
    temp_data = delta * np.sin((dt_year / pd.Timedelta("1D") - time_offset) * 2 * np.pi / nday_year) + mean
    if return_series:
        return pd.Series(data=temp_data, index=index_datetime, name="wvp_model_temp")
    else:
        return temp_data.values


def visc_ratio(temp, temp_ref=10.0):
    visc_ref = (1 + 0.0155 * (temp_ref - 20.0)) ** -1.572  # / 1000  removed the division because we re taking a ratio.
    visc = (1 + 0.0155 * (temp - 20.0)) ** -1.572  # / 1000
    return visc / visc_ref


def dis(dis1, dis2):
    return (dis1 * dis1 + dis2 * dis2) ** 0.5


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
    alpha, beta, kD0, temp_delta, temp_time_offset = args[:5]
    s = f"{alpha}, {beta}, {kD0}, {temp_delta}, {temp_time_offset}"

    if "rain" in pextra:
        rain_gain, rain_timescale = args[-2:]
        s += f", {rain_gain}, {rain_timescale}"

    if "multiwell" in pextra and not pextra["multiwell_contains_r_self"]:
        alpha_multi = args[-3]
        s += f", {alpha_multi}"
    elif "multiwell" not in pextra and pextra["multiwell_contains_r_self"]:
        assert 0, "Define multiwell"

    print(s)

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

    if pextra["multiwell_contains_r_self"]:
        grouper_list = []

        for multi, distance in pextra["multiwell"]:
            grouper_list.append(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    multihantush,
                    1,
                    distance * alpha,
                    beta,
                    kD,
                )
            )
    else:
        grouper_list = [
            asyncio.get_event_loop().run_in_executor(
                None,
                multihantush,
                1,
                alpha,
                beta,
                kD,
            )
        ]

        if pextra.get("multiwell"):
            for multi, distance in pextra["multiwell"]:
                grouper_list.append(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        multihantush,
                        multi,
                        distance * alpha_multi * alpha,
                        beta,
                        kD,
                    )
                )

    # if "rain" in pextra:

    loop = asyncio.get_event_loop()
    looper = asyncio.gather(*grouper_list)
    results = loop.run_until_complete(looper)
    drawdown_model = np.stack(results).sum(axis=0)

    if return_result:
        return drawdown_model

    else:
        return np.square(
            drawdown_model[~np.isnan(pextra["drawdown_obs"])]
            - pextra["drawdown_obs"][~np.isnan(pextra["drawdown_obs"])]
        )


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


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
    dQ = np.diff(pextra["Q_obs"], prepend=0.0)
    rho = 2 * alpha * beta / np.sqrt(kD)
    cS = 1 / np.square(beta)

    # from pastas
    tmax_val = min(365, max(lambertw(1 / ((1 - pextra["frac_step_max"]) * k0(rho))).real) * cS)
    # print(f"tmax: {tmax_val:.1f} days")

    tmax = pd.Timedelta(
        value=tmax_val,
        unit="D",
    )

    nt = pextra["index"].size
    nt_max = min(np.ceil(tmax / pextra["dt_lower"]).astype(int), nt)

    it_arr = np.arange(nt_max)[None, :] + np.arange(nt)[:, None]
    exclude = it_arr > (nt - 1)
    it_arr[exclude] = nt - 1
    time_arr = (pextra["index"][it_arr] - pextra["index"][:, None]) / pd.Timedelta(1.0, unit="D")

    # compute h for nt * nt_max
    beta_arr = beta
    # u_arr = r**2 * S / (4 * kD_arr * time_arr)
    tau_arr = np.log(
        (2 / rho[it_arr] * beta_arr**2) * time_arr,
        where=time_arr != 0,
        out=np.full_like(time_arr, fill_value=-10.0),
    )
    # assert Wh goes to 1
    dh, h_inf = Wh_approx(rho, tau_arr, it_arr=it_arr)
    ds = dQ[:, None] / (4 * np.pi * kD[it_arr]) * dh

    # transient
    ds_flipped = np.fliplr(ds)
    drawdown = np.array([np.trace(ds_flipped, i) for i in np.arange(nt_max - 1, -nt + nt_max - 1, -1)])

    # steady
    drawdown[nt_max:] += np.cumsum(dQ)[: nt - nt_max] / (4 * np.pi * kD[nt_max:]) * h_inf[nt_max:]

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
    tau[tau > 100.0] = 100.0
    h_inf = k0(rho)
    expintrho = expint(rho)
    w = (expintrho - h_inf) / (expintrho - expint(rho / 2))
    I = (
        h_inf[it_arr]
        - w[it_arr] * expint(rho[it_arr] / 2 * np.exp(np.abs(tau)))
        + (w - 1)[it_arr] * expint(rho[it_arr] * np.cosh(tau))
    )
    return h_inf[it_arr] + np.sign(tau) * I, 2 * h_inf


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
