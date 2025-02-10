import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.optimize import least_squares

from productiecapaciteit import data_dir, plot_styles_dir, results_dir
from productiecapaciteit.src.strang_analyse_fun2 import (
    get_config,
    get_false_measurements,
    smooth,
    werkzaamheden_dates,
)
from productiecapaciteit.src.weerstand_pandasaccessors import LeidingResistanceAccessor  # noqa: F401

res_folder = results_dir / "Leidingweerstand"
logger_handler = logging.FileHandler(results_dir / "Leidingweerstand" / "Leidingweerstandcoefficient.log", mode="w")
stdout = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logger_handler, stdout],
)

plt.style.use(plot_styles_dir / "unhcrpyplotstyle.mplstyle")
plt.style.use(plot_styles_dir / "types" / "line.mplstyle")


def get_covariance_least_squares(res):
    # return np.sum(res.fun**2) / (len(res.fun) - len(res.x))
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    return np.dot(VT.T / s**2, VT)


def get_leiding_slope(df_dP_, df_Q_, datum, slope_val=None, fit_pt10=False):
    mask = np.logical_and(np.isfinite(df_dP_), np.isfinite(df_Q_))
    df_dP = df_dP_[mask]
    df_Q = df_Q_[mask]
    nper = len(datum)

    def get_df_a(theta, slope_val=None):
        if slope_val is None:
            slope_val, offsets = theta[0], theta[1:]
        else:
            offsets = theta[1:]

        return pd.DataFrame({"datum": datum, "offset": offsets, "slope": slope_val})

    def fun(theta):
        return get_df_a(theta, slope_val=slope_val).leiding.dp_model(df_dP.index, df_Q)

    def cost2(theta):
        if fit_pt10:
            return (fun(theta[:-1]) - theta[-1] - df_dP) ** 2
        return (fun(theta) - df_dP) ** 2

    a_approx = min((df_dP / df_Q.clip(lower=1) ** 2).median() / 2, -1e-8)

    if slope_val is None:
        x0 = [-1e-9] + nper * [a_approx]
        bounds_ = ([-5e-6, -1e-10], *(nper * ([a_approx * 100 - 0.5, 0],)))

    else:
        x0 = [slope_val] + nper * [a_approx]
        bounds_ = ([slope_val * 10, slope_val / 10], *(nper * ([a_approx * 100, 0],)))

    if fit_pt10:
        show_small_flows = df_Q_ < 1.0
        offset_est = -df_dP_[show_small_flows].median() if np.any(show_small_flows) else 0.0
        offset_est_std = df_dP_[show_small_flows].std() if np.any(show_small_flows) else 0.0
        x0 += [offset_est]
        bounds_ = (*bounds_, [offset_est - 2.0, offset_est + 2.0])

    bounds = np.array(bounds_).T

    try:
        res = least_squares(
            cost2,
            x0=x0,
            bounds=bounds,
            loss="arctan",
            f_scale=0.5,
            gtol=1e-14,
            ftol=1e-14,
            xtol=1e-14,
        )
        if np.any(res.active_mask):
            print("Optimal parameters are outside bounds")
            print(res.active_mask)

        if np.any(res.x[1:] == x0[1:]):
            print("Solver issues")

        if slope_val is not None:
            assert res.x[0] == slope_val

        if fit_pt10:
            cov = get_covariance_least_squares(res)
            offset_std = np.sqrt(cov[-1, -1])
            logging.warning(
                f"Additional offset pt10 included in result: {res.x[-1]:.2f}m +/- {offset_std:.2f}m. Median offset at Q<1m3/h: {offset_est:.2f}m +/- {offset_est_std:.2f}m"
            )
            return res, get_df_a(res.x[:-1], slope_val=slope_val)
        return res, get_df_a(res.x, slope_val=slope_val)

    except:
        res = least_squares(cost2, x0=x0, bounds=bounds, loss="arctan", f_scale=0.5)
        assert ~np.any(res.active_mask), "Optimal parameters are outside bounds"

        return None, None


def analyse_a_leiding(
    df_dP,
    df_Q,
    werkzh_datums,
    Q_avg,
    t_projectie="2023-10-31 00:00:00",
    slope=None,
    fit_pt10=False,
):
    """Returns df_a"""
    werkzh_datums = list(werkzh_datums)

    _, df_a = get_leiding_slope(df_dP, df_Q, werkzh_datums, slope_val=slope, fit_pt10=fit_pt10)

    if np.all(df_a.leiding.a_effect[1:] <= 1):
        logging.info(f"Effectieve schoonmaken: {werkzh_datums[1:]}")

    else:
        idrop = np.argmax(df_a.leiding.a_effect[1:]) + 1
        removed = werkzh_datums.pop(idrop)
        logging.info(f"=> Dropping: {removed}. Remaining dates: {werkzh_datums}")

        df_a = analyse_a_leiding(df_dP, df_Q, werkzh_datums, Q_avg, t_projectie=t_projectie, slope=slope)

    for datum, dp_voor, dp_na in zip(df_a.datum, df_a.leiding.dp_voor(Q_avg), df_a.leiding.dp_na(Q_avg), strict=False):
        logging.info(f"Schoonmaak van {datum}: Drukval bij mediaan debiet gaat van {dp_voor:.2f}m naar {dp_na:.2f}m")

    dp_voor = df_a.leiding.dp_projectie_voor(t_projectie, Q_avg)
    dp_na = df_a.leiding.dp_projectie_na(t_projectie, Q_avg, method="mean")
    logging.info(
        f"Bij schoonmaak in {t_projectie} gaat drukval bij Q={Q_avg:.1f}m3/h gaat van {dp_voor:.2f}m naar {dp_na:.2f}m"
    )
    return df_a


temp_ref = 12.0
t_projectie = "2025-10-31 00:00:00"

fig_folder = os.path.join("Resultaat")

config = get_config()
gridspec_kw = {
    "left": 0.07,
    "bottom": 0.12,
    "right": 0.94,
    "top": 0.88,
    "wspace": 0.3,
    "hspace": 0.3,
}
df_a_fp = results_dir / "Leidingweerstand" / "Leidingweerstand_modelcoefficienten.xlsx"

for strang, c in config.iterrows():
    # if "P" in strang or "Q" in strang:
    #     continue
    # if strang != 'IK96':
    #     continue

    # print(strang)
    logger_handler.setFormatter(logging.Formatter(f"{strang}\t| %(message)s"))
    stdout.setFormatter(logging.Formatter(f"{strang}\t| %(message)s"))

    logging.info("Strang:\t%s", strang)

    df_fp = data_dir / "Merged" / f"{strang}.feather"
    df = pd.read_feather(df_fp).set_index("Datum")
    df2 = df[df.Q < 1.0]
    isspui = df.spui > 2.0

    include_rules = [
        "Unrealistic flow",
        "Tijdens spuien",
        # "Little flow",
        # "Niet steady"
    ]
    untrusted_measurements = get_false_measurements(df, c, extend_hours=1, include_rules=include_rules)

    df.loc[untrusted_measurements] = np.nan
    df["dP"] = df.P - df.gws0
    df["dPdQ2"] = df.dP / df.Q**2
    df["dPdQ2_smooth"] = smooth(df.dPdQ2, days=0.5)

    dates = werkzaamheden_dates()[strang]

    # prepend first non-nan date
    dates = dates[dates > df.dPdQ2.dropna().index[0]]
    werkzh_datums = pd.Index(np.concatenate((df.dPdQ2.dropna().index[[0]].values, dates)))

    if 1:
        show1 = df.Q < 1.0
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), gridspec_kw=gridspec_kw)
        ax.scatter(df.index[show1], df.dP[show1], s=1, c="C0", alpha=0.2)
        fig_path = results_dir / "PT10offset" / f"pt10offset - timeseries - {strang}.png"
        fig.savefig(fig_path, dpi=300)
        continue

    Q_avg = df.Q.mean()
    slope = c.leiding_a_slope  # Use as starting value in optimization

    df_a = analyse_a_leiding(
        df.dP,
        df.Q,
        werkzh_datums,
        Q_avg,
        t_projectie=t_projectie,
        slope=slope,
        fit_pt10=True,
    )
    continue

print("hoi")
