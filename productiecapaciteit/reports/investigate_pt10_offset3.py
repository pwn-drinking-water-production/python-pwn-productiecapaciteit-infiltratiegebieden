import logging
import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from strang_analyse_fun2 import (
    get_config,
    get_false_measurements,
    model_a_leiding,
    get_werkzaamheden_intervals,
    remove_per_from_werkzh_per,
)

res_path = os.path.join("Resultaat", "PT10offset")
logger_handler = logging.FileHandler(
    os.path.join(res_path, "Leidingweerstandcoefficient.log"), mode="w"
)  # , encoding='utf-8', level=logging.DEBUG)
stdout = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logger_handler, stdout],
)
"""
De dP_leiding van IK104, IK105 en IK106 vertonen grote offsets
  - 
"""


def smooth(df, days=1):
    window = int(timedelta(days=days) / (df.index[1] - df.index[0]))
    return df.dropna().rolling(window=window, center=True).median().reindex(df.index)


def get_leiding_slope_per_year2(df_, periods, freq="3H", Q=None, slope=None):
    isna1 = pd.isna(df_.Q) | pd.isna(df_.dP_leiding)
    df = df_[~isna1].groupby(pd.Grouper(freq=freq)).median()
    isna2 = pd.isna(df.Q) | pd.isna(df.dP_leiding)
    df = df[~isna2]
    nper = len(periods)
    # df = pd.concat((
    #     df.loc[:'2016-01-01'],
    #     df.loc['2018-01-01':]
    # ))

    if Q is None:
        dP = df.dP_leiding
    else:
        dP = df.dP_leiding / df.Q**2 * Q**2
        assert np.all(np.isfinite(dP)), "Q is zero or nan values in df"

    def fun(theta):
        slope, offsets = theta[0], theta[1:]
        return model_a_leiding(df, periods, slope, offsets, Q=Q)

    def cost2(args, slope=None):
        beta = args[0]

        theta = args[1:]

        if slope is not None:
            theta[0] = slope

        return (fun(theta) + beta - dP) ** 2

    cost3 = lambda args: cost2(args, slope=slope)

    a_approx = -(df.dP_leiding / df.Q**2).median()
    dx_days = (dP.index - dP.index[0]) / timedelta(days=1)
    slope_approx = (np.sum(dx_days * dP) - dP.size * np.median(dx_days) * np.median(dP)) / (
        np.sum(dx_days * dx_days) - dP.size * np.median(dx_days) * np.median(dx_days)
    )
    x0 = [0.0, -5e-9] + nper * [a_approx]

    bounds_ = (
        [-0.5, 0.5],
        [-5e-6, -1e-10],
        *(nper * ([a_approx / 10, a_approx * 10],)),
    )
    bounds = np.array(bounds_).T

    try:
        # res0 = minimize(lambda x: sum(cost3(x)), x0=x0, bounds=bounds_)
        res = least_squares(cost3, x0=x0, bounds=bounds, loss="arctan", f_scale=0.5)
        if np.any(res.active_mask):
            print("Optimal parameters are outside bounds")

        if slope is not None:
            res.x[1] = slope
        return res, fun(res.x[1:])

    except:
        res = least_squares(cost3, x0=x0, bounds=bounds, loss="arctan", f_scale=0.5)
        return None, None


temp_ref = 12.0

fig_folder = os.path.join("Resultaat")

data_fd = os.path.join("..", "Data")
config_fn = "strang_props6.xlsx"
config = get_config(os.path.join(data_fd, config_fn))
gridspec_kw = {
    "left": 0.07,
    "bottom": 0.12,
    "right": 0.94,
    "top": 0.88,
    "wspace": 0.2,
    "hspace": 0.2,
}

werkzh_fp = os.path.join("..", "Data", "Werkzaamheden.xlsx")


def analyse_a_leiding_voordeel_schoonmaak(
    df, werkzh_per, slope, offsets, Q_avg=None, t_projectie="2023-10-31 00:00:00"
):
    if Q_avg is None:
        Q_avg = df.Q.median()

    m_meas = model_a_leiding(df, werkzh_per, slope, offsets, Q=None)
    m_avg = model_a_leiding(df, werkzh_per, slope, offsets, Q=Q_avg)
    m_a = model_a_leiding(df, werkzh_per, slope, offsets, Q=1.0)

    if len(werkzh_per) < 2:
        logging.info("Geen schoonmaak tijdens de meetperiode")
        return [], (np.mean(offsets), np.mean(offsets))

    m_a_red_frac_result = []

    for start, _ in werkzh_per[1:]:
        ilook = -2
        m_a_red = m_a[start:][0] - m_a[:start][ilook]
        m_a_red_frac = m_a_red / m_a[:start][ilook] * 100
        m_avg_red = m_avg[start:][0] - m_avg[:start][ilook]
        m_avg_red_frac = m_avg_red / m_avg[:start][ilook] * 100
        m_a_red_frac_result.append(m_a_red_frac)

        logging.info(
            f"Schoonmaak van {start} reduceert de frictie coefficient met {-m_a_red_frac:.1f}%. Verval bij gem debiet gaat van {m_avg[:start][-2]:.2f}m naar {m_avg[start:][0]:.2f}m"
        )

    mean_result = np.mean(m_a_red_frac_result)

    if mean_result > 0:
        logging.info("Een schoonmaak heeft geen effect op de leidingweerstand.")

        return np.array(m_a_red_frac_result), (np.mean(offsets), np.mean(offsets))

    else:
        tend = m_a[~m_a.isna()].index[-1]
        end = m_a[~m_a.isna()][-1]
        projected_voor = end + slope * (pd.Timestamp(t_projectie) - tend) / pd.Timedelta(days=1)
        projected_na = projected_voor * (100 + mean_result) / 100

        projected_dP_voor = projected_voor * Q_avg**2
        projected_dP_na = projected_na * Q_avg**2

        logging.info(
            f"Bij schoonmaak in {t_projectie} gaat verval bij gem debiet gaat van {projected_dP_voor:.2f}m naar {projected_dP_na:.2f}m"
        )
        return np.array(m_a_red_frac_result), (projected_voor, projected_na)


def analyse_a_leiding(df, res, werkzh_per, Q_avg=None, t_projectie="2023-10-31 00:00:00", slope=None):
    """
    *  0 : a constraint is not active.
    * -1 : a lower bound is active.
    *  1 : an upper bound is active.
    :param res:
    :return:
    """
    islope = 1
    ioffsets = range(2, res.x.size)

    if slope is None:
        slope = res.x[islope]

    offsets = res.x[ioffsets]

    pers = np.array([i[0] for i in werkzh_per][1:])

    # fractional gains of past maintenance and maintenance prediction
    gains, projected = analyse_a_leiding_voordeel_schoonmaak(
        df, werkzh_per, slope, offsets, Q_avg=Q_avg, t_projectie=t_projectie
    )

    if len(werkzh_per) > 1 and np.any(gains > 0):
        # Running analysis again excluding the least favorable schoonmaak.
        idrop = np.argmax(gains)
        logging.info(f"=> Dropping: {pers[idrop]}")

        werkzh_per = remove_per_from_werkzh_per(werkzh_per, idrop)
        logging.info(f"Remaining periods {werkzh_per}")

        res, dP_leiding_model = get_leiding_slope_per_year2(df, werkzh_per, freq="1H", Q=Q_avg, slope=slope)

        gains, projected, werkzh_per = analyse_a_leiding(df, res, werkzh_per, Q_avg=Q_avg, t_projectie=t_projectie)
    else:
        logging.info("Alle schoonmaken hebben zin")

    return gains, projected, werkzh_per


def remove_per_from_werkzh_per(werkzh_per, idrop):
    nper = len(werkzh_per) - 1
    assert idrop <= nper - 1

    dates = np.unique(werkzh_per)
    dates = np.array([date for i, date in enumerate(dates) if i != idrop + 1])

    out = []

    for start, end in zip(dates[:-1], dates[1:]):
        out.append((start, end))

    return out


for strang, c in config.iterrows():
    if strang != "IK104":
        continue

    # print(strang)
    logger_handler.setFormatter(logging.Formatter(f"{strang}\t| %(message)s"))
    stdout.setFormatter(logging.Formatter(f"{strang}\t| %(message)s"))

    logging.info(f"Strang: {strang}")

    # df_fp = os.path.join(data_fd, "Merged", f"{strang}.feather")
    df_fp = os.path.join(data_fd, "Merged", f"{strang}-PKA-DSEW036680.feather")
    df = pd.read_feather(df_fp)
    df["Datum"] = pd.to_datetime(df["Datum"])
    df.set_index("Datum", inplace=True)

    include_rules = [
        "Unrealistic flow",
        # "Tijdens spuien",
        # "Little flow"
        # "Niet steady"
    ]
    untrusted_measurements = get_false_measurements(df, c, extend_hours=1, include_rules=include_rules)
    df.loc[untrusted_measurements] = np.nan
    df["dP_leiding"] = smooth(df.P - df.gws0, days=0.5)

    werkzh_per = get_werkzaamheden_intervals(df.dP_leiding.dropna().index, werkzh_fp, strang)
    pers = np.array([i[0] for i in werkzh_per][1:])

    # bonus mask
    a200 = df.dP_leiding / df.Q**2 * 200**2
    lims = np.nanpercentile(a200, q=[5, 95])
    df.loc[a200 < lims[0]] = np.nan
    df.loc[a200 > lims[1]] = np.nan

    Q_fit = df.Q.median()

    # sv_out_ = get_leiding_slope_per_year1(df, starting_values[strang], freq='6H')
    res, dP_leiding_model = get_leiding_slope_per_year2(df, werkzh_per, freq="1H", Q=Q_fit, slope=c.leiding_a_slope)
    if res is None:
        logging.info("FAILED")
        continue

    # Continue computation with only the werkzh_per that matter
    gains, projected, werkzh_per2 = analyse_a_leiding(
        df,
        res,
        werkzh_per,
        Q_avg=None,
        t_projectie="2023-10-31 00:00:00",
        slope=c.leiding_a_slope,
    )
    res, dP_leiding_model = get_leiding_slope_per_year2(df, werkzh_per2, freq="1H", Q=Q_fit, slope=c.leiding_a_slope)
    pers2 = np.array([i[0] for i in werkzh_per2][1:])

    # Plot results of werkzh that matter
    sv_out = res.x
    beta = sv_out[0]
    theta = sv_out[1:]
    logging.info(f"('{strang}',\t{beta},\t{theta}),")

    slope, offsets = theta[0], theta[1:]

    fig, axs = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw=gridspec_kw)
    fig.suptitle(strang)
    ax = axs[0]
    ax.vlines(
        pers,
        ymin=0,
        ymax=1,
        linewidth=1,
        color="C4",
        transform=ax.get_xaxis_transform(),
        label="Werkzaamheden volgens Excel",
    )
    ax.vlines(
        pers2,
        ymin=0,
        ymax=1,
        linewidth=1,
        color="C5",
        transform=ax.get_xaxis_transform(),
        label="Werkzaamheden meegenomen in model",
    )
    ax.plot(df.index, df["dP_leiding"], c="C0")
    m = model_a_leiding(df, werkzh_per2, slope, offsets)
    ax.plot(m.index, m, c="C4")
    ax.legend(fontsize="small")

    ax2 = ax.twinx()
    ax2.plot(df.index, smooth(df.Q, days=0.5), linewidth=0.8, c="C2")

    df["axx_leiding"] = df.dP_leiding / df.Q**2 * Q_fit**2

    m = model_a_leiding(df, werkzh_per2, slope, offsets, Q=Q_fit)
    m.plot(ax=axs[1])
    df["axx_leiding"].plot(
        ax=axs[1],
    )
    axs[0].set_ylim((-4, 0))
    axs[1].set_ylim((-4, 0))
    fig.savefig(os.path.join(res_path, f"Leidingweerstandcoefficient - {strang}.png"), dpi=300)
    logging.info(f"Saved result to {os.path.join('Resultaat', f'Leidingweerstandcoefficient - {strang}.png')}")

    plt.show()

# print("hoi")

print("hoi")
