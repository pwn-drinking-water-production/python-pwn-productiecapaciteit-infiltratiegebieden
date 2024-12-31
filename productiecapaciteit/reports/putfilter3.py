"""
Gebuik dit script om de filterweerstandcoefficienten te berekenen.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from productiecapaciteit import data_dir, results_dir
from productiecapaciteit.src.strang_analyse_fun2 import get_config, get_false_measurements, smooth, werkzaamheden_dates
from productiecapaciteit.src.weerstand_pandasaccessors import WellResistanceAccessor  # noqa: F401

res_folder = results_dir / "Filterweerstand"
logger_handler = logging.FileHandler(res_folder / "Putweerstandcoefficient.log", mode="w")
stdout = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logger_handler, stdout],
)


def get_put_slope_per_year2(df_dPdQ_, datum, slope_val=None):
    df_dPdQ = df_dPdQ_[np.isfinite(df_dPdQ_)]
    nper = len(datum)

    # a_lim = np.quantile(df_dPdQ - slope_val * days, 0.5)

    if slope_val is None:

        def get_df_a(theta):
            slope, offsets = theta[0], theta[1:]
            return pd.DataFrame({"datum": datum, "offset": offsets, "slope": slope})

        def cost2(theta):
            return (get_df_a(theta).wel.a_model(df_dPdQ.index) - df_dPdQ) ** 2

        a_lim, a_approx = np.quantile(df_dPdQ, [0.1, 0.8])
        a_approx = min(a_approx, -1e-8)
        bounds_ = ([-5e-4, -5e-6], *(nper * ([a_lim, -1e-9],)))
        x0 = [-5e-5] + nper * [a_approx]
    else:
        days = (df_dPdQ.index - df_dPdQ_.index[0]) / pd.Timedelta(days=1)

        def get_df_a(theta):
            offsets = theta
            return pd.DataFrame({"datum": datum, "offset": offsets, "slope": slope_val})

        def cost2(theta):
            return (get_df_a(theta).wel.a_model(df_dPdQ.index) - df_dPdQ) ** 2

        a_lim, a_approx = np.quantile(df_dPdQ - slope_val * days, [0.1, 0.5])
        a_approx = min(a_approx, -1e-8)
        a_lim = min(-0.5, a_lim)
        bounds_ = nper * ([a_lim, 0],)
        x0 = nper * [a_approx]

    bounds = np.array(bounds_).T

    try:
        res = least_squares(cost2, x0=x0, bounds=bounds, loss="arctan", f_scale=0.5, gtol=1e-14, ftol=1e-14, xtol=1e-14)
        if np.any(res.active_mask):
            print("Optimal parameters are outside bounds")

        return res, get_df_a(res.x)

    except:
        res = least_squares(cost2, x0=x0, bounds=bounds, loss="arctan", f_scale=0.5)
        assert ~np.any(res.active_mask), "Optimal parameters are outside bounds"

        return None, None


def analyse_a_put2(df_dPdQ, werkzh_datums, Q_avg, t_projectie="2023-10-31 00:00:00", slope=None):
    """returns df_a"""
    werkzh_datums = list(werkzh_datums)

    _, df_a = get_put_slope_per_year2(df_dPdQ, werkzh_datums, slope_val=slope)

    if np.all(df_a.wel.a_effect[1:] <= 1):
        logging.info(f"Effectieve schoonmaken: {werkzh_datums[1:]}")
        pass

    else:
        idrop = np.argmax(df_a.wel.a_effect[1:]) + 1
        removed = werkzh_datums.pop(idrop)
        logging.info(f"=> Dropping: {removed}. Remaining dates: {werkzh_datums}")

        df_a = analyse_a_put2(df_dPdQ, werkzh_datums, Q_avg, t_projectie=t_projectie, slope=slope)

    for datum, dp_voor, dp_na in zip(df_a.datum, df_a.wel.dp_voor(Q_avg), df_a.wel.dp_na(Q_avg)):
        logging.info(f"Schoonmaak van {datum}: Drukval bij mediaan debiet gaat van {dp_voor:.2f}m naar {dp_na:.2f}m")

    dp_voor = df_a.wel.dp_projectie_voor(t_projectie, Q_avg)
    dp_na = df_a.wel.dp_projectie_na(t_projectie, Q_avg, method="mean")
    logging.info(
        f"Bij schoonmaak in {t_projectie} gaat drukval bij Q={Q_avg:.1f}m3/h gaat van {dp_voor:.2f}m naar {dp_na:.2f}m"
    )
    return df_a


temp_ref = 12.0

fig_folder = os.path.join("Resultaat")

config = get_config()
gridspec_kw = {
    "left": 0.07,
    "bottom": 0.12,
    "right": 0.9,
    "top": 0.88,
    "wspace": 0.2,
    "hspace": 0.2,
}

df_a_fp = os.path.join(res_folder, "Filterweerstand_modelcoefficienten.xlsx")


for strang, c in config.iterrows():
    # if strang != "IK105":
    #     continue

    print(strang)
    logger_handler.setFormatter(logging.Formatter(f"{strang}\t| %(message)s"))
    stdout.setFormatter(logging.Formatter(f"{strang}\t| %(message)s"))

    logging.info(f"Strang: {strang}")

    df_fp = data_dir / "Merged" / f"{strang}.feather"
    df = pd.read_feather(df_fp).set_index("Datum")

    include_rules = ["Unrealistic flow"]
    untrusted_measurements = get_false_measurements(df, c, extend_hours=1, include_rules=include_rules)
    df.loc[untrusted_measurements] = np.nan

    df["dPdQ"] = (df.gws0 - df.gws1) / (df.Q / c.nput)
    df["dPdQ_smooth"] = smooth(df["dPdQ"], days=1)

    dates = werkzaamheden_dates()[strang]
    dates = dates[dates > df.dPdQ.dropna().index[0]]
    werkzh_datums = pd.Index(np.concatenate((df.dPdQ.dropna().index[[0]].values, dates)))

    Q_avg = df.Q.mean() / c.nput
    slope = c.put_a_slope
    df_a = analyse_a_put2(df.dPdQ, werkzh_datums, Q_avg, t_projectie="2025-10-31 00:00:00", slope=slope)

    # save results
    df_a["gewijzigd"] = pd.Timestamp.now()
    df_a = df_a.leiding.add_zero_effect_dates(dates)
    with pd.ExcelWriter(df_a_fp, if_sheet_exists="replace", mode="a", engine="openpyxl") as writer:
        df_a.to_excel(writer, sheet_name=strang)

    plt.style.use(["unhcrpyplotstyle", "line"])
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 5), gridspec_kw=gridspec_kw)
    ax = ax2.twinx()
    fig.suptitle(strang)
    ax.axhline(0, c="black", lw="0.8")
    df_a.leiding.plot_werkzh(ax, werkzh_datums)

    ax.plot(df.index, df["dPdQ_smooth"], label="dP/dQ (dag gem.)")
    ax.plot(df.index, df_a.wel.a_model(df.index), label=f"Model {df_a.slope.mean():.3g}")
    ax.set_ylabel("Weerstand: Verlaging bij Q = 1 m3/h (m)")
    ax.set_ylim(-4 / Q_avg, 1 / Q_avg)
    ax.set_yticks(np.arange(-4, 2) / Q_avg)
    # ax.set_xlim(df.index[[0, -1]])
    # ax.legend(fontsize="small")
    ax.legend(loc=(0, 1), ncol=4)

    ax2.set_ylim((-4, 1))
    ax2.set_yticks(np.arange(-4, 2))
    ax2.set_xlim(df.index[[0, -1]])
    ax2.set_ylabel(f"Verlaging bij Q_put={Q_avg:.1f}m3/h (m)")

    fig_path = os.path.join(res_folder, f"Putweerstandcoefficient - {strang}.png")
    fig.savefig(fig_path, dpi=300)
    logging.info(f"Saved result to {fig_path}")

print("hoi")
