"""
Analyse van de leidingweerstand van de strangen.

De leidingweerstand van de strangen wordt geanalyseerd door de drukval over de strang te relateren aan het debiet. De
leidingweerstand wordt gemodelleerd als een lineaire functie van het debiet. De modelcoëfficiënten worden bepaald door
de drukval bij verschillende debieten te meten en te modelleren.

TODO: pt10offset laat zien dat soms de de sensoren opnieuw ingehangen zijn. => Opsplitsen offset in tijdreeks en in prepare_data.py toevoegen
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from productiecapaciteit import data_dir, plot_styles_dir
from productiecapaciteit.src.strang_analyse_fun2 import (
    get_config,
    get_false_measurements,
    smooth,
    werkzaamheden_dates,
)
from productiecapaciteit.src.weerstand_pandasaccessors import LeidingResistanceAccessor  # noqa: F401

res_folder = os.path.abspath(os.path.join(__file__, "..", "..", "results", "Leidingweerstand"))
logger_handler = logging.FileHandler(
    os.path.join(res_folder, "Leidingweerstandcoefficient.log"), mode="w"
)  # , encoding='utf-8', level=logging.DEBUG) res_folder,
stdout = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logger_handler, stdout],
)

plt.style.use(plot_styles_dir / "unhcrpyplotstyle.mplstyle")
plt.style.use(plot_styles_dir / "types" / "line.mplstyle")


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
        show_small_flows = df_Q < 1.0
        offset_est = df_dP_[show_small_flows].median() if np.any(show_small_flows) else 0.0
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
            logging.info(f"Additional offset pt10: {res.x[-1]:.2f}m")
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
df_a_fp = os.path.join(res_folder, "Leidingweerstand_modelcoefficienten.xlsx")

for strang, c in config.iterrows():
    # if "P" in strang or "Q" in strang:
    #     continue
    # if strang != 'IK96':
    #     continue

    # print(strang)
    logger_handler.setFormatter(logging.Formatter(f"{strang}\t| %(message)s"))
    stdout.setFormatter(logging.Formatter(f"{strang}\t| %(message)s"))

    logging.info("Strang: %s", strang)

    df_fp = data_dir / "Merged" / f"{strang}.feather"
    df = pd.read_feather(df_fp).set_index("Datum")
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

    Q_avg = df.Q.mean()
    slope = c.leiding_a_slope  # Use as starting value in optimization

    df_a = analyse_a_leiding(
        df.dP,
        df.Q,
        werkzh_datums,
        Q_avg,
        t_projectie=t_projectie,
        slope=slope,
        fit_pt10=False,
    )

    df_a["gewijzigd"] = pd.Timestamp.now()
    df_a = df_a.leiding.add_zero_effect_dates(dates)
    with pd.ExcelWriter(df_a_fp, if_sheet_exists="replace", mode="a", engine="openpyxl") as writer:
        df_a.to_excel(writer, sheet_name=strang)

    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw=gridspec_kw)
    fig.suptitle(strang)

    # Leidingweerstand coeff
    ax.fill_between(
        df.index,
        0,
        1,
        where=isspui,
        color="green",
        alpha=0.4,
        transform=ax.get_xaxis_transform(),
        label="Spuien",
        linewidth=1.5,
    )

    ax.axhline(0, c="black", lw="0.8")
    df_a.leiding.plot_werkzh(ax, werkzh_datums)
    ax.plot(df.index, df["dPdQ2_smooth"], label="dP/dQ2 (dag gem.)")
    ax.plot(df.index, df_a.leiding.a_model(df.index), label=f"Model {df_a.slope.mean():.3g}")
    ax.set_ylabel("Verlaging bij Q = 1 m3/h (m)")
    ax.set_ylim(-4 / Q_avg**2, 1 / Q_avg**2)
    ax.set_xlim(df.index[[0, -1]])
    # ax.legend(fontsize="small")
    ax.legend(loc=(0.05, 1), ncol=4)

    # Gemeten en gemodelleerde verlaging bij gemeten debiet
    ax2.axhline(0, c="black", lw="0.8")
    df_a.leiding.plot_werkzh(ax2, werkzh_datums)
    ax2.plot(df.index, smooth(df.P - df.gws0, days=1), label="Gemeten verlaging (dag gem.)")
    ax2.plot(
        df.index,
        df_a.leiding.dp_model(df.index, smooth(df.Q, days=1)),
        label=f"Model {df_a.slope.mean():.3g}",
    )
    ax2.set_ylim(-4, 1)
    ax2.set_xlim(df.index[[0, -1]])
    ax2.set_ylabel("Verlaging bij gemeten Q (m)")
    # ax2.legend(fontsize="small")
    ax2.legend(loc=(0, 1), ncol=4)

    # Gemeten en gemodelleerde verlaging bij gem debiet
    ax3.axhline(0, c="black", lw="0.8")
    df_a.leiding.plot_werkzh(ax3, werkzh_datums)

    ax3.plot(
        df.index,
        smooth((df.P - df.gws0) / df.Q**2 * Q_avg**2, days=1),
        label="Gemeten verlaging (dag gem.)",
    )
    ax3.plot(
        df.index,
        df_a.leiding.dp_model(df.index, Q_avg),
        label=f"Model {df_a.slope.mean():.3g}",
    )
    ax3.set_ylim(-4, 1)
    ax3.set_xlim(df.index[[0, -1]])
    ax3.set_ylabel(f"Verlaging bij Q={Q_avg:.0f} m3/h (m)")
    # ax3.legend(fontsize="small")
    ax3.legend(loc=(0, 1), ncol=4)

    fig_path = os.path.join(res_folder, f"Leidingweerstandcoefficient - {strang}.png")
    fig.savefig(fig_path, dpi=300)
    logging.info(f"Saved result to {fig_path}")

    if 0:
        res, d2 = get_leiding_slope(df.dP, df.Q, df_a.datum, slope_val=c.leiding_a_slope, fit_pt10=True)
        offset = res.x[-1]
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), gridspec_kw=gridspec_kw)
        ax.scatter(df.Q, df.dP + offset, s=1, c="C0", alpha=0.1)
        ax.scatter(df.Q, d2.leiding.dp_model(df.index, flow=df.Q), s=1, c="C1", alpha=0.1)
        fig.suptitle(strang + f": inclusief offset {offset:.2f}m")
        fig_path = os.path.join(res_folder, f"pt10offset - {strang}.png")
        fig.savefig(fig_path, dpi=300)

plt.show()
print("hoi")
