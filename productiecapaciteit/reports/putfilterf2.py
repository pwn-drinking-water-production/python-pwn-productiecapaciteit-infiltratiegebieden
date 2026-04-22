"""
Gebuik dit script om de filterweerstandcoefficienten te berekenen.

Ongecorrigeerd voor temperatuur. Implementeer viscositeit.
"""

import logging
import os
from sys import stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from productiecapaciteit import data_dir, plot_styles_dir, results_dir
from productiecapaciteit.src.strang_analyse_fun2 import get_config, get_false_measurements, smooth, werkzaamheden_dates
from productiecapaciteit.src.weerstand_pandasaccessors import WellResistanceAccessor  # noqa: F401

res_folder = results_dir / "Alleputweerstanden"
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

filterweerstand_fp = results_dir / "Filterweerstand" / "Filterweerstand_modelcoefficienten.xlsx"

for strang, c in config.iterrows():
    df_a = pd.read_excel(filterweerstand_fp, sheet_name=strang)
    print(strang)

    df_fp = data_dir / "Merged" / f"{strang}.feather"
    df = pd.read_feather(df_fp).set_index("Datum")

    include_rules = ["Unrealistic flow"]
    untrusted_measurements = get_false_measurements(df, c, extend_hours=1, include_rules=include_rules)
    df.loc[untrusted_measurements] = np.nan

    df["dPdQ"] = (df.gws0 - df.gws1) / (df.Q / c.nput)
    df["dPdQ_smooth"] = smooth(df["dPdQ"], days=1)

    df["R_f"] = (df.gws0 - df.gws1) / (df.Q / c.nput) # resistance at/near the filter
    df["R_f"] = smooth(df["R_f"], days=1)
    df["R_bw"] = (df.gws1 - df.gws2) / (df.Q / c.nput) # resistance at borehole wall and nearby aquifer, possibly affected by vertical resistance below the filter
    df["R_bw"] = smooth(df["R_bw"], days=1)
    df["R_tot"] = df["R_f"] + df["R_bw"]


    dates = werkzaamheden_dates()[strang]
    dates = dates[dates > df.dPdQ.dropna().index[0]]
    werkzh_datums = pd.Index(np.concatenate((df.dPdQ.dropna().index[[0]].values, dates)))

    Q_avg = df.Q.quantile(0.95) / c.nput

    plt.style.use(plot_styles_dir / "unhcrpyplotstyle.mplstyle")
    plt.style.use(plot_styles_dir / "types" / "line.mplstyle")
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 5), gridspec_kw=gridspec_kw)
    ax = ax2.twinx()
    fig.suptitle(strang)
    ax.axhline(0, c="black", lw="0.8")
    df_a.leiding.plot_werkzh(ax, werkzh_datums)

    ax.plot(df.index, df_a.wel.a_model(df.index), label=f"Model {df_a.slope.mean():.3g}")
    ax.plot(df.index, df["R_f"], label="R_f (dag gem.)")
    ax.plot(df.index, df["R_bw"], label="R_bw (dag gem.)")
    ax.plot(df.index, df["R_tot"], label="R_tot (dag gem.)")
    ax.set_ylabel("Weerstand: Verlaging bij Q = 1 m3/h (m)")
    ax.set_ylim(-6 / Q_avg, 1 / Q_avg)
    ax.set_yticks(np.arange(-6, 2) / Q_avg)
    # ax.set_xlim(df.index[[0, -1]])
    # ax.legend(fontsize="small")
    ax.legend(loc=(0, 1), ncol=4)

    ax2.set_ylim((-6, 1))
    ax2.set_yticks(np.arange(-6, 2))
    ax2.set_xlim(df.index[[0, -1]])
    ax2.set_ylabel(f"Verlaging bij Q_put={Q_avg:.1f}m3/h (m)")

    fig_path = os.path.join(res_folder, f"Putweerstandcoefficient - {strang}.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


