"""
Report to show how to increase capacity of the system
Start with IKIEF 
Goals: 
1) how many m3/h does each "strang" currently produce at a suction pressure of -7.5 mwk?
2) Per strang per element: how much m3/h can the production be increased by improving that element (e.g. the well clogging)
    Elements:
    1) infiltration lake clogging
    2) borehole wall/ nearby aquifer / annulus clogging
    3) screen clogging
    4) "strangleiding" diameter (fix clogging amount @ 1yr since last maintenance)
    5) "verzamelleiding" diameter (fix clogging amount @ 1yr since last maintenance)

# Actions
0) Validate pressure sensors (PTofsett)
1) Get WVPweerstand (infiltration lake + well clogging) 
    - Validate method of computation
    - Validate relationship with Q
2) Get filterweerstand (screen resistance)
    - Validate method of computation
    - Validate relationship with Q
3) Get leidingweerstand (pipe resistance)
    - Validate method of computation
    - Validate relationship with Q
    - fit to D'arcy-Weisbach? 
    - Split into pipe resistance north-south and pipe resistance of the pipe towards the secundair
4) Create graph m3/h (x) vs required suction pressure (y) for CURRENT situation
    - With bedrijfsvoering: get realistic max suction pressure
    - plot max suction pressure => get max m3/h
5) Create scenario's for:
    - pandschoonmaak
    - chemische regeneratie (HD is regular maintenance, not part of this report)
    - redrill wells / extra wells
    - increase diameter pipe N-S
    - increase diameter pipe towards secundair
6) create graphs / obtain estimate of increase in max m3/h per strang
7) sit with Amon to determine how many m3/h he should take into account (scenario's?)
"""

# %% imports
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import least_squares

from productiecapaciteit import data_dir
from productiecapaciteit.src.strang_analyse_fun2 import (
    get_config,
    get_false_measurements,
)

# %% settings
res_folder = os.path.abspath(
    os.path.join(__file__, "..", "..", "results", "Increase_capacity")
)
os.makedirs(res_folder, exist_ok=True)

# %% 0A. Check pt10 offset using prolonged periods of low flow. 
# %% 0A. Check pt10 offset using prolonged periods of low flow


# %% 0A. Check pt10 offset using low-flow moments (all datapoints)

Q_threshold = 1  # m3/h (user adjustable)

config = get_config()
## optional subsetting during testing phase
strang_test = "IK92"
config = config.loc[[strang_test]]


for strang, c in config.iterrows():

    print(f"\nProcessing (low-flow) {strang}")

    # --- load data ---
    df_fp = data_dir / "Merged" / f"{strang}.feather"
    df = pd.read_feather(df_fp).set_index("Datum")

    # --- filter bad data ---
    bad = get_false_measurements(
        df,
        c,
        extend_hours=1,
        include_rules=["Unrealistic flow", "Tijdens spuien"],
    )
    df.loc[bad] = np.nan

    # --- compute dP ---
    df["dP"] = df.P - df.gws0

    # =========================================================
    # ✅ SELECT LOW-FLOW DATA (NO DURATION FILTER)
    # =========================================================
    df_low = df[df["Q"] < Q_threshold].copy()

    # compute instantaneous PT10
    df_low["pt10_inst"] = -df_low["dP"]

    # =========================================================
    # ✅ OPTIONAL: still identify continuous periods for plotting A
    # (purely visual, not used in B)
    # =========================================================
    low_flow_mask = df["Q"] < Q_threshold
    groups = (low_flow_mask != low_flow_mask.shift()).cumsum()

    periods = []

    for _, g in df.groupby(groups):

        if not (g["Q"] < Q_threshold).all():
            continue

        periods.append({
            "start": g.index[0],
            "end": g.index[-1],
        })

    df_periods = pd.DataFrame(periods)

    # =========================================================
    # ✅ PLOTTING
    # =========================================================

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True
    )

    # =========================================================
    # ✅ SUBPLOT A — Q scatter + horizontal low-flow segments
    # =========================================================

    # all Q points (default color)
    ax1.scatter(
        df.index,
        df["Q"],
        s=5,
        color="C0",
        alpha=0.5,
        label="Q"
    )

    # highlight low-flow points
    if not df_low.empty:
        ax1.scatter(
            df_low.index,
            df_low["Q"],
            s=8,
            color="green",
            alpha=0.8,
            label=f"Low flow (Q < {Q_threshold})"
        )

    # threshold line
    ax1.axhline(
        Q_threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Threshold ({Q_threshold} m³/h)"
    )

    ax1.set_ylabel("Q (m³/h)")
    ax1.set_title(f"{strang} – Low-flow identification")
    ax1.grid(True)
    ax1.legend()


    # =========================================================
    # ✅ SUBPLOT B — ALL LOW-FLOW PT10 VALUES
    # =========================================================

    if not df_low.empty:
        ax2.scatter(
            df_low.index,
            df_low["pt10_inst"],
            s=10,
            alpha=0.3,
            color="C0",
            label="PT10 (all low-flow points)"
        )

    ax2.axhline(0, color="black", linewidth=0.8)

    ax2.set_ylabel("PT10 (m)")
    ax2.set_xlabel("Time")
    ax2.grid(True)
    ax2.legend()

    # =========================================================

    fig.tight_layout()

    fig.savefig(
        os.path.join(res_folder, f"{strang}_pt10_lowflow.png"),
        dpi=300
    )

    plt.close(fig)

    print(f"Saved low-flow PT10 plot for {strang}")

print("Done (0A low-flow PT10).")







# %% 0B. Check pt10 offset per year of data part 2: clustered data fitting
# %% coworker fit (lean version)
def fit_pt10_coworker(df_dP, df_Q, slope_val):

    mask = np.isfinite(df_dP) & np.isfinite(df_Q)
    dP = df_dP[mask].values
    Q = df_Q[mask].values

    if len(dP) < 30:
        return np.nan

    def model(theta):
        pt10 = theta[0]
        return slope_val * Q**2 - pt10

    def cost(theta):
        return model(theta) - dP

    try:
        res = least_squares(cost, x0=[0.0], bounds=([-5], [5]))
        return -res.x[0]
    except:
        return np.nan


# %% representative points (1 per 10 m³/h bin)
def get_cluster_representatives(df_year):

    df = df_year[df_year.Q > 50].copy()

    if len(df) < 50:
        return None

    df["Q_bin"] = (df.Q / 10).round() * 10

    reps = []

    for Q_bin, g in df.groupby("Q_bin"):

        if len(g) < 15:
            continue

        # robust vertical position (tunable between 0.3–0.5)
        dP_rep = g.dP.quantile(0.4)

        reps.append({"Q": Q_bin, "dP": dP_rep})

    if len(reps) < 5:
        return None

    return pd.DataFrame(reps)


# %% cluster-based fit
def fit_cluster_darcy(df_rep, slope_init):

    if df_rep is None or len(df_rep) < 5:
        return np.nan, np.nan

    Q = df_rep.Q.values
    dP = df_rep.dP.values

    def model(theta):
        a, pt10 = theta
        return a * Q**2 - pt10

    def cost(theta):
        return model(theta) - dP

    x0 = [slope_init, 0.0]
    bounds = ([-1e-4, -5], [-1e-10, 5])

    try:
        res = least_squares(cost, x0=x0, bounds=bounds)
        a_fit, pt10 = res.x
        return a_fit, -pt10
    except:
        return np.nan, np.nan


# %% MAIN LOOP
#config = get_config()

for strang, c in config.iterrows():

    print(f"\nProcessing {strang}")

    # --- load data ---
    df_fp = data_dir / "Merged" / f"{strang}.feather"
    df = pd.read_feather(df_fp).set_index("Datum")

    # --- filter bad data ---
    bad = get_false_measurements(
        df,
        c,
        extend_hours=1,
        include_rules=["Unrealistic flow", "Tijdens spuien"],
    )
    df.loc[bad] = np.nan

    # --- compute pressure drop ---
    df["dP"] = df.P - df.gws0

    years = sorted(df.index.year.unique())

    # --- figure setup ---
    ncols = 3
    nrows = int(np.ceil(len(years) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, year in enumerate(years):

        ax = axes[i]

        df_year = df[df.index.year == year].dropna(subset=["Q", "dP"])

        if len(df_year) < 50:
            continue

        # =========================================================
        # 1. Coworker fit (least squares)
        # =========================================================
        pt10_old = fit_pt10_coworker(
            df_year.dP,
            df_year.Q,
            slope_val=c.leiding_a_slope
        )

        if not np.isnan(pt10_old):
            Q_line = np.linspace(0, df_year.Q.max(), 200)
            dP_old = c.leiding_a_slope * Q_line**2 - pt10_old
        else:
            Q_line = None

        # =========================================================
        # 2. Representative points (1 per bin)
        # =========================================================
        df_rep = get_cluster_representatives(df_year)

        # =========================================================
        # 3. Cluster-based fit
        # =========================================================
        a_new, pt10_new = fit_cluster_darcy(df_rep, c.leiding_a_slope)

        if not np.isnan(a_new):
            Q_line_new = np.linspace(0, df_year.Q.max(), 200)
            dP_new = a_new * Q_line_new**2 - pt10_new
        else:
            Q_line_new = None

        # =========================================================
        # PLOT
        # =========================================================

        # raw data
        ax.scatter(
            df_year.Q,
            df_year.dP,
            s=5,
            alpha=0.15,
            label="Data"
        )

        # ✅ ONLY representative points (large markers)
        if df_rep is not None:
            ax.scatter(
                df_rep.Q,
                df_rep.dP,
                s=80,
                color="orange",
                edgecolor="black",
                label="Bin representatives"
            )

        # coworker fit
        if Q_line is not None:
            ax.plot(
                Q_line,
                dP_old,
                color="C3",
                linewidth=2,
                label=f"LS fit ({pt10_old:.2f})"
            )

        # cluster fit
        if Q_line_new is not None:
            ax.plot(
                Q_line_new,
                dP_new,
                color="green",
                linewidth=2,
                label=f"Cluster fit ({pt10_new:.2f})"
            )

        ax.set_title(f"{year}")
        ax.set_xlabel("Q (m³/h)")
        ax.set_ylabel("dP (m)")
        ax.grid(True)

    # remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{strang} – PT10 fit (representative points)", fontsize=14)

    fig.tight_layout(rect=[0, 0, 0.95, 0.95])

    fig.savefig(
        os.path.join(res_folder, f"{strang}_cluster_fit.png"),
        dpi=300
    )
    plt.close(fig)

    print(f"Saved {strang}")

print("Done.")
