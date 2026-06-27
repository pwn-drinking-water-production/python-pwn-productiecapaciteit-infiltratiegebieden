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
=> semi-done. Check plots with Bas
=> moving maximum / 99prctl
=> model: 90th prctl instead of median of
=> Bas: transient model to compute PT10offset
1) Get WVPweerstand (infiltration lake + well clogging)
    - Validate method of computation
    - Validate relationship with Q
=>
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
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from productiecapaciteit import data_dir
from productiecapaciteit.src.strang_analyse_fun2 import (
    get_config,
    get_false_measurements,
)

# %% settings
res_folder = os.path.abspath(os.path.join(__file__, "..", "..", "results", "Increase_capacity"))
os.makedirs(res_folder, exist_ok=True)

"""
PT offset estimation using system curve reconstruction
P=Poffset\u200b−(Rlin\u200b⋅Q+Rnl\u200b⋅Qβ)
so the amount of linear term is reflected by Rlin
and the amount of nonlinear term is reflected by Rnl
=> expectation: IKIEF higher Rnl (more 'pipe')
but not necessarily higher beta.

Includes:
✅ Stable period detection
✅ Median aggregation
✅ Linearisation: 1/Qspec vs Q
✅ Robust fitting (soft_l1)
✅ Rolling yearly fit (continuous)
✅ Yearly summary + rolling diagnostics
"""

# %% settings
window_hours = 12  # window length to identify stable flow periods
Q_range_threshold = 20  # stability definition (m3/h)
min_points = 8  # minimum samples inside window

rolling_days = 90  # window length to fit total system behaviour
min_points_fit = 10
lowflowcutoff = 2  # maximum flow m3/h to be classified as low flow
mintimelowflow = 4  # at least 4 hours of consecutive low flow for 'system at rest'
# beta per strang setting (estimated using below piece of script)
# %% beta settings (per strang)
beta_settings = {
    # IKIEF
    "IK91": 2.01,
    "IK92": 1.99,
    "IK93": 2.36,
    "IK94": 2.26,
    "IK95": 2.17,
    "IK96": 1.96,
    "IK101": 1.88,
    "IK102": 1.59,
    "IK103": 1.57,
    "IK104": 1.78,
    "IK105": 1.81,
    "IK106": 1.79,
    # ICAS
    "P100": 2.09,
    "P200": 2.34,
    "P300": 2.02,
    "P400": 2.47,
    "P500": 2.15,
    "P600": 2.07,
    "Q100": 2.01,
    "Q200": 1.82,
    "Q300": 2.17,
    "Q400": 2.18,
    "Q500": 1.90,
    "Q600": 1.71,
}


# %% get config
config = get_config()
# config = config.loc[["IK91", "IK93", "IK101", "IK103"]]

# %% new: focus on low-flow data and good stable periods. Define settings per strang to get 'best' datapoints without too few datapoints.
config = config.loc[["IK91", "IK93", "IK101", "IK103"]]


# # %% ESTIMATE BETA PER STRANG (ROBUST VERSION)

# beta_results = {}

# for strang, c in config.iterrows():

#     print(f"\n=== Estimating beta for {strang} ===")

#     # -------------------------------------
#     # LOAD
#     # -------------------------------------
#     df_fp = data_dir / "Merged" / f"{strang}.feather"
#     df = pd.read_feather(df_fp).set_index("Datum").sort_index()

#     # -------------------------------------
#     # FILTER BAD DATA
#     # -------------------------------------
#     bad = get_false_measurements(
#         df,
#         c,
#         extend_hours=1,
#         include_rules=["Unrealistic flow", "Tijdens spuien"],
#     )
#     df.loc[bad] = np.nan

#     # -------------------------------------
#     # WINDOWING (same as main model)
#     # -------------------------------------
#     groups = pd.Grouper(freq=f"{window_hours}h")

#     period_stats = []

#     for t, g in df.groupby(groups):

#         if len(g) < min_points:
#             continue

#         Q_range = g["Q"].quantile(0.95) - g["Q"].quantile(0.05)
#         if Q_range > Q_range_threshold:
#             continue

#         Q = g["Q"].median(skipna=True)
#         P = g["P"].median(skipna=True)

#         if np.isnan(Q) or np.isnan(P):
#             continue

#         if Q < 0:
#             continue

#         period_stats.append({
#             "time": t,
#             "Q": Q,
#             "P": P,
#         })

#     if len(period_stats) < 20:
#         print("Too few datapoints → skipping")
#         continue

#     dfp = pd.DataFrame(period_stats).set_index("time").sort_index()

#     print(f"Number of dfp points: {len(dfp)}")

#     # -------------------------------------
#     # BETA ESTIMATION
#     # -------------------------------------
#     beta_list = []
#     time_list = []

#     Q_scale = 200.0

#     for t in dfp.index:

#         window_start = t - pd.Timedelta(days=rolling_days)
#         g = dfp.loc[window_start:t]

#         if len(g) < min_points_fit:
#             continue

#         Q_min = g["Q"].min()
#         Q_max = g["Q"].max()

#         # ✅ LESS strict than main script
#         if (Q_max - Q_min) < 20:
#             continue

#         X = g["Q"].values
#         P_obs = g["P"].values
#         Xn = X / Q_scale

#         def residuals(p):
#             P0, R_lin, R_nl, beta = p
#             model = P0 - (R_lin * Xn + R_nl * Xn**beta)
#             return model - P_obs

#         try:
#             res = least_squares(
#                 residuals,
#                 x0=[
#                     np.median(P_obs),
#                     0.01,
#                     0.01,
#                     2.0
#                 ],
#                 bounds=([
#                     np.min(P_obs) - 2,
#                     0,
#                     0,
#                     1.0   # lower bound
#                 ], [
#                     np.max(P_obs) + 2,
#                     10,
#                     10,
#                     5.0   # upper bound
#                 ]),
#                 loss="soft_l1",
#                 max_nfev=80
#             )
#         except:
#             continue

#         # ✅ CHECK FIT QUALITY
#         if not res.success:
#             continue

#         beta = res.x[3]

#         beta_list.append(beta)
#         time_list.append(t)

#     if len(beta_list) < 10:
#         print("Too few beta estimates → skipping")
#         continue

#     beta_series = pd.Series(beta_list, index=time_list)

#     # -------------------------------------
#     # ✅ FILTER BAD BETA VALUES
#     # -------------------------------------

#     # 1. remove boundary hits
#     beta_filtered = beta_series[
#         (beta_series > 1.2) &
#         (beta_series < 4.5)
#     ]

#     # 2. remove extreme outliers (robust trimming)
#     if len(beta_filtered) > 10:
#         q10 = beta_filtered.quantile(0.1)
#         q90 = beta_filtered.quantile(0.9)

#         beta_filtered = beta_filtered[
#             (beta_filtered >= q10) &
#             (beta_filtered <= q90)
#         ]

#     # -------------------------------------
#     # SUMMARY STATISTICS
#     # -------------------------------------
#     beta_raw_median = beta_series.median()

#     if len(beta_filtered) > 0:
#         beta_final = beta_filtered.median()
#     else:
#         beta_final = np.nan

#     print(f"\n--- Beta stats for {strang} ---")
#     print(f"Raw median         : {beta_raw_median:.2f}")
#     print(f"Filtered median    : {beta_final:.2f}")
#     print(f"Used / total       : {len(beta_filtered)} / {len(beta_series)}")

#     if len(beta_filtered) < 0.5 * len(beta_series):
#         print("⚠️ WARNING: many beta values filtered → low identifiability")

#     beta_results[strang] = beta_filtered

#     # -------------------------------------
#     # PLOTTING
#     # -------------------------------------
#     fig, axes = plt.subplots(2, 1, figsize=(10, 6))

#     # histogram
#     ax1 = axes[0]
#     ax2 = ax1.twinx()

#     # --- ALL values (left axis)
#     ax1.hist(
#         beta_series,
#         bins=30,
#         alpha=0.25,
#         color="C0",
#         label="all"
#     )

#     # --- FILTERED values (right axis)
#     ax2.hist(
#         beta_filtered,
#         bins=30,
#         alpha=0.7,
#         color="C1",
#         label="filtered"
#     )

#     # --- median lines
#     ax1.axvline(
#         beta_raw_median,
#         color="C0",
#         linestyle="--",
#         linewidth=2,
#         label="raw median"
#     )

#     ax2.axvline(
#         beta_final,
#         color="red",
#         linewidth=2,
#         label="filtered median"
#     )

#     # --- labels
#     ax1.set_xlabel("beta")
#     ax1.set_ylabel("count (all)", color="C0")
#     ax2.set_ylabel("count (filtered)", color="C1")

#     # --- styling
#     ax1.grid(True)

#     # --- combine legends
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()

#     ax1.legend(
#         lines1 + lines2,
#         labels1 + labels2,
#         loc="upper right"
#     )

#     # --- title
#     ax1.set_title(f"{strang} – Beta distribution")


#     # time series
#     axes[1].plot(beta_series.index, beta_series, alpha=0.3, label="all")
#     axes[1].plot(beta_filtered.index, beta_filtered, alpha=0.8, label="filtered")

#     axes[1].axhline(beta_final, color="red", label="filtered median")

#     axes[1].set_title("Beta over time")
#     axes[1].set_ylabel("beta")
#     axes[1].grid(True)
#     axes[1].legend()

#     plt.tight_layout()

#     plt.savefig(
#         os.path.join(res_folder, f"{strang}_beta_estimation.png"),
#         dpi=300
#     )

#     plt.close()

# print("\n✅ Beta estimation complete.")


# below function gets stable low flow periods as reference for
# pt10ofset. It deletes the first mintimelowflow and the last
# low flow value from each period with stable low flow.
# what is low flow? => determined by lowflowcutoff
# -------------------------------------
# FUNCTION: stable low-flow detection
# -------------------------------------
def get_stable_low_flow(df, lowflowcutoff, mintimelowflow):

    low_mask = df["Q"] < lowflowcutoff
    low_idx = df.index[low_mask]

    if len(low_idx) == 0:
        return df.iloc[[]]

    groups = []
    current_block = [low_idx[0]]

    for i in range(1, len(low_idx)):
        if (low_idx[i] - low_idx[i - 1]) <= pd.Timedelta("1.5h"):
            current_block.append(low_idx[i])
        else:
            groups.append(current_block)
            current_block = [low_idx[i]]

    groups.append(current_block)

    keep_indices = []

    for block in groups:
        # require enough length AFTER trimming
        if len(block) < (mintimelowflow + 2):
            continue

        # remove transient start + last point
        trimmed = block[mintimelowflow:-1]

        keep_indices.extend(trimmed)

    return df.loc[keep_indices]


# %% loop
for strang, c in config.iterrows():
    print(f"\nProcessing {strang}")

    # -------------------------------------
    # LOAD
    # -------------------------------------
    df_fp = data_dir / "Merged" / f"{strang}.feather"
    df = pd.read_feather(df_fp).set_index("Datum").sort_index()

    # -------------------------------------
    # FILTER BAD DATA
    # -------------------------------------
    bad = get_false_measurements(
        df,
        c,
        extend_hours=1,
        include_rules=["Unrealistic flow", "Tijdens spuien"],
    )
    df.loc[bad] = np.nan

    # -------------------------------------
    # WINDOWING
    # -------------------------------------
    groups = pd.Grouper(freq=f"{window_hours}h")

    period_stats = []

    for t, g in df.groupby(groups):
        if len(g) < min_points:
            continue

        Q_range = g["Q"].quantile(0.95) - g["Q"].quantile(0.05)
        if Q_range > Q_range_threshold:
            continue

        Q = g["Q"].median(skipna=True)
        P = g["P"].median(skipna=True)

        if np.isnan(Q) or np.isnan(P):
            continue

        if Q < 0:
            continue

        period_stats.append({
            "time": t,
            "Q": Q,
            "P": P,
        })

    if len(period_stats) < 10:
        continue

    dfp = pd.DataFrame(period_stats).set_index("time").sort_index()

    # -------------------------------------
    # MODEL SETTINGS
    # -------------------------------------
    beta_fixed = beta_settings.get(strang)
    Q_scale = 200.0

    # -------------------------------------
    # ROLLING FIT (NOW CENTERED)
    # -------------------------------------
    results = []
    prev_params = None

    for t in dfp.index:
        window_start = t - pd.Timedelta(days=rolling_days)
        g = dfp.loc[window_start:t]

        if len(g) < min_points_fit:
            continue

        Q_min = g["Q"].min()
        Q_max = g["Q"].max()

        if (Q_max - Q_min) < 20:
            continue

        X = g["Q"].values
        P_obs = g["P"].values
        Xn = X / Q_scale

        weights = 1 / (X + 20)

        def residuals(p):
            P0, R_lin, R_nl = p
            model = P0 - (R_lin * Xn + R_nl * Xn**beta_fixed)
            return weights * (model - P_obs)

        if prev_params is None:
            x0 = [np.median(P_obs), 0.01, 0.01]
        else:
            x0 = prev_params

        try:
            res = least_squares(
                residuals,
                x0,
                bounds=([np.min(P_obs) - 2, 0, 0], [np.max(P_obs) + 2, 10, 10]),
                loss="soft_l1",
                max_nfev=80,
            )
        except:
            continue

        P0, R_lin, R_nl = res.x
        prev_params = res.x

        # OFFSET
        offsets = g["P"] + (R_lin * (g["Q"] / Q_scale) + R_nl * ((g["Q"] / Q_scale) ** beta_fixed))

        P_offset = np.median(offsets)

        # ✅ CENTER TIME HERE
        mid_time = t - pd.Timedelta(days=rolling_days / 2)

        results.append({
            "time": mid_time,
            "R_lin": R_lin,
            "R_nl": R_nl,
            "beta": beta_fixed,
            "P_offset": P_offset,
        })

    if len(results) == 0:
        continue

    dfr = pd.DataFrame(results).set_index("time").sort_index()

    # -------------------------------------
    # SMOOTHING (CENTERED)
    # -------------------------------------
    rolling_days_smooth = 365

    R_lin_roll = dfr["R_lin"].rolling(f"{rolling_days_smooth}D", center=True).median()
    R_nl_roll = dfr["R_nl"].rolling(f"{rolling_days_smooth}D", center=True).median()
    P_offset_roll = dfr["P_offset"].rolling(f"{rolling_days_smooth}D", center=True).median()

    # -------------------------------------
    # LOW FLOW
    # -------------------------------------
    low_flow = get_stable_low_flow(df, lowflowcutoff, mintimelowflow)

    if len(low_flow) > 0:
        low_flow_roll = low_flow["P"].rolling(f"{rolling_days_smooth}D", center=True).median()
    else:
        low_flow_roll = None

    # -------------------------------------
    # MAIN PLOT (FULLY CONSISTENT)
    # -------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # -------------------------------------
    # R_lin
    # -------------------------------------
    axes[0].scatter(dfr.index, dfr["R_lin"], s=10, alpha=0.5, color="C0", label="R_lin")
    axes[0].plot(R_lin_roll.index, R_lin_roll, color="C0", linewidth=2)
    axes[0].set_ylabel("R_lin")
    axes[0].grid(True)
    axes[0].legend()

    # -------------------------------------
    # R_nl
    # -------------------------------------
    axes[1].scatter(dfr.index, dfr["R_nl"], s=10, alpha=0.5, color="C1", label="R_nl")
    axes[1].plot(R_nl_roll.index, R_nl_roll, color="C1", linewidth=2)
    axes[1].set_ylabel("R_nl")
    axes[1].grid(True)
    axes[1].legend()

    # -------------------------------------
    # P_offset + comparison
    # -------------------------------------

    # ✅ 1. WELL HEAD (BOTTOM LAYER)
    axes[2].scatter(low_flow.index, low_flow["gws0"], s=6, alpha=0.5, color="blue", label="Well head (gws0)")

    # ✅ rolling well head
    low_flow_gws0_roll = low_flow["gws0"].rolling("365D", center=True).median()

    axes[2].plot(low_flow_gws0_roll.index, low_flow_gws0_roll, color="blue", linewidth=2)

    # ✅ 2. PT10 LOW-FLOW (MIDDLE LAYER)
    axes[2].scatter(low_flow.index, low_flow["P"], s=6, alpha=0.5, color="green", label="PT10 (low-flow)")

    if low_flow_roll is not None:
        axes[2].plot(low_flow_roll.index, low_flow_roll, color="green", linewidth=2)

    # ✅ 3. MODEL (TOP LAYER)
    axes[2].scatter(dfr.index, dfr["P_offset"], s=10, alpha=0.7, color="black", label="PT10 (model)")

    axes[2].plot(P_offset_roll.index, P_offset_roll, color="black", linewidth=2)

    # -------------------------------------
    # FORMATTING
    # -------------------------------------
    axes[2].set_ylabel("Head (m NAP)")
    axes[2].grid(True)

    axes[2].legend(loc="upper left", fontsize="small", ncol=2)

    plt.xlabel("Time")
    fig.suptitle(f"{strang} – PT10 vs well head (low-flow + model)")

    plt.tight_layout()

    plt.savefig(os.path.join(res_folder, f"{strang}_rolling.png"), dpi=300)

    plt.close()

    # -------------------------------------
    # RANDOM FIT VISUALISATION (PER YEAR)
    # -------------------------------------
    print("\n--- RANDOM FIT VISUALISATION (PER YEAR) ---")

    n_per_year = 5

    # group results per year
    dfr_valid = dfr.dropna().copy()
    dfr_valid["year"] = dfr_valid.index.year

    years = sorted(dfr_valid["year"].unique())

    n_rows = len(years)
    n_cols = n_per_year

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False, sharey=False)

    if n_rows == 1:
        axes = np.array([axes])

    for i, year in enumerate(years):
        dfr_year = dfr_valid[dfr_valid["year"] == year]

        if len(dfr_year) < n_per_year:
            sample_idx = dfr_year.index
        else:
            sample_idx = dfr_year.sample(n_per_year, random_state=42).index

        for j in range(n_cols):
            ax = axes[i, j]

            if j >= len(sample_idx):
                ax.axis("off")
                continue

            t = sample_idx[j]

            # ✅ centered window (FIXED)
            window_start = t - pd.Timedelta(days=rolling_days / 2)
            window_end = t + pd.Timedelta(days=rolling_days / 2)
            g = dfp.loc[window_start:window_end]

            row = dfr.loc[t]

            R_lin = row["R_lin"]
            R_nl = row["R_nl"]
            P_offset = row["P_offset"]

            # data
            ax.scatter(g["Q"], g["P"], s=10, alpha=0.4)

            # model curve
            Q_line = np.linspace(0, g["Q"].max() * 1.1, 100)
            Qn_line = Q_line / Q_scale

            P_line = P_offset - (R_lin * Qn_line + R_nl * Qn_line**beta_fixed)

            ax.plot(Q_line, P_line, color="red")
            ax.scatter([0], [P_offset], color="black", s=50)

            ax.set_xlabel("Q")
            ax.set_ylabel("P")
            ax.grid(True)

            ax.set_title(t.strftime("%Y-%m-%d"))

        axes[i, 0].set_ylabel(f"{year}\nP")

    fig.suptitle(f"{strang} – Random rolling fits (CENTERED)")

    plt.tight_layout()

    plt.savefig(os.path.join(res_folder, f"{strang}_random_fits_per_year.png"), dpi=300)

    plt.close()

print("\nDone.")
