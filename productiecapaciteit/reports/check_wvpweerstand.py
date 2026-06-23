# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta

from productiecapaciteit import data_dir, results_dir
from productiecapaciteit.src.strang_analyse_fun2 import (
    get_config,
    get_false_measurements,
)
from productiecapaciteit.src.weerstand_pandasaccessors import (
    WellResistanceAccessor,
    WvpResistanceAccessor,
    LeidingResistanceAccessor,
)

# --------------------------------------------------
# OUTPUT FOLDER
# --------------------------------------------------
res_folder = r"C:\PythonScripts\Repositories\pwn-drinking-water-production\python-pwn-productiecapaciteit-infiltratiegebieden\productiecapaciteit\results\Wvpweerstand_check"
os.makedirs(res_folder, exist_ok=True)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
config = get_config()

filterweerstand_fp = results_dir / "Filterweerstand" / "Filterweerstand_modelcoefficienten.xlsx"

# --------------------------------------------------
# LOOP
# --------------------------------------------------
for strang, c in config.iterrows():

    print(f"\n=== VALIDATING {strang} ===")

    df_fp = data_dir / "Merged" / f"{strang}.feather"
    df = pd.read_feather(df_fp).set_index("Datum")

    # -----------------------------
    # FILTER BAD DATA
    # -----------------------------
    include_rules = [
        "Unrealistic flow",
        "Tijdens spuien",
        "Tijdens proppen",
        "Little flow",
    ]

    bad = get_false_measurements(df, c, extend_hours=10, include_rules=include_rules)
    df.loc[bad] = np.nan

    # -----------------------------
    # LOAD FILTER MODEL
    # -----------------------------
    df_a_filter = pd.read_excel(filterweerstand_fp, sheet_name=strang)

    # -----------------------------
    # RECONSTRUCT OUTER WELL HEAD
    # -----------------------------
    q_per_well = df.Q / c.nput

    p_omstorting_reconstructed = df.gws0 - df_a_filter.wel.dp_model(df.index, q_per_well)
    p_omstorting = df.gws1.where(~df.gws1.isna(), p_omstorting_reconstructed)

    df["p_omstorting"] = p_omstorting
    df["dP_wvp"] = df["p_omstorting"] - df["pandpeil"]

    # --------------------------------------------------
    # 1. SENSOR RECONSTRUCTION CHECK
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    mask = df.gws1.notna()

    ax.scatter(df.gws1[mask], p_omstorting_reconstructed[mask], s=4, alpha=0.5)
    ax.plot(
        [df.gws1.min(), df.gws1.max()],
        [df.gws1.min(), df.gws1.max()],
        "k--"
    )

    ax.set_xlabel("gws1 (measured)")
    ax.set_ylabel("gws1 (reconstructed)")
    ax.set_title(f"{strang} – gws1 reconstruction")

    plt.tight_layout()
    plt.savefig(os.path.join(res_folder, f"{strang}_01_gws1_check.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    # 2. HEAD COMPARISON
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.scatter(df.index, df["pandpeil"], s=2, label="lake (pandpeil)")
    ax.scatter(df.index, df["p_omstorting"], s=2, label="well outer")

    ax.set_ylabel("Head (m NAP)")
    ax.set_title(f"{strang} – Lake vs well outer head")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(res_folder, f"{strang}_02_heads.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    # 3. LINEARITY CHECK
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(df.Q, df.dP_wvp, s=2, alpha=0.3)

    ax.set_xlabel("Q (m3/h)")
    ax.set_ylabel("dP_wvp (m)")
    ax.set_title(f"{strang} – dP vs Q")

    plt.tight_layout()
    plt.savefig(os.path.join(res_folder, f"{strang}_03_dp_vs_q.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    # 4. NORMALIZED RESISTANCE
    # --------------------------------------------------
    df["dP_over_Q"] = df["dP_wvp"] / df["Q"]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.scatter(df.index, df["dP_over_Q"], s=2, alpha=0.3)

    ax.set_title(f"{strang} – dP/Q over time")
    ax.set_ylabel("Resistance (m/(m3/h))")

    plt.tight_layout()
    plt.savefig(os.path.join(res_folder, f"{strang}_04_dp_over_q.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    # 5. FLOW STABILITY METRIC
    # --------------------------------------------------
    window = int(timedelta(days=2) / (df.index[1] - df.index[0]))

    dQ_rel = np.abs(df.Q.diff() / df.Q)
    dQ_rol = dQ_rel.rolling(window=window).max()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df.index, dQ_rol)

    ax.set_title(f"{strang} – flow stability")
    ax.set_ylabel("max |dQ/Q|")

    plt.tight_layout()
    plt.savefig(os.path.join(res_folder, f"{strang}_05_flow_stability.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    # 6. FILTERED DATA RESULT
    # --------------------------------------------------
    percentage = 0.20
    n_keep = int((1 - percentage) * len(df))

    stable_idx = dQ_rol.nsmallest(n_keep).index
    df_filtered = df.loc[stable_idx]

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(df_filtered.Q, df_filtered.dP_wvp, s=5, alpha=0.5)

    ax.set_xlabel("Q")
    ax.set_ylabel("dP_wvp")
    ax.set_title(f"{strang} – filtered dP vs Q")

    plt.tight_layout()
    plt.savefig(os.path.join(res_folder, f"{strang}_06_filtered_dp_vs_q.png"), dpi=300)
    plt.close()

print("\n✅ All validation plots saved.")

# %%
