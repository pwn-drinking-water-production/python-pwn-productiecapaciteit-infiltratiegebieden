import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from productiecapaciteit import data_dir, results_dir
from productiecapaciteit.src.strang_analyse_fun2 import (
    get_config,
    get_false_measurements,
)
from productiecapaciteit.src.weerstand_pandasaccessors import (
    LeidingResistanceAccessor,  # noqa: F401
    WellResistanceAccessor,  # noqa: F401
    WvpResistanceAccessor,  # noqa: F401
)
from productiecapaciteit.src.wvp_transient_funs import (
    build_multiwell_geometry,
    infer_lower_timestep,
    objective,
)


CONFIG_FN = "strang_props7.csv"
STRANG = "IK102"

HOURS_PER_DAY = 24.0

# Hantush parameters are expressed in days and meters. Flow measurements in this
# repository are m3/h, so the forcing is converted to m3/d before modelling.
WELL_RADIUS_M = 0.6
KD0_M2_PER_D = 200.0
LEAKAGE_RESISTANCE_D = 50.0
STORAGE_COEFFICIENT = 0.2

RESAMPLE_FREQUENCY = "12h"
BAD_DATA_RULES = ["Unrealistic flow"]

# None selects one of the center wells. Set a zero-based index for a specific
# well if the observation point is known.
TARGET_WELL_INDEX = None


def load_wvp_coefficients(strang):
    wvpweerstand_fp = results_dir / "Wvpweerstand" / "Wvpweerstand_modelcoefficienten.xlsx"
    return pd.read_excel(wvpweerstand_fp, sheet_name=strang, index_col=0).squeeze("columns")


def load_filter_coefficients(strang):
    filterweerstand_fp = results_dir / "Filterweerstand" / "Filterweerstand_modelcoefficienten.xlsx"
    return pd.read_excel(filterweerstand_fp, sheet_name=strang)


def load_observations(strang, ci):
    df_fp = os.path.join(data_dir, "Merged", f"{strang}.feather")
    df = pd.read_feather(df_fp)
    df["Datum"] = pd.to_datetime(df["Datum"])
    df.set_index("Datum", inplace=True)

    untrusted_measurements = get_false_measurements(
        df,
        ci,
        extend_hours=1,
        include_rules=BAD_DATA_RULES,
    )
    df = df.loc[~np.asarray(untrusted_measurements)].copy()
    df = df.loc[np.isfinite(df.Q)]

    dfm = df.resample(RESAMPLE_FREQUENCY, label="right").mean()
    dfm = dfm.loc[np.isfinite(dfm.Q)].copy()
    if dfm.empty:
        raise ValueError(f"No finite flow observations remain for {strang}")
    return dfm


def add_aquifer_drawdown_observations(dfm, ci, df_a_filter):
    q_per_well_m3h = dfm.Q / ci.nput
    p_omstorting_reconstructed = dfm.gws0 - df_a_filter.wel.dp_model(dfm.index, q_per_well_m3h)
    dfm["p_omstorting"] = dfm.gws1.where(dfm.gws1.notna(), p_omstorting_reconstructed)
    dfm["drawdown_aquifer"] = dfm.pandpeil - dfm.p_omstorting

    valid_drawdown = np.isfinite(dfm.drawdown_aquifer)
    if valid_drawdown.sum() == 0:
        raise ValueError("No finite aquifer drawdown observations are available")
    if np.nanmedian(dfm.drawdown_aquifer) <= 0.0:
        raise ValueError(
            "Median aquifer drawdown is not positive; check head columns and sign convention"
        )
    return dfm


def build_pextra(dfm, ci):
    multiwell, multiwell_counts = build_multiwell_geometry(
        ci.dx_tussenputten,
        ci.dx_mirrorwell,
        ci.nput,
        target_well_index=TARGET_WELL_INDEX,
        distance_scale=1.0 / WELL_RADIUS_M,
        include_self=True,
        self_distance=1.0,
    )

    q_obs_m3d_per_well = dfm.Q.to_numpy(dtype=float) / ci.nput * HOURS_PER_DAY
    if not np.isfinite(q_obs_m3d_per_well).all():
        raise ValueError("Q_obs contains NaN or infinite values")

    temp_ref = float(dfm.gwt0.median())
    if not np.isfinite(temp_ref):
        raise ValueError("Cannot determine finite reference groundwater temperature")

    return {
        "index": dfm.index,
        "drawdown_obs": dfm.drawdown_aquifer.to_numpy(dtype=float),
        "Q_obs": q_obs_m3d_per_well,
        "temp_ref": temp_ref,
        "dt_lower": infer_lower_timestep(dfm.index),
        "multiwell_contains_r_self": True,
        "multiwell": multiwell,
        "multiwell_counts": multiwell_counts,
        "log_multiwell": False,
        "frac_step_max": 0.95,
        "initial_condition": "steady",
        "tmax_days_cap": None,
    }


def get_initial_params(df_a_wvp):
    alpha = (WELL_RADIUS_M**2 * STORAGE_COEFFICIENT / 4.0) ** 0.5
    beta = (1.0 / (LEAKAGE_RESISTANCE_D * STORAGE_COEFFICIENT)) ** 0.5
    return [
        alpha,
        beta,
        KD0_M2_PER_D,
        df_a_wvp.temp_delta,
        df_a_wvp.time_offset,
    ]


def get_single_well_pextra(pextra):
    return {
        **pextra,
        "multiwell": [(1.0, 1.0)],
        "multiwell_counts": {
            "self_wells": 1,
            "neighbor_well_terms": 0,
            "neighbor_wells": 0,
            "self_mirrorwell_terms": 0,
            "self_mirrorwells": 0,
            "neighbor_mirrorwell_terms": 0,
            "neighbor_mirrorwells": 0,
        },
    }


def plot_result(index, drawdown_obs, drawdown_multiwell, drawdown_single_well, strang):
    fig, ax = plt.subplots()
    ax.plot(index, drawdown_obs, label="observed aquifer drawdown")
    ax.plot(index, drawdown_multiwell, label="multiwell + image wells")
    ax.plot(index, drawdown_single_well, label="single well")
    ax.set_ylabel("Drawdown (m)")
    ax.legend()
    fig.tight_layout()

    output_dir = results_dir / "Wvpweerstand_transient"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_fp = output_dir / f"wvp_transient_{strang}.png"
    fig.savefig(output_fp, dpi=300)
    plt.close(fig)
    return output_fp


def main():
    config = get_config(CONFIG_FN)
    ci = config.loc[STRANG]

    df_a_wvp = load_wvp_coefficients(STRANG)
    df_a_filter = load_filter_coefficients(STRANG)
    dfm = load_observations(STRANG, ci)
    dfm = add_aquifer_drawdown_observations(dfm, ci, df_a_filter)
    pextra = build_pextra(dfm, ci)

    args = get_initial_params(df_a_wvp)

    single_well_pextra = {**get_single_well_pextra(pextra), "log_multiwell": True}
    multiwell_pextra = {**pextra, "log_multiwell": True}

    drawdown_single_well = objective(args, return_result=True, **single_well_pextra)
    drawdown_multiwell = objective(args, return_result=True, **multiwell_pextra)
    drawdown_obs = pextra["drawdown_obs"]

    if drawdown_single_well.shape != drawdown_multiwell.shape:
        raise ValueError("Single-well and multiwell model outputs have different shapes")
    if drawdown_multiwell.shape != drawdown_obs.shape:
        raise ValueError("Modeled drawdown and observed drawdown have different shapes")
    if not np.isfinite(drawdown_single_well).all():
        raise ValueError("Single-well model contains NaN or infinite values")
    if not np.isfinite(drawdown_multiwell).all():
        raise ValueError("Multiwell model contains NaN or infinite values")

    output_fp = plot_result(
        pextra["index"],
        drawdown_obs,
        drawdown_multiwell,
        drawdown_single_well,
        STRANG,
    )
    print(f"Saved transient WVP drawdown plot to {output_fp}")


if __name__ == "__main__":
    main()
