import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import k0

from productiecapaciteit import data_dir, results_dir
from productiecapaciteit.src.strang_analyse_fun2 import (
    get_config,
    get_false_measurements,
)
from productiecapaciteit.src.weerstand_pandasaccessors import (
    LeidingResistanceAccessor,  # noqa: F401
    WellResistanceAccessor,  # noqa: F401
)
from productiecapaciteit.src.wvp_transient_funs import (
    build_multiwell_geometry,
    get_temp,
    infer_lower_timestep,
    objective,
    visc_ratio,
)

CONFIG_FN = "strang_props7.csv"
STRANG = "IK102"

HOURS_PER_DAY = 24.0

# Hantush parameters are expressed in days and meters. Flow measurements in this
# repository are m3/h, so the forcing is converted to m3/d before modelling.
WELL_RADIUS_M = 0.2
LEAKAGE_RESISTANCE_D = 200.0
STORAGE_COEFFICIENT = 0.2
KD_SOLVE_MIN_M2_PER_D = 1.0
KD_SOLVE_MAX_M2_PER_D = 1_000.0

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
        raise ValueError("Median aquifer drawdown is not positive; check head columns and sign convention")
    return dfm


def get_reference_wvp_resistance(dfm, df_a_wvp):
    index = pd.DatetimeIndex(dfm.index)
    days_since_offset = (index - pd.Timestamp(df_a_wvp.offset_datum)) / pd.Timedelta("1D")
    resistance_at_reference_temp = df_a_wvp.offset + df_a_wvp.slope * days_since_offset
    resistance = -np.asarray(resistance_at_reference_temp, dtype=float)
    if not np.isfinite(resistance).all():
        raise ValueError("Reference WVP resistance contains NaN or infinite values")
    if np.any(resistance <= 0.0):
        raise ValueError("Reference WVP resistance must be positive after sign conversion")
    return resistance


def get_wvp_viscosity_ratio(dfm, df_a_wvp):
    if df_a_wvp.method == "Niet":
        ratio = np.ones(dfm.index.size, dtype=float)
    elif df_a_wvp.method == "sin":
        temp = get_temp(
            dfm.index,
            df_a_wvp.temp_mean,
            df_a_wvp.temp_delta,
            df_a_wvp.time_offset,
            return_series=False,
        )
        ratio = visc_ratio(temp, temp_ref=df_a_wvp.temp_ref)
    else:
        raise ValueError(f"Unsupported WVP temperature method: {df_a_wvp.method}")

    if not np.isfinite(ratio).all():
        raise ValueError("WVP viscosity ratio contains NaN or infinite values")
    if np.any(ratio <= 0.0):
        raise ValueError("WVP viscosity ratio must be positive")
    return ratio


def get_steady_wvp_resistance(dfm, df_a_wvp):
    return get_reference_wvp_resistance(dfm, df_a_wvp) * get_wvp_viscosity_ratio(
        dfm,
        df_a_wvp,
    )


def get_steady_wvp_drawdown(dfm, df_a_wvp):
    return get_steady_wvp_resistance(dfm, df_a_wvp) * dfm.Q.to_numpy(dtype=float)


def steady_multiwell_resistance_from_kd(kD, multiwell, ci):
    kD = np.asarray(kD, dtype=float)
    leakage_factor = np.sqrt(kD * LEAKAGE_RESISTANCE_D)
    well_function_sum = np.zeros_like(kD, dtype=float)
    for multiplicity, normalized_distance in multiwell:
        distance = normalized_distance * WELL_RADIUS_M
        well_function_sum += multiplicity * 2.0 * k0(distance / leakage_factor)
    return HOURS_PER_DAY / ci.nput * well_function_sum / (4.0 * np.pi * kD)


def solve_steady_multiwell_kd(target_resistance, multiwell, ci):
    def residual(kD):
        return float(steady_multiwell_resistance_from_kd(kD, multiwell, ci) - target_resistance)

    lower = KD_SOLVE_MIN_M2_PER_D
    upper = KD_SOLVE_MAX_M2_PER_D
    if residual(lower) < 0.0:
        raise ValueError(
            "Steady multiwell resistance is below target at the lower kD bound: "
            f"kD={lower:g} m2/d, target={target_resistance:.4g}"
        )
    if residual(upper) > 0.0:
        raise ValueError(
            "Steady multiwell resistance is above target at the upper kD bound: "
            f"kD={upper:g} m2/d, target={target_resistance:.4g}"
        )
    return brentq(residual, lower, upper, xtol=1e-10, rtol=1e-10)


def get_calibrated_kd_from_steady_wvp(dfm, df_a_wvp, ci, multiwell):
    reference_resistance = get_reference_wvp_resistance(dfm, df_a_wvp)
    kD_reference_temp = np.array(
        [solve_steady_multiwell_kd(target_resistance, multiwell, ci) for target_resistance in reference_resistance],
        dtype=float,
    )
    kD = kD_reference_temp / get_wvp_viscosity_ratio(dfm, df_a_wvp)
    if not np.isfinite(kD).all():
        raise ValueError("Calibrated kD contains NaN or infinite values")
    if np.any(kD <= 0.0):
        raise ValueError("Calibrated kD must be positive")
    return kD


def build_pextra(dfm, ci, df_a_wvp):
    multiwell, multiwell_counts = build_multiwell_geometry(
        ci.dx_tussenputten,
        ci.r_mirrorwel,
        ci.nput,
        target_well_index=TARGET_WELL_INDEX,
        distance_scale=1.0 / WELL_RADIUS_M,
        include_self=True,
        self_distance=1.0,
    )

    q_obs_m3d_per_well = dfm.Q.to_numpy(dtype=float) / ci.nput * HOURS_PER_DAY
    if not np.isfinite(q_obs_m3d_per_well).all():
        raise ValueError("Q_obs contains NaN or infinite values")

    kD = get_calibrated_kd_from_steady_wvp(dfm, df_a_wvp, ci, multiwell)

    return {
        "index": dfm.index,
        "drawdown_obs": dfm.drawdown_aquifer.to_numpy(dtype=float),
        "Q_obs": q_obs_m3d_per_well,
        "kD": kD,
        "dt_lower": infer_lower_timestep(dfm.index),
        "multiwell_contains_r_self": True,
        "multiwell": multiwell,
        "multiwell_counts": multiwell_counts,
        "log_multiwell": False,
        "frac_step_max": 0.95,
        "initial_condition": "steady",
        "tmax_days_cap": None,
    }


def get_initial_params():
    alpha = (WELL_RADIUS_M**2 * STORAGE_COEFFICIENT / 4.0) ** 0.5
    beta = (1.0 / (LEAKAGE_RESISTANCE_D * STORAGE_COEFFICIENT)) ** 0.5
    return [alpha, beta]


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


def plot_result(
    index,
    drawdown_obs,
    drawdown_steady_wvp,
    drawdown_multiwell,
    drawdown_single_well,
    kD,
    strang,
):
    fig, (ax, ax_kd) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax.plot(index, drawdown_obs, label="observed aquifer drawdown")
    ax.plot(index, drawdown_steady_wvp, label="calibrated steady WVP model with temperature")
    ax.plot(index, drawdown_multiwell, label="multiwell + image wells")
    ax.plot(index, drawdown_single_well, label="single well")
    ax.set_ylabel("Drawdown (m)")
    ax.legend()
    ax_kd.plot(index, kD, label="kD from reference coefficients with temperature")
    ax_kd.set_ylabel("kD (m2/d)")
    ax_kd.legend()
    fig.tight_layout()

    output_dir = results_dir / "Wvptweerstand"
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
    pextra = build_pextra(dfm, ci, df_a_wvp)

    args = get_initial_params()

    single_well_pextra = {**get_single_well_pextra(pextra), "log_multiwell": True}
    multiwell_pextra = {**pextra, "log_multiwell": True}

    drawdown_single_well = objective(args, return_result=True, **single_well_pextra)
    drawdown_multiwell = objective(args, return_result=True, **multiwell_pextra)
    drawdown_obs = pextra["drawdown_obs"]
    drawdown_steady_wvp = get_steady_wvp_drawdown(dfm, df_a_wvp)

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
        drawdown_steady_wvp,
        drawdown_multiwell,
        drawdown_single_well,
        pextra["kD"],
        STRANG,
    )
    print(f"Saved transient WVP drawdown plot to {output_fp}")


if __name__ == "__main__":
    main()
