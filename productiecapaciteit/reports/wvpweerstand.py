"""
Calculate the model coefficients for the WVP resistance model.

TODO: Use mutiple Hantush wells to calculate the model coefficients for the WVP resistance model.
"""

import logging
from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from productiecapaciteit import data_dir, plot_styles_dir, results_dir
from productiecapaciteit.src.strang_analyse_fun2 import (
    get_config,
    get_false_measurements,
)
from productiecapaciteit.src.weerstand_pandasaccessors import (
    LeidingResistanceAccessor,  # noqa: F401
    WellResistanceAccessor,  # noqa: F401
    WvpResistanceAccessor,  # noqa: F401
)

res_folder = results_dir / "Wvpweerstand"
logger_handler = logging.FileHandler(res_folder / "Wvpweerstandcoefficient.log", mode="w")
stdout = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logger_handler, stdout],
)

plt.style.use(plot_styles_dir / "unhcrpyplotstyle.mplstyle")
plt.style.use(plot_styles_dir / "types" / "line.mplstyle")


def get_wvp_slope_per_year2(index, flow, dp, offset_datum, temp_wvp):
    """
    Compute the aquifer resistance (WVP) and adjust for temperature effects using a two-step least squares optimization process.

    Parameters
    ----------
    index : array-like
        Array representing time or index values for the observations.
    flow : array-like
        Array of flow measurements corresponding to the observations.
    dp : array-like
        Array representing differential pressures for the observations.
    offset_datum : object
        Reference datum used for offset calculations (e.g., a baseline date or measurement).
    temp_wvp : array-like
        Array of temperature values related to water vapor pressure used in the modeling process.

    Returns
    -------
    pandas.Series or None
        A pandas Series containing the optimized parameters and model metadata with the following keys:
            - "temp_mean": Mean temperature computed from temp_wvp.
            - "method": Indicates the method used ("Niet" for the initial fit and "sin" for the temperature-adjusted fit).

    Notes
    -----
    The function works in two distinct optimization steps:
        1. It performs an initial optimization on the pressure model using the current temperature values.
        2. It refines the model by incorporating temperature effects.
    The optimization is executed using a least squares method with a custom 'arctan' loss function. If the optimization
    results in parameters at the active bounds, a message is printed and a fallback optimization is attempted. If the
    fallback optimization fails to produce parameters within the specified bounds, the function returns None.

    Exceptions
    ----------
    AssertionError
        Raised if the fallback optimization produces parameters outside the specified bounds.
    """
    mask = np.logical_and(np.isfinite(dp), np.isfinite(temp_wvp))
    temp_wvp = temp_wvp[mask]
    index = index[mask]
    flow = flow[mask]
    dp = dp[mask]
    temp_mean = temp_wvp.mean()

    def get_df_a(theta):
        slope_val, offset = theta

        return pd.Series({
            "offset": offset,
            "offset_datum": offset_datum,
            "slope": slope_val,
            "temp_mean": 0,
            "temp_delta": 0,
            "time_offset": 0,
            "method": "Niet",
            "temp_ref": 12.0,
        })

    def fun_a(theta):
        return get_df_a(theta).wvp.dp_model(index, flow, temp_wvp=temp_wvp)

    def cost_a(theta):
        return (fun_a(theta) - dp) ** 2

    a_approx = (dp / flow).median()
    x0 = [-1e-8, a_approx]
    bounds_ = ([-1e-1, -1e-10], [(dp / flow).median(), -0.001])
    bounds = np.array(bounds_).T

    try:
        res = least_squares(
            cost_a,
            x0=x0,
            bounds=bounds,
            loss="arctan",
            f_scale=0.5,
            xtol=1e-14,
            ftol=1e-14,
        )
        if np.any(res.active_mask):
            print("Optimal parameters are outside bounds")

        df_a1 = get_df_a(res.x)  # using temp_wvp

    except:
        res = least_squares(cost_a, x0=x0, bounds=bounds, loss="arctan", f_scale=0.5)
        assert ~np.any(res.active_mask), "Optimal parameters are outside bounds"

        return None

    def get_df_temp_model(theta):
        temp_delta, time_offset = theta

        return pd.Series({
            "offset": df_a1.offset,
            "offset_datum": df_a1.offset_datum,
            "slope": df_a1.slope,
            "temp_mean": temp_mean,
            "temp_delta": temp_delta,
            "time_offset": time_offset,
            "method": "sin",
            "temp_ref": df_a1.temp_ref,
        })

    def fun_temp(theta):
        return get_df_temp_model(theta).wvp.dp_model(index, flow)

    def cost_temp(theta):
        return (fun_temp(theta) - dp) ** 2

    temp_min, temp_max = temp_wvp.quantile([0.05, 0.95])
    x0 = [(temp_max - temp_min) / 2, 150]
    bounds_ = ([0, 10], [0, 365])
    bounds = np.array(bounds_).T

    try:
        res = least_squares(
            cost_temp,
            x0=x0,
            bounds=bounds,
            loss="arctan",
            f_scale=0.5,
            xtol=1e-14,
            ftol=1e-14,
        )
        if np.any(res.active_mask):
            print("Optimal parameters are outside bounds")

        return get_df_temp_model(res.x)

    except:
        res = least_squares(cost_temp, x0=x0, bounds=bounds, loss="arctan", f_scale=0.5)
        assert ~np.any(res.active_mask), "Optimal parameters are outside bounds"

        return None


temp_ref = 12.0

config = get_config()
gridspec_kw = {
    "left": 0.07,
    "bottom": 0.12,
    "right": 0.94,
    "top": 0.88,
    "wspace": 0.2,
    "hspace": 0.2,
}

filterweerstand_fp = results_dir / "Filterweerstand" / "Filterweerstand_modelcoefficienten.xlsx"
leidingweerstand_fp = results_dir / "Leidingweerstand" / "Leidingweerstand_modelcoefficienten.xlsx"
df_a_fp = res_folder / "Wvpweerstand_modelcoefficienten.xlsx"

for strang, c in config.iterrows():
    logger_handler.setFormatter(logging.Formatter(f"{strang}\t| %(message)s"))
    stdout.setFormatter(logging.Formatter(f"{strang}\t| %(message)s"))

    logging.info("Strang: %s", strang)

    df_fp = data_dir / "Merged" / f"{strang}.feather"
    df = pd.read_feather(df_fp).set_index("Datum")

    include_rules = [
        "Unrealistic flow",
        "Tijdens spuien",
        "Tijdens proppen",
        "Little flow",
        # "Niet steady",
        # "Niet 3-day steady"
    ]
    untrusted_measurements = get_false_measurements(df, c, extend_hours=10, include_rules=include_rules)

    df.loc[untrusted_measurements, :] = np.nan

    # only use steady state (2 days constant)
    df_a_filter = pd.read_excel(filterweerstand_fp, sheet_name=strang)
    df_a_leiding = pd.read_excel(leidingweerstand_fp, sheet_name=strang)
    p_omstorting = df.gws1.where(~df.gws1.isna(), df.gws0 - df_a_filter.wel.dp_model(df.index, df.Q / c.nput))
    df["dP_wvp2"] = p_omstorting - df.pandpeil
    percentage = 0.20

    window = int(timedelta(days=2) / (df.index[1] - df.index[0]))
    dQ_rol = np.abs(df.Q.diff() / df.Q).rolling(window=window, min_periods=window, center=False).max()
    out = pd.Series(index=df.index, data=True)
    n_true = int(percentage * len(df))
    is_true = dQ_rol.nsmallest(n_true)
    out[is_true.index] = False
    df.loc[out, :] = np.nan

    df_a = get_wvp_slope_per_year2(df.index, df.Q, df.dP_wvp2, df.index[0], df.T_bodem)

    # measured dp
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True, gridspec_kw=gridspec_kw)
    ax0.plot(
        df.index,
        -df_a.wvp.dp_model(df.index, df.Q),
        c="C1",
        label="Model met sinus temp_wvp",
        lw=0.8,
    )
    ax0.plot(df.index, -df.dP_wvp2, c="C0", label="Gemeten")
    # ax0.legend(fontsize="small")
    ax0.legend(loc=(0, 1), ncol=2)
    ax0.set_ylabel("Drukverlies wvp bij gemeten Q (m)")
    ax0.xaxis.set_major_locator(mdates.YearLocator())
    ax0.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax0.xaxis.get_major_locator()))
    Q_avg = df.Q.median()
    std = (df_a.wvp.dp_model(df.index, Q_avg, df.T_bodem) - df.dP_wvp2 / df.Q * Q_avg).std()
    model_std = (df_a.wvp.dp_model(df.index, Q_avg) - df.dP_wvp2 / df.Q * Q_avg).std()
    df_a["model_std"] = model_std

    ax1.plot(
        df.index,
        -df_a.wvp.dp_model(df.index, Q_avg),
        c="C1",
        label=f"Model met sinus temp_wvp (std={model_std:.2f}m)",
        lw=0.8,
    )
    ax1.plot(df.index, -df.dP_wvp2 / df.Q * Q_avg, c="C0", label="Gemeten (std=0m)")
    # ax1.legend(fontsize="small")
    ax1.legend(loc=(0, 1), ncol=2)
    ax1.set_ylabel(f"Drukverlies wvp bij Q={Q_avg:.0f}m3/h (m)")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    fig.tight_layout()
    fig_path = res_folder / f"Wvpweerstandcoefficient - {strang}.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logging.info("Saved result to %s", fig_path)

    df_a["gewijzigd"] = pd.Timestamp.now()
    with pd.ExcelWriter(df_a_fp, if_sheet_exists="replace", mode="a", engine="openpyxl") as writer:
        df_a.to_excel(writer, sheet_name=strang)

print("hoi")
