"""Fit legacy empirical or explicitly temperature-normalized filter coefficients.

Run as a module. Temperature normalization is opt-in and writes a separate
reference-temperature workbook, never silently replacing legacy coefficients.
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from productiecapaciteit.src.filter_calibration import (
    ExtractedWaterTemperatureError,
    extracted_water_viscosity_ratio,
    fit_normalized_filter,
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
    """Returns df_a"""
    werkzh_datums = list(werkzh_datums)

    _, df_a = get_put_slope_per_year2(df_dPdQ, werkzh_datums, slope_val=slope)

    if np.all(df_a.wel.a_effect[1:] <= 1):
        logging.info(f"Effectieve schoonmaken: {werkzh_datums[1:]}")

    else:
        idrop = np.argmax(df_a.wel.a_effect[1:]) + 1
        removed = werkzh_datums.pop(idrop)
        logging.info("=> Dropping: %s. Remaining dates: %s", removed, werkzh_datums)

        df_a = analyse_a_put2(df_dPdQ, werkzh_datums, Q_avg, t_projectie=t_projectie, slope=slope)

    for datum, dp_voor, dp_na in zip(df_a.datum, df_a.wel.dp_voor(Q_avg), df_a.wel.dp_na(Q_avg), strict=False):
        logging.info(f"Schoonmaak van {datum}: Drukval bij mediaan debiet gaat van {dp_voor:.2f}m naar {dp_na:.2f}m")

    dp_voor = df_a.wel.dp_projectie_voor(t_projectie, Q_avg)
    dp_na = df_a.wel.dp_projectie_na(t_projectie, Q_avg, method="mean")
    logging.info(
        f"Bij schoonmaak in {t_projectie} gaat drukval bij Q={Q_avg:.1f}m3/h gaat van {dp_voor:.2f}m naar {dp_na:.2f}m"
    )
    return df_a


def fit_filter_observations(
    observations: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    normalized: bool = False,
    configured_slope: float | None = None,
    nput: int | None = None,
    temperature_column: str | None = None,
    sensor_id: str = "unspecified",
    head_provenance: str = "direct",
    upstream_fit_dependencies: list[dict] | None = None,
    sensor_flow_column: str | None = None,
    allow_effective_equal_flow: bool = False,
    reference_temperature: float = 12.0,
    fit_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Select the legacy fit or an opt-in native reference-temperature fit.

    Parameters
    ----------
    observations : DataFrame
        Native timestamp-indexed gws0/gws1 heads (m), Q and optional individual
        flow (m3/h), and explicitly selected extracted-water temperature (degC).
    dates : DatetimeIndex
        Initial coefficient datum and cleaning dates.
    normalized : bool, default False
        Refit both offsets and slope in reference-temperature space. Otherwise
        preserve the existing empirical fitting algorithm.
    configured_slope : float or None, optional
        Legacy put_a_slope. Never used by the normalized route.
    nput : int or None, optional
        Active well count for Q/nput. Required by the legacy route.
    temperature_column, sensor_id, head_provenance : str, optional
        Explicit temperature channel, sensor identity and direct/reconstructed
        outside-head provenance. Normalized fitting requires a known sensor.
    upstream_fit_dependencies : list of dict or None, optional
        Upstream parameter fits and periods. Required explicitly for normalized
        fitting, including [] if no upstream dependencies exist.
    sensor_flow_column : str or None, optional
        Individual measured flow; preferred by normalized fitting.
    allow_effective_equal_flow : bool, default False
        Explicit normalized-fitting permission to use effective Q/nput.
    reference_temperature : float, default 12
        Reference temperature in degC for normalized slope and offsets.
    fit_periods : list of (Timestamp, Timestamp) or None, optional
        Inclusive periods for the normalized route. The legacy route retains
        its existing supplied-span fit.

    Returns
    -------
    DataFrame
        Legacy-compatible coefficient columns with persisted semantics and
        provenance. Normalized coefficients require temperature-aware evaluation.
    """
    if normalized:
        if temperature_column is None:
            message = "Declare the extracted-water temperature_column"
            raise ExtractedWaterTemperatureError(message)
        if sensor_id == "unspecified" or upstream_fit_dependencies is None:
            message = "Normalized fitting requires sensor_id and explicit upstream_fit_dependencies"
            raise ValueError(message)
        return fit_normalized_filter(
            observations,
            dates,
            temperature_column=temperature_column,
            sensor_id=sensor_id,
            head_provenance=head_provenance,
            upstream_fit_dependencies=upstream_fit_dependencies,
            sensor_flow_column=sensor_flow_column,
            nput=nput,
            allow_effective_equal_flow=allow_effective_equal_flow,
            reference_temperature=reference_temperature,
            fit_periods=fit_periods,
        )
    if nput is None or nput <= 0 or fit_periods is not None:
        message = "Legacy fitting requires positive nput and its original full-span fitting period"
        raise ValueError(message)
    coefficient = (observations.gws0 - observations.gws1) / (observations.Q / nput)
    result = analyse_a_put2(
        coefficient, dates, observations.Q.mean() / nput, t_projectie="2025-10-31 00:00:00", slope=configured_slope
    )
    result["gewijzigd"] = pd.Timestamp.now()
    result = result.wel.add_zero_effect_dates(dates[1:])
    result["temperature_semantics"] = "legacy_empirical"
    result["temperature_ref_degC"] = np.nan
    result["temperature_policy"] = "not_corrected"
    result["coefficient_units"] = "h/m2"
    result["slope_units"] = "h/m2/day"
    result["sensor_id"] = sensor_id
    result["head_provenance"] = head_provenance
    result["inside_head_channel"] = "gws0"
    result["outside_head_channel"] = "gws1"
    result["flow_denominator_provenance"] = f"effective_equal_flow:Q/nput={nput}"
    result["fit_periods"] = json.dumps([[observations.index[0].isoformat(), observations.index[-1].isoformat()]])
    result["upstream_fit_dependencies"] = json.dumps(upstream_fit_dependencies)
    result["fit_method"] = "legacy_arctan_squared_residuals"
    result["observation_support"] = "native_instantaneous"
    result["n_observations"] = int(np.isfinite(coefficient).sum())
    return result


def main() -> None:
    """Run the workbook/plot report; keep data loading outside import time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalize-temperature", action="store_true")
    parser.add_argument("--strang")
    parser.add_argument("--temperature-column")
    parser.add_argument("--sensor-id")
    parser.add_argument("--sensor-flow-column")
    parser.add_argument("--head-provenance", choices=("direct", "reconstructed"))
    parser.add_argument("--upstream-fit-dependencies-json")
    parser.add_argument("--allow-effective-equal-flow", action="store_true")
    parser.add_argument("--reference-temperature", type=float, default=12.0)
    args = parser.parse_args()
    if args.normalize_temperature and not all([
        args.strang,
        args.temperature_column,
        args.sensor_id,
        args.head_provenance,
        args.upstream_fit_dependencies_json,
    ]):
        parser.error(
            "Normalized fitting requires one --strang, temperature channel, sensor, head provenance and dependencies"
        )
    dependencies = json.loads(args.upstream_fit_dependencies_json) if args.upstream_fit_dependencies_json else None
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from productiecapaciteit import data_dir, plot_styles_dir, results_dir  # noqa: PLC0415
    from productiecapaciteit.src.strang_analyse_fun2 import (  # noqa: PLC0415
        get_config,
        get_false_measurements,
        smooth,
        werkzaamheden_dates,
    )

    res_folder = results_dir / "Filterweerstand"
    res_folder.mkdir(parents=True, exist_ok=True)
    logger_handler = logging.FileHandler(res_folder / "Putweerstandcoefficient.log", mode="w")
    stdout = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logger_handler, stdout],
    )
    config = get_config()
    if args.strang:
        config = config.loc[[args.strang]]
    gridspec_kw = {"left": 0.07, "bottom": 0.12, "right": 0.9, "top": 0.88, "wspace": 0.2, "hspace": 0.2}
    suffix = "_reference" if args.normalize_temperature else ""
    df_a_fp = res_folder / f"Filterweerstand_modelcoefficienten{suffix}.xlsx"

    for strang, c in config.iterrows():
        logger_handler.setFormatter(logging.Formatter(f"{strang}\t| %(message)s"))
        stdout.setFormatter(logging.Formatter(f"{strang}\t| %(message)s"))
        logging.info("Strang: %s", strang)
        df = pd.read_feather(data_dir / "Merged" / f"{strang}.feather").set_index("Datum")
        untrusted_measurements = get_false_measurements(df, c, extend_hours=1, include_rules=["Unrealistic flow"])
        df.loc[untrusted_measurements] = np.nan
        flow = df[args.sensor_flow_column] if args.normalize_temperature and args.sensor_flow_column else df.Q / c.nput
        df["dPdQ"] = (df.gws0 - df.gws1) / flow
        if args.normalize_temperature:
            valid = np.isfinite(df.dPdQ)
            if args.temperature_column not in df:
                message = f"Missing extracted-water temperature channel: {args.temperature_column}"
                raise ExtractedWaterTemperatureError(message)
            df.loc[valid, "dPdQ"] /= extracted_water_viscosity_ratio(
                df.loc[valid, args.temperature_column],
                df.index[valid],
                reference_temperature=args.reference_temperature,
            )
        dates = werkzaamheden_dates()[strang]
        dates = dates[dates > df.dPdQ.dropna().index[0]]
        if args.normalize_temperature:
            dates = dates[dates <= df.index[-1]]
        werkzh_datums = pd.DatetimeIndex(np.concatenate((df.dPdQ.dropna().index[[0]].values, dates)))
        df_a = fit_filter_observations(
            df,
            werkzh_datums,
            normalized=args.normalize_temperature,
            configured_slope=c.put_a_slope,
            nput=int(c.nput),
            temperature_column=args.temperature_column,
            sensor_id=args.sensor_id or "unspecified",
            head_provenance=args.head_provenance or "direct",
            upstream_fit_dependencies=dependencies,
            sensor_flow_column=args.sensor_flow_column,
            allow_effective_equal_flow=args.allow_effective_equal_flow,
            reference_temperature=args.reference_temperature,
        )
        df_a["gewijzigd"] = pd.Timestamp.now()
        existing_workbook = df_a_fp.exists()
        with pd.ExcelWriter(
            df_a_fp,
            mode="a" if existing_workbook else "w",
            if_sheet_exists="replace" if existing_workbook else None,
            engine="openpyxl",
        ) as writer:
            df_a.to_excel(writer, sheet_name=strang)

        df["dPdQ_smooth"] = smooth(df["dPdQ"], days=1)
        q_avg = flow.mean()
        plt.style.use(plot_styles_dir / "unhcrpyplotstyle.mplstyle")
        plt.style.use(plot_styles_dir / "types" / "line.mplstyle")
        fig, ax2 = plt.subplots(1, 1, figsize=(12, 5), gridspec_kw=gridspec_kw)
        ax = ax2.twinx()
        fig.suptitle(strang)
        ax.axhline(0, c="black", lw="0.8")
        df_a.wel.plot_werkzh(ax, werkzh_datums)
        ax.plot(df.index, df["dPdQ_smooth"], label="dP/dQ (dag gem.)")
        ax.plot(df.index, df_a.wel.a_model(df.index), label=f"Model {df_a.slope.mean():.3g}")
        reference_label = f" bij {args.reference_temperature:g} degC" if args.normalize_temperature else ""
        ax.set_ylabel(f"Weerstand{reference_label}: Verlaging bij Q = 1 m3/h (m)")
        ax.set_ylim(-4 / q_avg, 1 / q_avg)
        ax.set_yticks(np.arange(-4, 2) / q_avg)
        ax.legend(loc=(0, 1), ncol=4)
        ax2.set_ylim((-4, 1))
        ax2.set_yticks(np.arange(-4, 2))
        ax2.set_xlim(df.index[[0, -1]])
        ax2.set_ylabel(f"Verlaging{reference_label} bij Q_put={q_avg:.1f}m3/h (m)")
        fig_path = res_folder / f"Putweerstandcoefficient{suffix} - {strang}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        logging.info("Saved result to %s", fig_path)


if __name__ == "__main__":
    main()
