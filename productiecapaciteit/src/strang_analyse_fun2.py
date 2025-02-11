import os
from datetime import datetime, timedelta

import dawacotools as dw
import numpy as np
import pandas as pd

from productiecapaciteit import data_dir
from productiecapaciteit.data.data_functions import werkzaamheden_dict


def werkzaamheden_dates():
    def yr_wk_to_date(year, wk_start, wk_end):
        date_start = datetime.strptime(f"{year}-W{wk_start}-1", "%G-W%V-%u")
        date_end = datetime.strptime(f"{year}-W{wk_end}-5", "%G-W%V-%u")
        return date_start + (date_end - date_start) / 2

    d = {k: pd.DatetimeIndex([yr_wk_to_date(*vv) for vv in v]) for k, v in werkzaamheden_dict.items()}
    return d


def remove_per_from_werkzh_per(werkzh_per, idrop):
    nper = len(werkzh_per) - 1
    assert idrop <= nper - 1

    dates = np.unique(werkzh_per)
    dates = np.array([date for i, date in enumerate(dates) if i != idrop + 1])

    out = []

    for start, end in zip(dates[:-1], dates[1:], strict=False):
        out.append((start, end))

    return out


def get_rule_tijdens_proppen(d, c):
    def yr_wk_to_date(year, wk_start, wk_end):
        date_start = datetime.strptime(f"{year}-W{wk_start}-1", "%G-W%V-%u")
        date_end = datetime.strptime(f"{year}-W{wk_end}-7", "%G-W%V-%u")
        return date_start, date_end

    start_end = [yr_wk_to_date(*vv) for vv in werkzaamheden_dict[c.name]]

    tijdens_proppen = np.zeros_like(d.index, dtype=bool)

    for start, end in start_end:
        end2 = start + 1.25 * (end - start)
        start2 = end - 1.25 * (end - start)
        tijdens_proppen[(d.index >= start2) & (d.index < end2)] = True

    return tijdens_proppen


def get_rule_issteady(d, c):
    dQ = d.Q.diff() / d.Q * 100.0
    window = int(timedelta(days=1 / 12) / (d.index[1] - d.index[0]))
    is_steady = np.abs(dQ).rolling(window=window, min_periods=window, center=False).max() < 5.0
    return ~is_steady


def get_rule_is3dsteady(d, c):
    dQ = d.Q.diff() / d.Q * 100.0
    window = int(timedelta(days=3) / (d.index[1] - d.index[0]))
    is_steady = np.abs(dQ).rolling(window=window, min_periods=window, center=False).max() < 5.0
    return ~is_steady


def get_rule_is1dsteady(d, c):
    dQ = d.Q.diff() / d.Q * 100.0
    window = int(timedelta(days=1) / (d.index[1] - d.index[0]))
    is_steady = np.abs(dQ).rolling(window=window, min_periods=window, center=False).max() < 5.0
    return ~is_steady


def get_moststeady10(d, c):
    dQ = d.Q.diff() / d.Q * 100.0
    window = int(timedelta(days=3) / (d.index[1] - d.index[0]))
    dQ_rol = np.abs(dQ).rolling(window=window, min_periods=window, center=False).max()
    out = pd.Series(index=d.index, data=True)
    n_true = int(0.1 * len(d))
    is_true = dQ_rol.nsmallest(n_true)
    out[is_true.index] = False
    return out


def get_rules(exclude_rules=[], include_rules=[]):
    rules = [  # True when something is wrong
        ("NaNs in GWS0", lambda d, c: np.isnan(d["gws0"])),
        ("NaNs in GWT0", lambda d, c: np.isnan(d["gwt0"])),
        ("PT10 > GWS0", lambda d, c: d["P"] > d.gws0),
        # ('GWS0 > GWS1', lambda d, plenty_tag: d['gws0'] > d['gws1']),
        # ('GWS1 > GWS2', lambda d, plenty_tag: d['gws0'] > d['gws1']),
        ("NaNs in FT10", lambda d, c: np.isnan(d["Q"])),
        ("Little flow", lambda d, c: d["Q"] < 10),
        ("Very little flow", lambda d, c: d["Q"] < 5),
        ("Unrealistic flow", lambda d, c: d["Q"] > c.Qpomp),
        (
            "Improbable flow",
            lambda d, c: d["Q"] > np.nanmin((np.nanmax((c.test2009, c.test2015, c.test2021)), c.Qpomp)),
        ),
        (
            "Strang water pressure above surface level",
            lambda d, c: d["P"] > c.hmaaiveld,
        ),
        ("Well water pressure above surface level", lambda d, c: d.gws0 > c.hmaaiveld),
        (
            "Funny pandpeil",
            lambda d, c: (d.pandpeil - d.pandpeil.median()).abs() > 0.15,
        ),
        ("Tijdens proppen", get_rule_tijdens_proppen),
        ("Tijdens spuien", lambda d, c: d.spui > 2),
        ("Niet steady", get_rule_issteady),
        ("Niet 3-day steady", get_rule_is3dsteady),
        ("Niet 1-day steady", get_rule_is1dsteady),
    ]

    if include_rules:
        rules = [r for r in rules if r[0] in include_rules]

    if exclude_rules:
        rules = [r for r in rules if r[0] not in exclude_rules]

    return rules


def get_false_measurements(ds, c, extend_hours=None, exclude_rules=[], include_rules=[]):
    """True for untrusted values/times. c is row of config of type pandas dataseries"""
    rules = get_rules(exclude_rules=exclude_rules, include_rules=include_rules)

    r_space = max([len(r[0]) for r in rules])
    show_list = []

    for ir, (r_label, r_rule) in enumerate(rules):
        show = r_rule(ds, c)
        print(f"{r_label.ljust(r_space)}: {show.sum() / show.size * 100:4.1f}% of the measurements.")
        show_list.append(show)

    shows = np.logical_or.reduce(show_list, axis=0)

    if extend_hours is not None:
        # Also include the values `extend_hours` before and after
        ndt = int(timedelta(hours=extend_hours) / (ds.index[1] - ds.index[0]))
        shows = [any(shows[max((i - ndt, 0)) : min((i + ndt, shows.size - 1))]) for i in range(shows.size)]

    return shows


def smooth(df, days=1):
    window = int(timedelta(days=days) / (df.index[1] - df.index[0]))
    return df.dropna().rolling(window=window, center=True).median().reindex(df.index)


def plot_false_measurements(ax, ds, c, extend_hours=None, exclude_rules=[], include_rules=[]):
    rules = get_rules(exclude_rules=exclude_rules, include_rules=include_rules)

    for ir, (r_label, r_rule) in enumerate(rules):
        show = get_false_measurements(ds, c, extend_hours=extend_hours, include_rules=[r_label])

        # print(r_label, str(show.sum() / len(ds)))

        y1 = np.full(fill_value=ir, shape=len(ds))
        ax.fill_between(
            ds.index,
            y1=y1,
            y2=y1 + 1,
            where=show,
            alpha=0.8,
            label=r_label,
            edgecolor="None",
        )

    shows = get_false_measurements(
        ds,
        c,
        extend_hours=extend_hours,
        exclude_rules=exclude_rules,
        include_rules=include_rules,
    )

    ax.fill_between(
        ds.index,
        y1=y1 + 1,
        y2=y1 + 2,
        where=shows,
        alpha=0.8,
        label=f"Totaal: {sum(shows) / len(ds) * 100:.2f}%",
    )

    ax.set_ylabel("Foutieve metingen")
    ax.legend(fontsize="small")


def get_trusted_measurements(ds, c, extend_hours=None, exclude_rules=[], include_rules=[]):
    """True for trusted values/times"""
    return ~get_false_measurements(
        ds,
        c,
        extend_hours=extend_hours,
        exclude_rules=exclude_rules,
        include_rules=include_rules,
    )


def get_knmi_bodemtemperature(fn):
    bds = pd.read_csv(fn, sep=",", skiprows=16, engine="c", na_values="     ", parse_dates=[[1, 2]])
    bds["date"] = [
        datetime.strptime(k.split()[0], "%Y%m%d") + pd.Timedelta(int(k.split()[-1]), unit="h")
        for k in bds["YYYYMMDD_HH"]
    ]
    del bds["YYYYMMDD_HH"]
    del bds["Unnamed: 12"]
    del bds["# STN"]

    bds.columns = bds.columns.str.strip()

    bds.set_index("date", inplace=True)

    bds /= 10.0
    return bds


def get_config(fn="strang_props7.csv"):
    """Read a config file with the strangen configurations.

    Parameters
    ----------
    fn : str, Path
        File path to the config file

    Returns
    -------
    pandas.DataFrame
        DataFrame with the configurations
    """
    dtypes = {
        "leiding_a_slope": float,
        "put_a_slope": float,
        "T": str,
        "T_sigma": str,
        "hpand": float,
        "hluchthappen": float,
        "hverbindinghaalbuis": float,
        "hmaaiveld": float,
        "hbovenkantfilter": float,
        "honderkantfilter": float,
        "Qlim_bio": float,
        "Qpomp": float,
        "corr_temp_leiding": bool,
        "test2009": float,
        "test2015": float,
        "test2021": float,
        "nput": float,
        "Qlim_bio_per_put": float,
        "Qmin_inzetvolgorde20230523": float,
        "Qmax_inzetvolgorde20230523": float,
        "hverbindingvacuum": float,
        "test2015_per_put": float,
        "PA_tag_prefix": str,
        "PA_tag_flow": str,
        "PA_tag_hleiding": str,
        "Dawaco_tag": str,
        "PA_tag_pandpeil": str,
        "Binnendiameter:": float,
        "Wanddikte:": float,
        "Buitendiameter:": float,
        "Hydraulische diameter:": float,
        "Lengte (in meters):": float,
        "dx_tussenputten": float,
        "dx_mirrorwell": object,
    }
    fp = data_dir / fn
    out = pd.read_csv(fp, index_col=0, sep=";").T
    return out.astype(dtypes)


def read_plenty_excel(plenty_path):
    """Read the plenty excel file and return a pandas dataframe."""
    plenty_path_feather = plenty_path + ".feather"

    if not os.path.exists(plenty_path_feather):
        fn = os.path.join(plenty_path + ".xlsm")
        if not os.path.exists(fn):
            fn = os.path.join(plenty_path + ".xlsx")
        plenty_data = pd.ExcelFile(fn)
        plenty_data = plenty_data.parse(
            "SQLimport",
            skiprows=9,
            # index_col=0,
            header=0,
        )  # .iloc[:, :2]
        plenty_data.reset_index(inplace=True)
        plenty_data.drop(plenty_data.filter(regex="Unname"), axis=1, inplace=True)
        plenty_data.replace({"EOF": np.nan}, inplace=True)
        plenty_data["ophaal tijdstip"] = pd.to_datetime(plenty_data["ophaal tijdstip"])
        plenty_data.set_index("ophaal tijdstip", inplace=True)

        if "index" in plenty_data:  # in some datasets a column named index falsely appeared
            del plenty_data["index"]

        plenty_data.to_feather(plenty_path_feather)

    else:
        plenty_data = pd.read_feather(plenty_path_feather)
        plenty_data["ophaal tijdstip"] = pd.to_datetime(plenty_data["ophaal tijdstip"])

        if "index" in plenty_data:  # in some datasets a column named index falsely appeared
            del plenty_data["index"]

        plenty_data.set_index("ophaal tijdstip", inplace=True)

    return plenty_data


def prepare_strang_data(plenty_path, fp_out, config):
    """Combine plenty data with a dawaco measurement for entire secundair."""
    if not os.path.exists(fp_out):
        plenty_data = read_plenty_excel(plenty_path)

        config_sel_mask = config.PA_tag_prefix.isin(set(s.split("_")[0] for s in plenty_data))
        config_sel = config.loc[config_sel_mask]

        # parse pressure sensors:
        # for strang, c in config_sel.iterrows():
        # P = pt10/10 + hleidingdruk - hleidingdruk_offset
        # plenty_data[f'{c.PA_tag_prefix}_PT10'] / 10. + c.hleidingdruk
        # plenty_data[f"{c.PA_tag_prefix}_P"] =

        # Filter noisy vacuumurenteller and compute derivative
        for _, c in config_sel.iterrows():
            if f"{c.PA_tag_prefix}_OP20" not in plenty_data:
                continue

            # plenty_data[f'{c.PA_tag_prefix}_OP20.2'] = plenty_data[f'{c.PA_tag_prefix}_OP20']
            # plenty_data[f'{c.PA_tag_prefix}_OP20.2'].interpolate('slinear', inplace=True)
            # plenty_data[f'{c.PA_tag_prefix}_OP20.2delta'] = plenty_data[f'{c.PA_tag_prefix}_OP20.2'].diff()
            #
            # # remove large jumps
            # i_large = np.argwhere((plenty_data[f'{c.PA_tag_prefix}_OP20.2delta'].abs() > 0.25).values).squeeze()
            # plenty_data[f'{c.PA_tag_prefix}_OP20.2delta'][i_large] = 0.25
            # plenty_data[f'{c.PA_tag_prefix}_OP20.2delta'].fillna(0., inplace=True)
            #
            # # Change units: Aantal draaiuren per uur
            # plenty_data[f'{c.PA_tag_prefix}_OP20.2delta'][i_large] *= 4
            # plenty_data[f'{c.PA_tag_prefix}_OP20.2'] = np.cumsum(plenty_data[f'{c.PA_tag_prefix}_OP20.2delta'])

        filters = dw.get_daw_filters(mpcode=config_sel.Dawaco_tag)

        for mpcode, filtnr in filters.Filtnr.iteritems():
            print(mpcode, ",,,", filtnr)
            gws = dw.get_daw_ts_stijghgt(mpcode=mpcode, filternr=filtnr)
            gwt = dw.get_daw_ts_temp(mpcode=mpcode, filternr=filtnr)

            ndt = int(timedelta(days=2) / (plenty_data.index[1] - plenty_data.index[0]))
            name_gws, name_gwt = f"gws_{mpcode}_{filtnr}", f"gwt_{mpcode}_{filtnr}"
            plenty_data[name_gws] = gws.reindex(plenty_data.index).interpolate("slinear", limit=ndt)
            plenty_data[name_gwt] = gwt.reindex(plenty_data.index).interpolate("slinear", limit=ndt)

        # if pt10_nap is not None:
        #     pres_tags = [k for k in plenty_data.keys() if k.split('_')[-1] == 'P']
        #
        #     for tag in pres_tags:
        #         name = tag.split('_')[0]
        #         plenty_data[name +
        #                     '_dP'] = plenty_data[name + '_P'] - plenty_data['gws0']
        plenty_data.reset_index().to_feather(fp_out)

    else:
        plenty_data = pd.read_feather(fp_out)
        plenty_data["ophaal tijdstip"] = pd.to_datetime(plenty_data["ophaal tijdstip"])
        plenty_data.set_index("ophaal tijdstip", inplace=True)

    return plenty_data


def visc_ratio(temp, temp_ref=10.0):
    # visc_ratio(15, temp_ref=8) = 0.82
    visc_ref = (1 + 0.0155 * (temp_ref - 20.0)) ** -1.572  # / 1000  removed the division because we re taking a ratio.
    visc = (1 + 0.0155 * (temp - 20.0)) ** -1.572  # / 1000
    return visc / visc_ref


def deconvolve_wvp(series, shift, sigma):
    """Moving average with weights of a gausian with mean at `shift` and `sigma`. the windows is 2 * shift"""
    # Fill up nans
    series = series.interpolate(method="slinear")

    if shift[0] == "-":
        shift = shift[1:]
        int_shift = int(pd.Timedelta(shift) / (series.index[1] - series.index[0]))
        int_sigma = int(pd.Timedelta(sigma) / (series.index[1] - series.index[0]))

    elif shift == "Niet":
        return series

    else:
        int_shift = int(pd.Timedelta(shift) / (series.index[1] - series.index[0]))
        int_sigma = int(pd.Timedelta(sigma) / (series.index[1] - series.index[0]))

    if isinstance(shift, str):
        center = shift == "0D" or shift == "0"
    elif isinstance(shift, bool):
        center = ~shift
    elif isinstance(shift, (int, float)):
        center = bool(shift)

    if center:
        # int_shift changes meaning
        int_shift = 8 * int_sigma

    if shift[0] == "-":
        series = pd.Series(data=series.values[::-1], index=series.index)
        series = series.rolling(2 * int_shift, win_type="gaussian", center=center).mean(std=int_sigma)
        series = pd.Series(data=series.values[::-1], index=series.index)
        return series

    return series.rolling(2 * int_shift, win_type="gaussian", center=center).mean(std=int_sigma)


def temp_correct_discharge(discharge, temp_inf, residence_time, residence_sigma, temp_ref=12.0):
    if residence_time == "Niet":
        return pd.Series(data=discharge, index=temp_inf.index)
    aquifer_temp = deconvolve_wvp(temp_inf, residence_time, residence_sigma)
    return discharge * visc_ratio(aquifer_temp, temp_ref=temp_ref)


def temp_correct_drawdown_or_cwvp(drawdown, temp_inf, residence_time, residence_sigma, temp_ref=12.0):
    if residence_time == "Niet":
        return pd.Series(data=drawdown, index=temp_inf.index)
    aquifer_temp = deconvolve_wvp(temp_inf, residence_time, residence_sigma)
    return drawdown / visc_ratio(aquifer_temp, temp_ref=temp_ref)


def model_a_leiding(df, periods, slope, offsets, Q=None):
    d_offset = pd.Series(index=df.index, data=0)
    d_days_start_wzh = pd.Series(index=df.index, data=0.0)

    # Adjust final end time assume no schoonmaak happened
    periods[-1] = (periods[-1][0], df.index[-1])

    for offset, (start, end) in zip(offsets, periods, strict=False):
        d_offset[start:end] = offset
        d_days_start_wzh[start:end] = -(d_days_start_wzh[start:end].index - start) / pd.Timedelta(days=1)

    # Add only an offset to times before first period
    d_offset[: periods[0][0]] = offsets[0]

    if Q is None:
        return -(d_offset + slope * d_days_start_wzh) * df.Q**2
    return -(d_offset + slope * d_days_start_wzh) * Q**2


def model_a_put(df, periods, slope, offsets):
    d_offset = pd.Series(index=df.index, data=0)
    d_days_start_wzh = pd.Series(index=df.index, data=0.0)

    # Adjust final end time assume no schoonmaak happened
    periods[-1] = (periods[-1][0], df.index[-1])

    for offset, (start, end) in zip(offsets, periods, strict=False):
        d_offset[start:end] = offset
        d_days_start_wzh[start:end] = -(d_days_start_wzh[start:end].index - start) / pd.Timedelta(days=1)

    # Add only an offset to times before first period
    d_offset[: periods[0][0]] = offsets[0]

    return d_offset + slope * d_days_start_wzh
