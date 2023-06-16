import os

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dawacotools as dw


# vanaf najaar 2015 uit Sander Uitendaal's schemas
# Tussen najaar 2012 en najaar 2015 schatting uit SAP lijst van Gerhard
werkzaamheden_dict = {
    "Q100": [(2017, 40, 41), (2019, 1, 1), (2021, 35, 36)],
    "Q200": [(2016, 48, 49), (2020, 43, 44)],
    "Q300": [(2014, 38, 38), (2017, 36, 37), (2020, 48, 49)],
    "Q400": [(2017, 8, 10), (2020, 36, 37), (2023, 7, 8)],
    "Q500": [(2015, 43, 44), (2018, 41, 42), (2019, 41, 42), (2023, 4, 5)],
    "Q600": [(2015, 35, 37), (2017, 43, 45), (2021, 43, 44)],
    "P100": [(2014, 34, 34), (2017, 50, 50), (2021, 4, 8)],
    "P200": [(2016, 40, 42), (2018, 44, 45), (2019, 44, 45), (2021, 49, 50)],
    "P300": [(2014, 7, 7), (2015, 45, 47), (2017, 46, 47), (2018, 36, 38), (2019, 36, 38), (2021, 38, 39)],
    "P400": [(2015, 50, 51), (2018, 47, 48), (2019, 47, 48), (2022, 35, 36)],
    "P500": [(2015, 7, 8), (2018, 49, 51), (2020, 2, 3)],
    "P600": [(2014, 38, 38), (2020, 39, 41), (2022, 47, 49)],
    "IK91": [(2017, 50, 51), (2023, 10, 13)],
    "IK92": [(2014, 46, 46), (2018, 50, 50), (2020, 46, 46)],
    "IK93": [(2019, 10, 11), (2020, 10, 11), (2022, 1, 2), (2022, 50, 51)],
    "IK94": [(2014, 50, 50), (2018, 6, 7), (2021, 10, 11), (2022, 7, 8)],
    "IK95": [(2017, 11, 12), (2019, 2, 3), (2019, 49, 49)],
    "IK96": [(2016, 38, 39), (2020, 49, 50)],
    "IK101": [(2014, 49, 49), (2018, 3, 4), (2022, 4, 5)],
    "IK102": [(2014, 50, 50), (2018, 13, 14), (2020, 31, 32), (2021, 43, 50)],
    "IK103": [(2014, 39, 39), (2018, 6, 7), (2019, 6, 6), (2021, 1, 1), (2021, 9, 9)],
    "IK104": [(2015, 48, 49), (2018, 10, 11), (2019, 11, 11), (2023, 1, 2)],
    "IK105": [(2016, 44, 45), (2021, 13, 13), (2022, 41, 42)],
    "IK106": [(2017, 5, 6), (2019, 6, 7), (2020, 6, 6)],
}


def werkzaamheden_dates():
    def yr_wk_to_date(year, wk_start, wk_end):
        date_start = datetime.strptime(f"{year}-W{wk_start}-3", "%G-W%V-%u")
        date_end = datetime.strptime(f"{year}-W{wk_end}-3", "%G-W%V-%u")
        date_avg = date_start + (date_end - date_start) / 2
        return date_avg

    d = {k: pd.DatetimeIndex([yr_wk_to_date(*vv) for vv in v]) for k, v in werkzaamheden_dict.items()}
    return d


def remove_per_from_werkzh_per(werkzh_per, idrop):
    nper = len(werkzh_per) - 1
    assert idrop <= nper - 1

    dates = np.unique(werkzh_per)
    dates = np.array([date for i, date in enumerate(dates) if i != idrop + 1])

    out = []

    for start, end in zip(dates[:-1], dates[1:]):
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
    window = int(timedelta(days=1/12) / (d.index[1] - d.index[0]))
    is_steady = (
        np.abs(dQ).rolling(window=window, min_periods=window, center=False).max()
        < 5.
    )
    return ~is_steady


def get_rule_is3dsteady(d, c):
    dQ = d.Q.diff() / d.Q * 100.0
    window = int(timedelta(days=3) / (d.index[1] - d.index[0]))
    is_steady = (
        np.abs(dQ).rolling(window=window, min_periods=window, center=False).max()
        < 5.
    )
    return ~is_steady


def get_rule_is1dsteady(d, c):
    dQ = d.Q.diff() / d.Q * 100.0
    window = int(timedelta(days=1) / (d.index[1] - d.index[0]))
    is_steady = (
        np.abs(dQ).rolling(window=window, min_periods=window, center=False).max()
        < 5.
    )
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
            lambda d, c: d["Q"]
            > np.nanmin((np.nanmax((c.test2009, c.test2015, c.test2021)), c.Qpomp)),
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


def get_false_measurements(
    ds, c, extend_hours=None, exclude_rules=[], include_rules=[]
):
    """True for untrusted values/times. c is row of config of type pandas dataseries"""
    rules = get_rules(exclude_rules=exclude_rules, include_rules=include_rules)

    r_space = max([len(r[0]) for r in rules])
    show_list = []

    for ir, (r_label, r_rule) in enumerate(rules):
        show = r_rule(ds, c)
        print(f'{r_label.ljust(r_space)}: {show.sum() / show.size * 100:4.1f}% of the measurements.')
        show_list.append(show)

    shows = np.logical_or.reduce(show_list, axis=0)

    if extend_hours is not None:
        # Also include the values `extend_hours` before and after
        ndt = int(timedelta(hours=extend_hours) / (ds.index[1] - ds.index[0]))
        shows = [
            any(shows[max((i - ndt, 0)) : min((i + ndt, shows.size - 1))])
            for i in range(shows.size)
        ]

    return shows


def smooth(df, days=1):
    window = int(timedelta(days=days) / (df.index[1] - df.index[0]))
    return df.dropna().rolling(window=window, center=True).median().reindex(df.index)


def plot_false_measurements(
    ax, ds, c, extend_hours=None, exclude_rules=[], include_rules=[]
):
    rules = get_rules(exclude_rules=exclude_rules, include_rules=include_rules)

    for ir, (r_label, r_rule) in enumerate(rules):
        show = get_false_measurements(
            ds, c, extend_hours=extend_hours, include_rules=[r_label]
        )

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
        label="Totaal: {:.2f}%".format(sum(shows) / len(ds) * 100),
    )

    ax.set_ylabel("Foutieve metingen")
    ax.legend(fontsize='small')


def get_trusted_measurements(
    ds, c, extend_hours=None, exclude_rules=[], include_rules=[]
):
    """True for trusted values/times"""
    return ~get_false_measurements(
        ds,
        c,
        extend_hours=extend_hours,
        exclude_rules=exclude_rules,
        include_rules=include_rules,
    )


def get_knmi_bodemtemperature(fn):
    bds = pd.read_csv(
        fn, sep=",", skiprows=16, engine="c", na_values="     ", parse_dates=[[1, 2]]
    )
    bds["date"] = [
        datetime.strptime(k.split()[0], "%Y%m%d")
        + pd.Timedelta(int(k.split()[-1]), unit="h")
        for k in bds["YYYYMMDD_HH"]
    ]
    del bds["YYYYMMDD_HH"]
    del bds["Unnamed: 12"]
    del bds["# STN"]

    bds.columns = bds.columns.str.strip()

    bds.set_index("date", inplace=True)

    bds /= 10.0
    return bds


def get_config(fn):
    config = pd.read_excel(fn).set_index("Unnamed: 0").T
    config = config.loc[:, config.columns.notna()]
    return config


def read_plenty_excel(plenty_path):
    """
    Read plenty data to feather without dawaco data
    :param plenty_path:
    :return:
    """
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

        if (
            "index" in plenty_data
        ):  # in some datasets a column named index falsely appeared
            del plenty_data["index"]

        plenty_data.to_feather(plenty_path_feather)

    else:
        plenty_data = pd.read_feather(plenty_path_feather)
        plenty_data["ophaal tijdstip"] = pd.to_datetime(plenty_data["ophaal tijdstip"])

        if (
            "index" in plenty_data
        ):  # in some datasets a column named index falsely appeared
            del plenty_data["index"]

        plenty_data.set_index("ophaal tijdstip", inplace=True)

    return plenty_data


# def read_strang_data(strang, plenty_path, config):
#     assert strang in config
#
#     return None


def prepare_strang_data(plenty_path, fp_out, config):
    """
    Combines plenty data with a dawaco measurement for entire secundair.

    """
    if not os.path.exists(fp_out):
        plenty_data = read_plenty_excel(plenty_path)

        config_sel_mask = config.PA_tag_prefix.isin(
            set(s.split("_")[0] for s in plenty_data)
        )
        config_sel = config.loc[config_sel_mask]

        # parse pressure sensors:
        # for strang, c in config_sel.iterrows():
        # P = pt10/10 + hleidingdruk - hleidingdruk_offset
        # plenty_data[f'{c.PA_tag_prefix}_PT10'] / 10. + c.hleidingdruk
        # plenty_data[f"{c.PA_tag_prefix}_P"] =

        # Filter noisy vacuumurenteller and compute derivative
        for strang, c in config_sel.iterrows():
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
            plenty_data[name_gws] = gws.reindex(plenty_data.index).interpolate(
                "slinear", limit=ndt
            )
            plenty_data[name_gwt] = gwt.reindex(plenty_data.index).interpolate(
                "slinear", limit=ndt
            )

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
    visc_ref = (
        1 + 0.0155 * (temp_ref - 20.0)
    ) ** -1.572  # / 1000  removed the division because we re taking a ratio.
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
        series = series.rolling(2 * int_shift, win_type="gaussian", center=center).mean(
            std=int_sigma
        )
        series = pd.Series(data=series.values[::-1], index=series.index)
        return series

    else:
        return series.rolling(2 * int_shift, win_type="gaussian", center=center).mean(
            std=int_sigma
        )


def temp_correct_discharge(
    discharge, temp_inf, residence_time, residence_sigma, temp_ref=12.0
):
    if residence_time == "Niet":
        return pd.Series(data=discharge, index=temp_inf.index)
    else:
        aquifer_temp = deconvolve_wvp(temp_inf, residence_time, residence_sigma)
        return discharge * visc_ratio(aquifer_temp, temp_ref=temp_ref)


def temp_correct_drawdown_or_cwvp(
    drawdown, temp_inf, residence_time, residence_sigma, temp_ref=12.0
):
    if residence_time == "Niet":
        return pd.Series(data=drawdown, index=temp_inf.index)
    else:
        aquifer_temp = deconvolve_wvp(temp_inf, residence_time, residence_sigma)
        return drawdown / visc_ratio(aquifer_temp, temp_ref=temp_ref)


def model_a_leiding(df, periods, slope, offsets, Q=None):
    d_offset = pd.Series(index=df.index, data=0)
    d_days_start_wzh = pd.Series(index=df.index, data=0.)

    # Adjust final end time assume no schoonmaak happened
    periods[-1] = (periods[-1][0], df.index[-1])

    for offset, (start, end) in zip(offsets, periods):
        d_offset[start:end] = offset
        d_days_start_wzh[start:end] = - (d_days_start_wzh[start:end].index - start) / pd.Timedelta(days=1)

    # Add only an offset to times before first period
    d_offset[:periods[0][0]] = offsets[0]

    if Q is None:
        return -(d_offset + slope * d_days_start_wzh) * df.Q ** 2
    else:
        return -(d_offset + slope * d_days_start_wzh) * Q ** 2


def model_a_put(df, periods, slope, offsets):
    d_offset = pd.Series(index=df.index, data=0)
    d_days_start_wzh = pd.Series(index=df.index, data=0.)

    # Adjust final end time assume no schoonmaak happened
    periods[-1] = (periods[-1][0], df.index[-1])

    for offset, (start, end) in zip(offsets, periods):
        d_offset[start:end] = offset
        d_days_start_wzh[start:end] = - (d_days_start_wzh[start:end].index - start) / pd.Timedelta(days=1)

    # Add only an offset to times before first period
    d_offset[:periods[0][0]] = offsets[0]

    return d_offset + slope * d_days_start_wzh

