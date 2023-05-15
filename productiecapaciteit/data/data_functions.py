"""
1. Update PA data: Data/Plenty/PREP2XL_v122_ICAS sec ..xlsm files
2. Remove cache: Data/Plenty/Q, P, 09, and 10.feather
3. Update bodemtemps_260.txt
4. Check the content of config_fn, e.g., number of wells etc.

"""

import os
import dawacotools as dw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

werkzaamheden_dict = {
    "Q100": [(2017, 40, 41), (2021, 35, 36)],
    "Q200": [(2016, 48, 49), (2020, 43, 44)],
    "Q300": [(2017, 36, 37), (2020, 48, 49)],
    "Q400": [(2017, 8, 10), (2020, 36, 37), (2023, 7, 8)],
    "Q500": [(2015, 43, 44), (2018, 41, 42), (2019, 41, 42), (2023, 4, 5)],
    "Q600": [(2015, 35, 37), (2017, 43, 45), (2021, 43, 44)],
    "P100": [(2021, 4, 8)],
    "P200": [(2016, 40, 42), (2018, 44, 45), (2019, 44, 45), (2021, 49, 50)],
    "P300": [(2015, 45, 47), (2017, 46, 47), (2018, 36, 38), (2019, 36, 38), (2021, 38, 39)],
    "P400": [(2015, 50, 51), (2018, 47, 48), (2019, 47, 48), (2022, 35, 36)],
    "P500": [(2015, 7, 8), (2018, 49, 51), (2020, 2, 3)],
    "P600": [(2020, 39, 41), (2022, 47, 49)],
    "IK91": [(2017, 50, 51), (2023, 10, 13)],
    "IK92": [(2020, 46, 46)],
    "IK93": [(2019, 10, 11), (2020, 10, 11), (2022, 1, 2), (2022, 50, 51)],
    "IK94": [(2021, 10, 11), (2022, 7, 8)],
    "IK95": [(2017, 11, 12), (2019, 2, 3), (2019, 49, 49)],
    "IK96": [(2016, 38, 39), (2020, 49, 50)],
    "IK101": [(2018, 3, 4), (2022, 4, 5)],
    "IK102": [(2020, 31, 32), (2021, 43, 50)],
    "IK103": [(2018, 6, 7), (2021, 1, 1), (2021, 9, 9)],
    "IK104": [(2015, 48, 49), (2018, 10, 11), (2023, 1, 2)],
    "IK105": [(2016, 44, 45), (2021, 13, 13), (2022, 41, 42)],
    "IK106": [(2017, 5, 6), (2019, 6, 7), (2020, 6, 6)],
}


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

        # Filter noisy vacuumurenteller and compute derivative
        for strang, c in config_sel.iterrows():
            if f"{c.PA_tag_prefix}_OP20" not in plenty_data:
                continue

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

        plenty_data.reset_index().to_feather(fp_out)

    else:
        plenty_data = pd.read_feather(fp_out)
        plenty_data["ophaal tijdstip"] = pd.to_datetime(plenty_data["ophaal tijdstip"])
        plenty_data.set_index("ophaal tijdstip", inplace=True)

    return plenty_data


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


def _werkzaamheden_dates():
    def yr_wk_to_date(year, wk_start, wk_end):
        date_start = datetime.strptime(f"{year}-W{wk_start}-3", "%G-W%V-%u")
        date_end = datetime.strptime(f"{year}-W{wk_end}-3", "%G-W%V-%u")
        date_avg = date_start + (date_end - date_start) / 2
        return date_avg

    d = {k: [yr_wk_to_date(*vv) for vv in v] for k, v in werkzaamheden_dict.items()}
    return d


