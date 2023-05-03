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
