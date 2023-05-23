import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from productiecapaciteit.src.capaciteit_strang import strangWeerstand
from productiecapaciteit.src.capaciteit_strang import get_config

from productiecapaciteit.src.weerstand_pandasaccessors import LeidingResistanceAccessor
from productiecapaciteit.src.weerstand_pandasaccessors import WellResistanceAccessor
from productiecapaciteit.src.weerstand_pandasaccessors import WvpResistanceAccessor

data_fd = os.path.join("..", "data")
config_fn = "strang_props6.xlsx"
config = get_config(os.path.join(data_fd, config_fn))
config = config.loc[:, config.columns.notna()]

filterweerstand_fp = os.path.join(
    "..", "results", "Filterweerstand", "Filterweerstand_modelcoefficienten.xlsx"
)
leidingweerstand_fp = os.path.join(
    "..", "results", "Leidingweerstand", "Leidingweerstand_modelcoefficienten.xlsx"
)
wvpweerstand_fp = os.path.join(
    "..", "results", "Wvpweerstand", "Wvpweerstand_modelcoefficienten.xlsx"
)

date_goal = pd.Timestamp("2024-07-01")
date_clean = "2023-11-01"
index = pd.date_range("2012-05-01", date_clean)

som_dict = dict()
report = dict()

for strang, c in config.iterrows():
    # if strang != "IK91":
    #     continue

    df_a_filter = pd.read_excel(filterweerstand_fp, sheet_name=strang)
    df_a_leiding = pd.read_excel(leidingweerstand_fp, sheet_name=strang)
    df_a_wvp = pd.read_excel(wvpweerstand_fp, sheet_name=strang, index_col=0).squeeze(
        "columns"
    )

    weerstand = strangWeerstand(df_a_leiding, df_a_filter, df_a_wvp, **c.to_dict())
    effect_som, effect_dict = weerstand.report_capaciteit_effect_schoonmaak(date_clean, [date_goal])
    frac = effect_dict["ratio_lei"].item()
    cap_min = weerstand.capaciteit(index).min()

    lims = weerstand.lims([date_clean]).iloc[0]
    cap = lims.min()
    # Maak een lijst van alle limieten die dicht bij de capaciteit liggen.
    lim_cats = lims[lims < cap * 1.1].index

    lims_schoonmaak = weerstand.lims_schoonmaak(date_clean, [date_goal], leiding=True, wel=True).iloc[0]
    cap_schoonmaak = lims_schoonmaak.min()
    lim_cats_schoonmaak = lims_schoonmaak[lims_schoonmaak < cap_schoonmaak * 1.1].index

    if effect_som.item() / cap_min > 0.025:
        if frac > 0.1 and frac < 0.9:
            welke = "Leiding en filter"

        elif frac > 0.1:
            welke = "Leiding"

        elif frac < 0.9:
            welke = "Filter"

        print(f"{strang}\t{effect_som.item():.0f}\t{welke}\t{', '.join(lim_cats)}\t{', '.join(lim_cats_schoonmaak)}")
    else:
        print(f"{strang}\tNIHIL\t-\t{', '.join(lim_cats)}\t-")
print("hoi")
