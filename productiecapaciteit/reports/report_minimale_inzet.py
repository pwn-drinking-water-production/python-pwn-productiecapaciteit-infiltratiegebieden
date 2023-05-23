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
index = pd.date_range("2023-01-01", "2023-12-31")

flow_dict = dict()
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
    flow_mean = weerstand.capaciteit(index).median()

    deltah_veilig=1.5
    flow_min = weerstand.lim_flow_min(index, deltah_veilig=deltah_veilig)
    cap = weerstand.capaciteit(index)
    print(
        f"{strang}\t"
        f"{flow_min.max() / weerstand.nput:0.1f}\t"
        f"{weerstand.Qmin_inzetvolgorde20230523:.1f}\t"
        f"{cap.min() / weerstand.nput:0.1f}\t"
        f"{weerstand.Qmax_inzetvolgorde20230523:.1f}\t"
        
        f"{flow_min.max():0.0f}\t"
        f"{weerstand.Qmin_inzetvolgorde20230523 * weerstand.nput:.0f}\t"
        f"{cap.min():0.0f}\t"
        f"{weerstand.Qmax_inzetvolgorde20230523 * weerstand.nput:.0f}\t"
        )
# Qmax_inzetvolgorde20230523
print("hoi")
