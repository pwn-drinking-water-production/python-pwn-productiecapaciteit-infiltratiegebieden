import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from ..src.capaciteit_strang import strangWeerstand
from ..src.capaciteit_strang import get_config

from productiecapaciteit import LeidingResistanceAccessor
from productiecapaciteit import WellResistanceAccessor
from productiecapaciteit import WvpResistanceAccessor

res_folder = os.path.join("Resultaat", "Synthese")
logger_handler = logging.FileHandler(
    os.path.join(res_folder, "Capaciteit_analyse.log"), mode="w"
)  # , encoding='utf-8', level=logging.DEBUG)
stdout = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logger_handler, stdout],
)

gridspec_kw = {
    "left": 0.07,
    "bottom": 0.12,
    "right": 0.92,
    "top": 0.88,
    "wspace": 0.2,
    "hspace": 0.2,
}

data_fd = os.path.join("..", "Data")
config_fn = "strang_props6.xlsx"
config = get_config(os.path.join(data_fd, config_fn))
config = config.loc[:, config.columns.notna()]

filterweerstand_fp = os.path.join(
    "Resultaat", "Filterweerstand", "Filterweerstand_modelcoefficienten.xlsx"
)
leidingweerstand_fp = os.path.join(
    "Resultaat", "Leidingweerstand", "Leidingweerstand_modelcoefficienten.xlsx"
)
wvpweerstand_fp = os.path.join(
    "Resultaat", "Wvpweerstand", "Wvpweerstand_modelcoefficienten.xlsx"
)

index = pd.date_range("2012-05-01", "2025-12-31")


for strang, c in config.iterrows():
    # if strang != "Q300":
    #     continue

    df_a_filter = pd.read_excel(filterweerstand_fp, sheet_name=strang)
    df_a_leiding = pd.read_excel(leidingweerstand_fp, sheet_name=strang)
    df_a_wvp = pd.read_excel(wvpweerstand_fp, sheet_name=strang, index_col=0).squeeze(
        "columns"
    )

    weerstand = strangWeerstand(df_a_leiding, df_a_filter, df_a_wvp, **c.to_dict())
    date_clean = "2023-11-01"
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw=gridspec_kw)
    fig.suptitle(
        f"{strang}: Capaciteit met geplande schoonmaak op {pd.Timestamp(date_clean).strftime('%d-%m-%Y')}"
    )

    weerstand.plot_lims(index, date_clean, ax=ax0)
    weerstand.plot_capaciteit_effect_schoonmaak(date_clean, index, ax=ax1)

    fig_path = os.path.join(res_folder, f"Capaciteit - {strang}.png")
    fig.savefig(fig_path, dpi=300)
    logging.info(f"Saved capaciteit result to {fig_path}")

print("hoi")
