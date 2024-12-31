import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

from productiecapaciteit import results_dir
from productiecapaciteit.src.capaciteit_strang import strangWeerstand
from productiecapaciteit.src.strang_analyse_fun2 import get_config
from productiecapaciteit.src.weerstand_pandasaccessors import (
    LeidingResistanceAccessor,  # noqa: F401
    WellResistanceAccessor,  # noqa: F401
    WvpResistanceAccessor,  # noqa: F401
)

logger_handler = logging.FileHandler(results_dir / "Synthese" / "Capaciteit_analyse.log", mode="w")
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

config = get_config()
config = config.loc[:, config.columns.notna()]

filterweerstand_fp = results_dir / "Filterweerstand" / "Filterweerstand_modelcoefficienten.xlsx"
leidingweerstand_fp = results_dir / "Leidingweerstand" / "Leidingweerstand_modelcoefficienten.xlsx"
wvpweerstand_fp = results_dir / "Wvpweerstand" / "Wvpweerstand_modelcoefficienten.xlsx"

index = pd.date_range("2012-05-01", "2025-12-31")
date_goal = pd.Timestamp("2025-10-01")

for strang, c in config.iterrows():
    # if strang != "IK95":
    #     continue

    df_a_filter = pd.read_excel(filterweerstand_fp, sheet_name=strang)
    df_a_leiding = pd.read_excel(leidingweerstand_fp, sheet_name=strang)
    df_a_wvp = pd.read_excel(wvpweerstand_fp, sheet_name=strang, index_col=0).squeeze("columns")

    weerstand = strangWeerstand(df_a_leiding, df_a_filter, df_a_wvp, **c.to_dict())
    date_clean = "2025-04-01"

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw=gridspec_kw)
    fig.suptitle(
        f"{strang}: Capaciteit met geplande schoonmaak putten en leiding op {pd.Timestamp(date_clean).strftime('%d-%m-%Y')}\n"
        f""
    )

    weerstand.plot_lims(index, date_clean, ax=ax0)
    weerstand.plot_capaciteit_effect_schoonmaak(date_clean, index, date_goal, ax=ax1)

    fig_path = results_dir / "Synthese" / f"Capaciteit - {strang}.png"
    fig.savefig(fig_path, dpi=300)
    logging.info(f"Saved capaciteit result to {fig_path}")

print("hoi")
