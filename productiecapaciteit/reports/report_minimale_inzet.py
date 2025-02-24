import os

import pandas as pd

from productiecapaciteit import results_dir
from productiecapaciteit.src.capaciteit_strang import strangWeerstand
from productiecapaciteit.src.strang_analyse_fun2 import (
    get_config,
)
from productiecapaciteit.src.weerstand_pandasaccessors import (
    LeidingResistanceAccessor,  # noqa: F401
    WellResistanceAccessor,  # noqa: F401
    WvpResistanceAccessor,  # noqa: F401
)

data_fd = os.path.join("..", "data")
config_fn = "strang_props7.csv"
config = get_config(os.path.join(data_fd, config_fn))
config = config.loc[:, config.columns.notna()]

filterweerstand_fp = results_dir / "Filterweerstand" / "Filterweerstand_modelcoefficienten.xlsx"
leidingweerstand_fp = results_dir / "Leidingweerstand" / "Leidingweerstand_modelcoefficienten.xlsx"
wvpweerstand_fp = results_dir / "Wvpweerstand" / "Wvpweerstand_modelcoefficienten.xlsx"

deltah_veilig = 2.0
index = pd.date_range("2024-01-01", "2024-12-31")

# Create output file
output_file = results_dir / "Synthese" / "min_max_table.txt"

# Write header to file
with open(output_file, mode='w', encoding="utf-8") as f:
    f.write(f"Strang\tQ_min [m3/h] (model;Δh={deltah_veilig:.1f}m)\tQ_min [m3/h] (inzetvolgorde feb 2025)\tQ_max [m3/h] (model)\tQ_max [m3/h] (inzetvolgorde 2023)\tAantal putten\tQ_min [m3/h] (model;Δh=1,5m)\tQ_min [m3/h] (inzetvolgorde 2023)\tQ_max [m3/h] (model)\tQ_max [m3/h] (inzetvolgorde 2023)\n")
    
    for strang, c in config.iterrows():
        # if strang != "IK105":
        #     continue

        df_a_filter = pd.read_excel(filterweerstand_fp, sheet_name=strang)
        df_a_leiding = pd.read_excel(leidingweerstand_fp, sheet_name=strang)
        df_a_wvp = pd.read_excel(wvpweerstand_fp, sheet_name=strang, index_col=0).squeeze("columns")

        weerstand = strangWeerstand(df_a_leiding, df_a_filter, df_a_wvp, **c.to_dict())
        flow_mean = weerstand.capaciteit(index).median()

        flow_min = weerstand.lim_flow_min(index, deltah_veilig=deltah_veilig)
        cap = weerstand.capaciteit(index)

        f.write(
            f"{strang}\t"
            f"{flow_min.max() / weerstand.nput:0.1f}\t"
            f"{weerstand.Qmin_inzetvolgorde20250219:.1f}\t"
            f"{cap.min() / weerstand.nput:0.1f}\t"
            f"{weerstand.Qmax_inzetvolgorde20250219:.1f}\t"
            f"{weerstand.nput:.0f}\t"
            f"{flow_min.max():0.0f}\t"
            f"{weerstand.Qmin_inzetvolgorde20250219 * weerstand.nput:.0f}\t"
            f"{cap.min():0.0f}\t"
            f"{weerstand.Qmax_inzetvolgorde20250219 * weerstand.nput:.0f}\t\n"
        )
        # print(
        #     f"{strang}\t"
        #     f"{flow_min.max() / weerstand.nput:0.1f}\t"
        #     f"{weerstand.Qmin_inzetvolgorde20230523:.1f}\t"
        #     f"{cap.min() / weerstand.nput:0.1f}\t"
        #     f"{weerstand.Qmax_inzetvolgorde20230523:.1f}\t"
        #     f"{flow_min.max():0.0f}\t"
        #     f"{weerstand.Qmin_inzetvolgorde20230523 * weerstand.nput:.0f}\t"
        #     f"{cap.min():0.0f}\t"
        #     f"{weerstand.Qmax_inzetvolgorde20230523 * weerstand.nput:.0f}\t"
        # )

        # df_fp = os.path.join(data_fd, "Merged", f"{strang}-PKA-DSEW036680.feather")
        # df = pd.read_feather(df_fp)
        # df["Datum"] = pd.to_datetime(df["Datum"])
        # df.set_index("Datum", inplace=True)
        # df = df.resample("1H").mean()
        #
        # # include_rules = ["Unrealistic flow", "Niet 1-day steady"]
        # include_rules = ["Unrealistic flow"]
        # untrusted_measurements = get_false_measurements(
        #     df, c, extend_hours=1, include_rules=include_rules
        # )
        # df.loc[untrusted_measurements] = np.nan


# Qmax_inzetvolgorde20230523
print("hoi")
