import pandas as pd

from productiecapaciteit import results_dir
from productiecapaciteit.src.capaciteit_strang import strangWeerstand
from productiecapaciteit.src.strang_analyse_fun2 import get_config
from productiecapaciteit.src.weerstand_pandasaccessors import (
    LeidingResistanceAccessor,  # noqa: F401
    WellResistanceAccessor,  # noqa: F401
    WvpResistanceAccessor,  # noqa: F401
)

config = get_config()

filterweerstand_fp = results_dir / "Filterweerstand" / "Filterweerstand_modelcoefficienten.xlsx"
leidingweerstand_fp = results_dir / "Leidingweerstand" / "Leidingweerstand_modelcoefficienten.xlsx"
wvpweerstand_fp = results_dir / "Wvpweerstand" / "Wvpweerstand_modelcoefficienten.xlsx"

index = pd.date_range("2012-05-01", "2025-12-31")
date_clean = pd.Timestamp("2025-04-01")
date_goal = pd.Timestamp("2025-10-01")

som_dict: dict[str, float] = {}
report: dict[str, float] = {}

fp = results_dir / "Synthese" / "Capaciteit" / "Effect_schoonmaak_tabel.txt"
with open(fp, "w", encoding="utf-8") as f:
    f.write("Strang\tEffect (m3/h)\tWelke schoonmaken?\tLimieten\tLimieten_na_schoonmaak\n")
    for strang, c in config.iterrows():
        df_a_filter = pd.read_excel(filterweerstand_fp, sheet_name=strang)
        df_a_leiding = pd.read_excel(leidingweerstand_fp, sheet_name=strang)
        df_a_wvp = pd.read_excel(wvpweerstand_fp, sheet_name=strang, index_col=0).squeeze("columns")

        weerstand = strangWeerstand(df_a_leiding, df_a_filter, df_a_wvp, **c.to_dict())
        effect_som, effect_dict = weerstand.report_capaciteit_effect_schoonmaak(date_clean, [date_goal])
        print(effect_som)
        frac = effect_dict["ratio_lei"].item()
        cap_min = weerstand.capaciteit(index).min()

        lims = weerstand.lims([date_clean]).iloc[0]
        cap = lims.min()
        # Maak een lijst van alle limieten die dicht bij de capaciteit liggen.
        lim_cats = lims[lims < cap * 1.1].index

        lims_schoonmaak = weerstand.lims_schoonmaak(date_clean, [date_goal], leiding=True, wel=True).iloc[0]
        cap_schoonmaak = lims_schoonmaak.min()
        lim_cats_schoonmaak = lims_schoonmaak[lims_schoonmaak < cap_schoonmaak * 1.1].index

        if not cap_min or effect_som.item() / cap_min > 0.025:
            if frac > 0.1 and frac < 0.9:
                welke = "Leiding en filter"

            elif frac > 0.1:
                welke = "Leiding"

            elif frac < 0.9:
                welke = "Filter"

            s = f"{strang}\t{effect_som.item():.0f}\t{welke}\t{', '.join(lim_cats)}\t{', '.join(lim_cats_schoonmaak)}"
        else:
            s = f"{strang}\tNIHIL\t-\t{', '.join(lim_cats)}\t-"

        f.write(s + "\n")
