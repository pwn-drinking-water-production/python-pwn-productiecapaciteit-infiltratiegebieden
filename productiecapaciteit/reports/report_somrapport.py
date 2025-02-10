import matplotlib.pyplot as plt
import pandas as pd

from productiecapaciteit import plot_styles_dir, results_dir
from productiecapaciteit.src.capaciteit_strang import strangWeerstand
from productiecapaciteit.src.strang_analyse_fun2 import get_config
from productiecapaciteit.src.weerstand_pandasaccessors import (
    LeidingResistanceAccessor,  # noqa: F401
    WellResistanceAccessor,  # noqa: F401
    WvpResistanceAccessor,  # noqa: F401
)

plt.style.use(plot_styles_dir / "unhcrpyplotstyle.mplstyle")
plt.style.use(plot_styles_dir / "types" / "line.mplstyle")

config = get_config()
config = config.loc[:, config.columns.notna()]

filterweerstand_fp = results_dir / "Filterweerstand" / "Filterweerstand_modelcoefficienten.xlsx"
leidingweerstand_fp = results_dir / "Leidingweerstand" / "Leidingweerstand_modelcoefficienten.xlsx"
wvpweerstand_fp = results_dir / "Wvpweerstand" / "Wvpweerstand_modelcoefficienten.xlsx"

index = pd.date_range("2012-05-01", "2026-12-31")

w_all = {}
w_offset: dict[int, dict] = {k: {} for k in [-3, -2, -1, 0, 1, 2, 3]}

for strang, c in config.iterrows():
    df_a_filter = pd.read_excel(filterweerstand_fp, sheet_name=strang, index_col=0)
    df_a_leiding = pd.read_excel(leidingweerstand_fp, sheet_name=strang, index_col=0)
    df_a_wvp = pd.read_excel(wvpweerstand_fp, sheet_name=strang, index_col=0).squeeze("columns")

    w_all[strang] = strangWeerstand(df_a_leiding, df_a_filter, df_a_wvp, **c.to_dict())
    for temp_opwarming in w_offset:
        w_offset[temp_opwarming][strang] = strangWeerstand(
            df_a_leiding, df_a_filter, df_a_wvp, temp_opwarming=temp_opwarming, **c.to_dict()
        )

lims = pd.DataFrame({k: v.capaciteit(index) for k, v in w_all.items()})
pd.DataFrame({k: v.capaciteit(index) for k, v in w_all.items()}).plot.area()

fig_path = results_dir / "Synthese" / "Capaciteit" / "Capaciteit Som.png"
plt.savefig(fig_path, dpi=300)

"""
Effect opwarming
"""
lims_sum_dict = {}
lims_offset = {}
for temp_opwarming, wi in w_offset.items():
    lims = pd.DataFrame({k: v.capaciteit(index) for k, v in wi.items()})
    lims_offset[temp_opwarming] = lims

    lims_sum_dict[temp_opwarming] = lims.sum(axis=1)

lims_sum = pd.DataFrame(lims_sum_dict)
dlims_sum = lims_sum - lims_sum[[0]].values
dlims_sum_frac = dlims_sum / lims_sum[[0]].values * 100


fig, ax = plt.subplots(figsize=(8.5, 5.75))
ax.set_title("Effect van opwarming watervoerendpakket op productiecap.\n2012-2022", pad=40)
dlims_sum_frac.rename(columns={k: rf"{k}$^\circ$C" for k in dlims_sum_frac}).plot(ax=ax)
ax.legend(loc=(0, 1), ncol=7)
ax.set_ylabel("Toename in productiecap. (%)")
fig.tight_layout()

fig_path = results_dir / "Synthese" / "Opwarming" / "Effect opwarming.png"
fig.savefig(fig_path, dpi=300)

"""
Effect opwarming zoom
"""
lims_sum_dict = {}
lims_offset = {}
for temp_opwarming, wi in w_offset.items():
    lims = pd.DataFrame({k: v.capaciteit(index) for k, v in wi.items()})
    lims_offset[temp_opwarming] = lims

    lims_sum_dict[temp_opwarming] = lims.sum(axis=1)

lims_sum = pd.DataFrame(lims_sum_dict)
dlims_sum = lims_sum - lims_sum[[0]].values
dlims_sum_frac = dlims_sum / lims_sum[[0]].values * 100


fig, ax = plt.subplots(figsize=(8.5, 5.75))
ax.set_title("Effect van opwarming watervoerendpakket op productiecap.\n2021", pad=40)
dlims_sum_frac.rename(columns={k: rf"{k}$^\circ$C" for k in dlims_sum_frac}).plot(ax=ax)
ax.legend(loc=(0, 1), ncol=7)
ax.set_ylabel("Toename in productiecap. (%)")
ax.set_xlim(("2021-01-01", "2022-01-01"))
fig.tight_layout()

fig_path = results_dir / "Synthese" / "Opwarming" / "Effect opwarming zoom.png"
fig.savefig(fig_path, dpi=300)

"""
Effect opwarming zoom except pompcapaciteit
{
    "Pompcapaciteit": self.Qpomp,
    "Vacuumsysteem": self.lim_vac(index),
    "Luchthappen": self.lim_luchthap(index),
    "Verblijftijd": self.lim_verblijf(index),
}
"""
use_lims = ["Vacuumsysteem", "Luchthappen", "Verblijftijd"]

lims_sum_dict = {}
lims_offset = {}
for temp_opwarming, wi in w_offset.items():
    lims = pd.DataFrame({k: v.capaciteit(index, use_lims=use_lims) for k, v in wi.items()})
    lims_offset[temp_opwarming] = lims

    lims_sum_dict[temp_opwarming] = lims.sum(axis=1)

lims_sum = pd.DataFrame(lims_sum_dict)
dlims_sum = lims_sum - lims_sum[[0]].values
dlims_sum_frac = dlims_sum / lims_sum[[0]].values * 100


fig, ax = plt.subplots(figsize=(8.5, 5.75))
ax.set_title("Effect van opwarming watervoerendpakket op productiecap.\n2021", pad=40)
dlims_sum_frac.rename(columns={k: rf"{k}$^\circ$C" for k in dlims_sum_frac}).plot(ax=ax)
ax.legend(loc=(0, 1), ncol=7)
ax.set_ylabel("Toename in productiecap. (%)")
ax.set_xlim(("2021-01-01", "2022-01-01"))
fig.tight_layout()

fig_path = results_dir / "Synthese" / "Opwarming" / "Effect opwarming zoom - geen pomplimiet.png"
fig.savefig(fig_path, dpi=300)
print("hoi")
