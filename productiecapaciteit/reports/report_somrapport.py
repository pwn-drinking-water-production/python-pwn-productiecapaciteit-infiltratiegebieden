import logging
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd

from productiecapaciteit.src.strang_analyse_fun2 import get_config
from productiecapaciteit.src.capaciteit_strang import strangWeerstand

from productiecapaciteit.src.weerstand_pandasaccessors import LeidingResistanceAccessor
from productiecapaciteit.src.weerstand_pandasaccessors import WellResistanceAccessor
from productiecapaciteit.src.weerstand_pandasaccessors import WvpResistanceAccessor

res_folder = os.path.join("..", "results", "Synthese", "Capaciteit")

data_fd = os.path.join("..", "Data")
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

index = pd.date_range("2012-05-01", "2023-01-01")

w_all = dict()
w_offset = {k: dict() for k in [-3, -2, -1, 0, 1, 2, 3]}

for strang, c in config.iterrows():
    df_a_filter = pd.read_excel(filterweerstand_fp, sheet_name=strang, index_col=0)
    df_a_leiding = pd.read_excel(leidingweerstand_fp, sheet_name=strang, index_col=0)
    df_a_wvp = pd.read_excel(wvpweerstand_fp, sheet_name=strang, index_col=0).squeeze(
        "columns"
    )

    w_all[strang] = strangWeerstand(df_a_leiding, df_a_filter, df_a_wvp, **c.to_dict())
    for temp_opwarming in w_offset:
        w_offset[temp_opwarming][strang] = strangWeerstand(
            df_a_leiding,
            df_a_filter,
            df_a_wvp,
            temp_opwarming=temp_opwarming,
            **c.to_dict())

lims = pd.DataFrame({k: v.capaciteit(index) for k, v in w_all.items()})
pd.DataFrame({k: v.capaciteit(index) for k, v in w_all.items()}).plot.area()

fig_path = os.path.join(res_folder, f"Capaciteit Som - {strang}.png")
plt.savefig(fig_path, dpi=300)

"""
Effect opwarming
"""
plt.style.use(['unhcrpyplotstyle', 'line'])
res_folder = os.path.join("..", "results", "Synthese", "Opwarming")

lims_sum_dict = dict()
lims_offset = dict()
for temp_opwarming, wi in w_offset.items():
    lims = pd.DataFrame({k: v.capaciteit(index) for k, v in wi.items()})
    lims_offset[temp_opwarming] = lims

    lims_sum_dict[temp_opwarming] = lims.sum(axis=1)

lims_sum = pd.DataFrame(lims_sum_dict)
dlims_sum = lims_sum - lims_sum[[0]].values
dlims_sum_frac = dlims_sum / lims_sum[[0]].values * 100


fig, ax = plt.subplots(figsize=(8.5, 5.75))
ax.set_title('Effect van opwarming watervoerendpakket op productiecap.\n2012-2022', pad=40)
dlims_sum_frac.rename(columns={k: f"{k}$^\circ$C" for k in dlims_sum_frac}).plot(ax=ax)
ax.legend(loc=(0, 1), ncol=7)
ax.set_ylabel('Toename in productiecap. (%)')
fig.tight_layout()

fig_path = os.path.join(res_folder, f"Effect opwarming.png")
fig.savefig(fig_path, dpi=300)


"""
Effect opwarming zoom
"""
plt.style.use(['unhcrpyplotstyle', 'line'])
res_folder = os.path.join("..", "results", "Synthese", "Opwarming")

lims_sum_dict = dict()
lims_offset = dict()
for temp_opwarming, wi in w_offset.items():
    lims = pd.DataFrame({k: v.capaciteit(index) for k, v in wi.items()})
    lims_offset[temp_opwarming] = lims

    lims_sum_dict[temp_opwarming] = lims.sum(axis=1)

lims_sum = pd.DataFrame(lims_sum_dict)
dlims_sum = lims_sum - lims_sum[[0]].values
dlims_sum_frac = dlims_sum / lims_sum[[0]].values * 100


fig, ax = plt.subplots(figsize=(8.5, 5.75))
ax.set_title('Effect van opwarming watervoerendpakket op productiecap.\n2021', pad=40)
dlims_sum_frac.rename(columns={k: f"{k}$^\circ$C" for k in dlims_sum_frac}).plot(ax=ax)
ax.legend(loc=(0, 1), ncol=7)
ax.set_ylabel('Toename in productiecap. (%)')
ax.set_xlim(("2021-01-01", "2022-01-01"))
fig.tight_layout()

fig_path = os.path.join(res_folder, f"Effect opwarming zoom.png")
fig.savefig(fig_path, dpi=300)


"""
Effect opwarming zoom
"""
plt.style.use(['unhcrpyplotstyle', 'line'])
res_folder = os.path.join("..", "results", "Synthese", "Opwarming")

lims_sum_dict = dict()
lims_offset = dict()
for temp_opwarming, wi in w_offset.items():
    lims = pd.DataFrame({k: v.capaciteit(index) for k, v in wi.items()})
    lims_offset[temp_opwarming] = lims

    lims_sum_dict[temp_opwarming] = lims.sum(axis=1)

lims_sum = pd.DataFrame(lims_sum_dict)
dlims_sum = lims_sum - lims_sum[[0]].values
dlims_sum_frac = dlims_sum / lims_sum[[0]].values * 100


fig, ax = plt.subplots(figsize=(8.5, 5.75))
ax.set_title('Effect van opwarming watervoerendpakket op productiecap.\n2021', pad=40)
dlims_sum_frac.rename(columns={k: f"{k}$^\circ$C" for k in dlims_sum_frac}).plot(ax=ax)
ax.legend(loc=(0, 1), ncol=7)
ax.set_ylabel('Toename in productiecap. (%)')
ax.set_xlim(("2021-01-01", "2022-01-01"))
fig.tight_layout()

fig_path = os.path.join(res_folder, f"Effect opwarming zoom - geen pomplimiet.png")
fig.savefig(fig_path, dpi=300)


print("hoi")