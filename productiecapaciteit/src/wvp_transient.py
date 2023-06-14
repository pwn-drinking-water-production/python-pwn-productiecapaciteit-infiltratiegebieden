import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import hmean
from productiecapaciteit.src.capaciteit_strang import strangWeerstand
import numpy as np
from scipy.optimize import least_squares
from scipy import linalg
from dawacotools import get_daw_ts_meteo

from productiecapaciteit.src.strang_analyse_fun2 import (
    get_config,
    get_false_measurements,
)
from wvp_transient_funs import objective
from wvp_transient_funs import dis

from productiecapaciteit.src.weerstand_pandasaccessors import LeidingResistanceAccessor
from productiecapaciteit.src.weerstand_pandasaccessors import WellResistanceAccessor
from productiecapaciteit.src.weerstand_pandasaccessors import WvpResistanceAccessor


data_fd = os.path.join("..", "data")
config_fn = "strang_props6.xlsx"
config = get_config(os.path.join(data_fd, config_fn))

r = 0.6
kD0 = 200.0  # m/d2 at reference temperature
c = 50.0  # d
S = 0.2

strang = "Q200" # "IK102"
ci = config.loc[strang]

# get observations
df_fp = os.path.join(data_fd, "Merged", f"{strang}-PKA-DSEW036680.feather")
df = pd.read_feather(df_fp)
df["Datum"] = pd.to_datetime(df["Datum"])
df.set_index("Datum", inplace=True)
include_rules = ["Unrealistic flow"]
untrusted_measurements = get_false_measurements(
    df, ci, extend_hours=1, include_rules=include_rules
)
df = df.loc[~np.array(untrusted_measurements)]
df = df.loc[~np.isnan(df.Q)]
dfm = df.resample("12h", label="right").mean()
dfm = dfm.loc[~np.isnan(dfm.Q)]

# get initial estimates
filterweerstand_fp = os.path.join(
    "..", "results", "Filterweerstand", "Filterweerstand_modelcoefficienten.xlsx"
)
wvpweerstand_fp = os.path.join(
    "..", "results", "Wvpweerstand", "Wvpweerstand_modelcoefficienten.xlsx"
)
df_a_wvp = pd.read_excel(wvpweerstand_fp, sheet_name=strang, index_col=0).squeeze(
    "columns"
)


# initial, lower limit, upper limit
params = dict(
    alpha=[(r**2 * S / 4) ** 0.5, 0.0, 100 * (r**2 * S / 4) ** 0.5],
    beta=[(1 / (c * S)) ** 0.5, 0.0, (1 / (c * S)) ** 0.5],
    kD0=[140.0, 50.0, 500.0],
    time_delta=[df_a_wvp.temp_delta, 0.0, 10.0],
    temp_time_offset=[df_a_wvp.time_offset, 31 + 2 * 20, 365],
    # alpha_multi=[(1**2 * S / 4) ** 0.5, 0.0, 100 * (1**2 * S / 4) ** 0.5],
    alpha_multi=[2.7, 0.1, 25.]
)

dx_put = int(ci.dx_tussenputten)
dx_mirrorwell = pd.eval(ci.dx_mirrorwell)  # [li for li in pd.eval(ci.dx_mirrorwell) if li[1] < 200]
nput_model = 15

multiwell = (
    [(li[0], 2 * li[1]) for li in dx_mirrorwell] +
    [(2, i * dx_put) for i in range(1, nput_model + 1)] +
    [(li[0], 2 * dis(i * dx_put, 2 * li[1])) for li in dx_mirrorwell for i in range(1, nput_model + 1)]
)
ds_rain = get_daw_ts_meteo('235', "Neerslag")
pextra = dict(
    index=dfm.index.values[:-1],
    drawdown_obs=(dfm.pandpeil - dfm.gws0).values[:-1],
    Q_obs=dfm.Q.values[1:] / ci.nput,
    temp_ref=dfm.gwt0.median(),
    dt_lower=pd.Timedelta(value=hmean(np.diff(dfm.index) / pd.Timedelta(1, unit='D')), unit="D"),
    multiwell_contains_r_self=False,
    multiwell=multiwell,
    # rain=np.interp(dfm.index[1:], ds_rain.index, ds_rain),
    frac_step_max=0.95
)
assert np.isnan(pextra["Q_obs"]).sum() == 0

args = [i[0] for i in params.values()]

res = least_squares(
    objective,
    x0=[i[0] for i in params.values()],
    bounds=([i[1] for i in params.values()], [i[2] for i in params.values()]),
    # loss="arctan",
    f_scale=0.5,
    kwargs=pextra
)

def get_perr(res):
    if np.any(res.active_mask):
        print(f"{res.active_mask} True for params at bounds")
    U, s, Vh = linalg.svd(res.jac, full_matrices=False)
    tol = np.finfo(float).eps * s[0] * max(res.jac.shape)
    w = s > tol
    cov = (Vh[w].T / s[w]**2) @ Vh[w]  # robust covariance matrix
    chi2dof = np.sum(res.fun**2) / (res.fun.size - res.x.size)
    cov *= chi2dof
    perr = np.sqrt(np.diag(cov))
    perr_rel = perr / res.x

    sl = []
    for xi, perr_ri in zip(res.x, perr_rel):
        sl.append(f"{xi} +/- {perr_ri * 100:.1f}%")

    print('\n'.join(sl))
    return perr


perr = get_perr(res)
a = objective(res.x, return_result=True, **pextra)
plt.plot(pextra["index"], pextra["drawdown_obs"])
plt.plot(pextra["index"], a)

"""
0.11959938975024936 +/- 121.1%
0.3798944820153483 +/- 6.6%
179.63725616239432 +/- 59.3%
6.058479229670381 +/- 5.5%
176.36014752388652 +/- 0.9%
0.16542716576149427 +/- 44.2%
"""
# a = objective((alpha, beta, kD0, temp_delta, temp_time_offset))
print("hoi")
