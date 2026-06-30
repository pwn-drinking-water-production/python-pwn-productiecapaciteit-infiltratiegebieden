
"""
script to investigate PT10 offset based on low flow ('off') data
"""

# %% imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

from productiecapaciteit import data_dir
from productiecapaciteit.src.strang_analyse_fun2 import get_config, get_false_measurements

# Q100 gedaan. Vanaf daar moet nog.
# %% settings
config = get_config()
config = config.loc[["Q100"]]
res_folder = os.path.abspath(os.path.join(__file__, "..", "..", "results", "PT10offset_fromlowflow"))
os.makedirs(res_folder, exist_ok=True)

# chosen fit type used for PT10 offset correction (-)
fit_choice = {"IK91": "fixed_offset","IK92": "fixed_offset","IK93": "fixed_offset","IK94": "huber_global","IK95": "huber_global","IK96": "huber_global",
"IK101": "huber_global", "IK102": "huber_global", "IK103": "fixed_offset", "IK104": "huber_global", "IK105": "huber_global", "IK106": "huber_global",
"P100": "huber_global", "P200": "fixed_offset", "P300": "huber_multi", "P400": "fixed_offset", "P500": "fixed_offset", "P600": "huber_multi",
"Q100": "huber_multi", "Q200": "fixed_offset", "Q300": "fixed_offset", "Q400": "fixed_offset", "Q500": "fixed_offset", "Q600": "fixed_offset"
}


# minimum number of consecutive low-flow datapoints required for a valid event (-)
min_lowflow_len = {"IK91": 24,"IK92": 12,"IK93": 24,"IK94": 24,"IK95": 12,"IK96": 24,
"IK101": 12,"IK102": 6,"IK103": 6,"IK104": 6,"IK105": 6,"IK106": 12,
"P100": 48,"P200": 24,"P300": 48,"P400": 48,"P500": 12,"P600": 48,
"Q100": 48,"Q200": 48,"Q300": 48,"Q400": 48,"Q500": 48,"Q600": 48}

# weighting exponent for event length (0 = equal weight, 1 = linear with duration) (-)
weighfac = {"IK91": 0.2,"IK92": 0.2,"IK93": 0.2,"IK94": 0.2,"IK95": 0.2,"IK96": 0.2,
"IK101": 0.2,"IK102": 0.2,"IK103": 0.2,"IK104": 0.2,"IK105": 0.2,"IK106": 0.2,
"P100": 0.2,"P200": 0.2,"P300": 0.2,"P400": 0.2,"P500": 0.2,"P600": 0.2,
"Q100": 0.2,"Q200": 0.2,"Q300": 0.2,"Q400": 0.2,"Q500": 0.2,"Q600": 0.2}


# manually defined breakpoints for multiline fit (ISO dates, one list per strang)
breakpoints = {
"IK91": ["2019-06-01"], 
"IK92": [], 
"IK93": [], 
"IK94": [], 
"IK95": [], 
"IK96": [],
"IK101": ["2022-06-01"], 
"IK102": [], 
"IK103": [], 
"IK104": [], 
"IK105": [], 
"IK106": [],
"P100": [], 
"P200": [], 
"P300": ["2018-01-01","2023-01-01"], 
"P400": [], 
"P500": [], 
"P600": ["2014-05-15","2022-01-01"],
"Q100": ["2019-03-01"],
"Q200": [], 
"Q300": [], 
"Q400": [], 
"Q500": [], 
"Q600": []
}

# flow threshold below which strang is considered 'off' (m³/h)
lowflow = {"IK91": 2,"IK92": 2,"IK93": 2,"IK94": 2,"IK95": 2,"IK96": 2,
"IK101": 2,"IK102": 2,"IK103": 2,"IK104": 2,"IK105": 2,"IK106": 2,
"P100": 2,"P200": 2,"P300": 2,"P400": 2,"P500": 2,"P600": 2,
"Q100": 2,"Q200": 2,"Q300": 2,"Q400": 2,"Q500": 2,"Q600": 2}

# maximum allowed slope magnitude per segment in multiline fit (m/day)
max_slope = {"IK91": 0.001,"IK92": 0.001,"IK93": 0.001,"IK94": 0.001,"IK95": 0.001,"IK96": 0.001,
"IK101": 0.001,"IK102": 0.001,"IK103": 0.001,"IK104": 0.001,"IK105": 0.001,"IK106": 0.001,
"P100": 0.001,"P200": 0.001,"P300": 0.001,"P400": 0.001,"P500": 0.001,"P600": 0.001,
"Q100": 0.001,"Q200": 0.001,"Q300": 0.001,"Q400": 0.001,"Q500": 0.001,"Q600": 0.001}

# fallback: fractional position in event used to pick representative datapoint (0=start, 1=end)
pick_frac = {"IK91": 0.9,"IK92": 0.9,"IK93": 0.9,"IK94": 0.9,"IK95": 0.9,"IK96": 0.9,
"IK101": 0.9,"IK102": 0.9,"IK103": 0.9,"IK104": 0.9,"IK105": 0.9,"IK106": 0.9,
"P100": 0.9,"P200": 0.9,"P300": 0.9,"P400": 0.9,"P500": 0.9,"P600": 0.9,
"Q100": 0.9,"Q200": 0.9,"Q300": 0.9,"Q400": 0.9,"Q500": 0.9,"Q600": 0.9}


# %% helpers
def medae(y, yhat):
    return np.median(np.abs(y - yhat))

def huber_fit(x, y, w):
    model = HuberRegressor()
    model.fit(x.reshape(-1, 1), y, sample_weight=w)
    return model.coef_[0], model.intercept_

def get_blocks(df, cut, minlen, frac, weighfac):
    m = df["Q"] < cut
    g = (m != m.shift()).cumsum()

    blocks, picks, weights = [], [], []

    for _, b in df[m].groupby(g):
        n = len(b)
        if n < minlen:
            continue

        blocks.append(b)

        # ---------- stability detection based on gws0 ----------
        dp = b["gws0"].diff()
        dt = b.index.to_series().diff().dt.total_seconds()

        slope = (dp / dt).rolling(3, center=True).median()

        # threshold (m/day → convert to m/s)
        thr = 0.002 / 86400

        stable = (np.abs(slope) < thr)

        # enforce persistence
        stable = stable.rolling(5, center=True).sum() >= 4

        stable_idx = b.index[stable]

        if len(stable_idx) >= 3:
            # ✅ take last 20% of stable region
            n_stable = len(stable_idx)
            tail = stable_idx[int(0.8 * n_stable):]

            # ✅ pick flattest point in that tail
            slope_tail = slope.loc[tail]
            pick_time = slope_tail.abs().idxmin()

        elif len(stable_idx) > 0:
            # fallback: just last stable point
            pick_time = stable_idx[-1]

        else:
            # fallback: original behaviour
            idx = int(np.round(frac * (n - 1)))
            pick_time = b.index[idx]

        # ✅ IMPORTANT: use SAME moment for both PT10 and gws0
        picks.append(b.loc[pick_time])

        weights.append(n ** weighfac)

    return blocks, pd.DataFrame(picks), np.array(weights)


def split_by_breakpoints(t, y, w, strang):

    # if no breakpoints → fallback (same as before)
    if strang not in breakpoints or len(breakpoints[strang]) == 0:
        x = (t.to_series() - t.min()).dt.total_seconds().to_numpy() / 86400
        a, b = huber_fit(x, y.to_numpy(), w)

        # ✅ enforce max slope
        if abs(a) > max_slope[strang]:
            a = np.sign(a) * max_slope[strang]

        return [(t, a, b)]

    bps = pd.to_datetime(breakpoints[strang])
    bps = np.sort(bps)

    segments = []
    t_start = t.min()

    for bp in list(bps) + [t.max()]:

        mask = (t >= t_start) & (t <= bp)

        tseg = t[mask]
        yseg = y.loc[tseg]
        wseg = w[mask]

        if len(yseg) < 2:
            t_start = bp
            continue

        xseg = (tseg.to_series() - tseg.min()).dt.total_seconds().to_numpy() / 86400
        a_seg, b_seg = huber_fit(xseg, yseg.to_numpy(), wseg)

        # ✅ ENFORCE MAX SLOPE HERE
        if abs(a_seg) > max_slope[strang]:
            a_seg = np.sign(a_seg) * max_slope[strang]

            # adjust intercept to keep midpoint stable
            x_mid = xseg.mean()
            y_mid = yseg.mean()
            b_seg = y_mid - a_seg * x_mid

        segments.append((tseg, a_seg, b_seg))

        t_start = bp

    return segments


def build_timeseries_fit(df_index, t_evt, method, strang,
                         a=None, b=None, segments=None, 
                         fixed_offset_val=None):

    t0 = t_evt.min()
    x_full = (df_index - t0).total_seconds() / 86400

    if method == "huber_global":

        ts = pd.Series(a * x_full + b, index=df_index)

        # ✅ flatten outside data range (like multiline)
        tmin = t_evt.min()
        tmax = t_evt.max()

        ts.loc[ts.index < tmin] = ts.loc[tmin]
        ts.loc[ts.index > tmax] = ts.loc[tmax]

        return ts


    elif method == "huber_multi":

        ts = pd.Series(index=df_index, dtype=float)
        anchors_t = []
        anchors_y = []

        for (tseg, a_seg, b_seg) in segments:

            xseg = (tseg - tseg.min()).total_seconds() / 86400
            yseg = a_seg * xseg + b_seg

            anchors_t += [tseg.min(), tseg.max()]
            anchors_y += [yseg[0], yseg[-1]]

            mask = (df_index >= tseg.min()) & (df_index <= tseg.max())
            xloc = (df_index[mask] - tseg.min()).total_seconds() / 86400
            ts.loc[mask] = a_seg * xloc + b_seg

        anchors = pd.Series(anchors_y, index=pd.to_datetime(anchors_t)).sort_index()
        ts.update(anchors)

        return ts.ffill().bfill()

  
    elif method == "fixed_offset":
        return pd.Series(fixed_offset_val, index=df_index)


# %% process
store = {}

for strang, c in config.iterrows():
    print(f"\nProcessing {strang}")

    df = pd.read_feather(data_dir / "Merged" / f"{strang}.feather").set_index("Datum").sort_index()

    bad = get_false_measurements(df, c, extend_hours=1,
                                 include_rules=["Unrealistic flow", "Tijdens spuien"])
    df.loc[bad] = np.nan

    blocks, picks, w = get_blocks(df, lowflow[strang],
                                  min_lowflow_len[strang],
                                  pick_frac[strang],
                                  weighfac[strang])

    if len(blocks) == 0:
        continue

    lf_all = pd.concat(blocks)

    t_evt = picks.index
    y = picks["P"] - picks["gws0"]

    mask = y.notna()
    t_evt = t_evt[mask]
    y = y[mask]
    w = w[mask]
    fixed_offset_val = y.median()
    if not np.isfinite(fixed_offset_val):
        fixed_offset_val = 0.0   # fallback

    x = (t_evt.to_series() - t_evt.min()).dt.total_seconds().to_numpy() / 86400

    # ✅ FIRST compute fits
    a, b = huber_fit(x, y.to_numpy(), w)
    segs = split_by_breakpoints(t_evt, y, w, strang)

    # --- evaluate fits on selected points ---
    # global
    yhat_glob = a * x + b

    # multiline
    yhat_multi = np.empty_like(y)
    for (tseg, a_seg, b_seg) in segs:
        mask = t_evt.isin(tseg)
        xseg = (t_evt[mask].to_series() - tseg.min()).dt.total_seconds().to_numpy() / 86400
        yhat_multi[mask] = a_seg * xseg + b_seg

    # fixed offset
    y_const = fixed_offset_val
    yhat_const = np.full_like(y, y_const)

    # robust scores
    score_glob = medae(y.to_numpy(), yhat_glob)
    score_multi = medae(y.to_numpy(), yhat_multi)
    score_const = medae(y.to_numpy(), yhat_const)

    # fits
    a, b = huber_fit(x, y.to_numpy(), w)
    segs = split_by_breakpoints(t_evt, y, w, strang)
    chosen = fit_choice[strang]

    corr_series = build_timeseries_fit(
        df.index, t_evt, chosen, strang,
        a=a, b=b, segments=segs,
        fixed_offset_val=fixed_offset_val
    )


    corr_series.name = strang

    # corrected PT10 during low-flow
    P_corr_raw = lf_all["P"] - corr_series.reindex(lf_all.index)
    P_corr_sel = P_corr_raw.loc[t_evt]


    store[strang] = dict(
        df=df,
        t_raw=lf_all.index,
        P_raw=lf_all["P"],
        G_raw=lf_all["gws0"],
        dP_raw=lf_all["P"] - lf_all["gws0"],
        t=t_evt,
        dP=y,
        P_corr_raw=P_corr_raw,     
        P_corr_sel=P_corr_sel,     
        fit_global=(a, b),
        fit_segments=segs,
        corr_series=corr_series,
        chosen_fit=chosen,
        fixed_offset_val=fixed_offset_val,
        score_glob=score_glob,
        score_multi=score_multi,
        score_const=score_const,

    )

# %% plotting
fit_colors = {
    "huber_global": "black",
    "huber_multi": "red",
    "fixed_offset": "blue"
}

for strang in store:
    print(f"\nPlotting {strang}")
    # --- manual time window (set to None to disable) ---
    tmin = None# pd.Timestamp("2018-10-01 00:00")
    tmax = None#pd.Timestamp("2019-03-03 00:00")

    d = store[strang]
    fig, ax = plt.subplots(6, 1, figsize=(12, 16), sharex=True)

    # ---------- PT10 ----------
    ax[0].scatter(d["t_raw"], d["P_raw"], color="lightgrey", s=5, label="all data")
    ax[0].scatter(d["t"], d["P_raw"].loc[d["t"]],
                  color="darkgreen", s=30, label="selected")
    ax[0].set_ylabel("PT10 (m)")
    ax[0].legend()

    # ---------- gws0 ----------
    ax[1].scatter(d["t_raw"], d["G_raw"], color="lightgrey", s=5, label="all data")
    ax[1].scatter(d["t"], d["G_raw"].loc[d["t"]],
                  color="darkblue", s=30, label="selected")
    ax[1].set_ylabel("gws0 (m)")
    ax[1].legend()

    # ---------- deltaP full ----------
    ax[2].scatter(d["t_raw"], d["dP_raw"], color="lightgrey", s=5, label="all data")
    ax[2].scatter(d["t"], d["dP"], color="darkmagenta", s=30, label="selected")

    # ---------- deltaP zoom ----------
    ax[3].scatter(d["t_raw"], d["dP_raw"], color="lightgrey", s=5)
    ax[3].scatter(d["t"], d["dP"], color="darkmagenta", s=30)

    # ---------- fits ----------
    t_evt = d["t"]
    a, b = d["fit_global"]
    segs = d["fit_segments"]
    chosen = d["chosen_fit"]

    x = (t_evt.to_series() - t_evt.min()).dt.total_seconds().to_numpy() / 86400
    xl = np.linspace(x.min(), x.max(), 200)
    tl = t_evt.min() + pd.to_timedelta(xl, unit="D")
    yl = a * xl + b

    # --- global ---
    ax[2].plot(tl, yl, color="black", lw=2,
           label=f"global (medAE={d['score_glob']:.3f})")
    ax[3].plot(tl, yl, color="black", lw=2)

    # --- multiline ---
    first = True
    last_t, last_y = None, None

    for (tseg, a_seg, b_seg) in segs:

        xseg = (tseg.to_series() - tseg.min()).dt.total_seconds().to_numpy() / 86400
        xl = np.linspace(xseg.min(), xseg.max(), 100)
        tl_seg = tseg.min() + pd.to_timedelta(xl, unit="D")
        yl_seg = a_seg * xl + b_seg


        if first:
            ax[2].plot(tl_seg, yl_seg, color="red", lw=2, label=f"multiline (medAE={d['score_multi']:.3f})")
            ax[3].plot(tl_seg, yl_seg, color="red", lw=2)
            first = False
        else:
            ax[2].plot(tl_seg, yl_seg, color="red", lw=2)
            ax[3].plot(tl_seg, yl_seg, color="red", lw=2)

            # horizontal step
            ax[2].plot([last_t, tl_seg[0]], [last_y, last_y],
                    color="red", lw=2)
            # vertical jump
            ax[2].plot([tl_seg[0], tl_seg[0]], [last_y, yl_seg[0]],
                    color="red", lw=2)
                        # horizontal step
            ax[3].plot([last_t, tl_seg[0]], [last_y, last_y],
                    color="red", lw=2)
            # vertical jump
            ax[3].plot([tl_seg[0], tl_seg[0]], [last_y, yl_seg[0]],
                    color="red", lw=2)


        last_t = tl_seg[-1]
        last_y = yl_seg[-1]
    
    if strang in breakpoints:
        for bp in pd.to_datetime(breakpoints[strang]):
            ax[2].axvline(bp, color="black", ls="--", alpha=0.4)
            ax[3].axvline(bp, color="black", ls="--", alpha=0.4)

    # --- fixed offset ---
    y_const = d["fixed_offset_val"]

    ax[2].hlines(y_const,
                xmin=d["t_raw"].min(),
                xmax=d["t_raw"].max(),
                color=fit_colors["fixed_offset"],
                lw=2,
                label=f"fixed offset ({y_const:.2f}, medAE={d['score_const']:.3f})")

    ax[3].hlines(y_const,
                xmin=d["t_raw"].min(),
                xmax=d["t_raw"].max(),
                color=fit_colors["fixed_offset"],
                lw=2)


    # ---------- zoom ----------
    y_base = a * x + b
    ax[3].set_ylim(y_base.min() - 0.2, y_base.max() + 0.2)

    ax[2].set_ylabel("ΔP (m)")
    ax[3].set_ylabel("ΔP zoom (m)")
    ax[2].axhline(0, color="grey", ls="--")
    ax[3].axhline(0, color="grey", ls="--")

    # ---------- correction ----------
    corr = d["corr_series"]
    ax[4].scatter(d["t_raw"], d["dP_raw"], color="lightgrey", s=5, label="all data")
    ax[4].scatter(d["t"], d["dP"], color="darkmagenta", s=30, label="selected")

    # fitted line
    ax[4].plot(corr.index, corr,
            color=fit_colors.get(chosen, "black"),
            lw=2, alpha=0.7, label="selected fit: "+chosen)
    ax[4].legend()
    ax[4].set_ylim(y_base.min() - 0.2, y_base.max() + 0.2)
    ax[4].set_ylabel("ΔP zoom (m)")
    # ---------- final legend ----------
    ax[2].legend()

    # ---------- corrected PT10 + gws0 (low flow) ----------
    # gws0
    ax[5].scatter(d["t_raw"], d["G_raw"],
                color="lightblue", s=5, label="gws0 (all)")

    ax[5].scatter(d["t"], d["G_raw"].loc[d["t"]],
                color="darkblue", s=30, label="gws0 (selected)")
        # PT10 corrected
    ax[5].scatter(d["t_raw"], d["P_corr_raw"],
                color="lightgreen", s=5, alpha = 0.4, label="PT10 corr (all)")

    ax[5].scatter(d["t"], d["P_corr_sel"],
                color="darkgreen", s=30, alpha = 0.4, label="PT10 corr (selected)")

    ax[5].set_ylabel("m")
    ax[5].set_xlabel("Time")
    ax[5].legend(ncol=2)
    ax[5].set_title("Low-flow: corrected PT10 vs gws0")
    if tmin is not None and tmax is not None:
        for axi in ax:
            axi.set_xlim(tmin, tmax)
    plt.tight_layout()
    plt.savefig(os.path.join(res_folder, f"{strang}.png"))
    plt.close()

# %% export

print("\nSaving combined correction file...")

corr_all = [store[s]["corr_series"] for s in store]

df_corr = pd.concat(corr_all, axis=1).sort_index()

out_fp = os.path.join(res_folder, "PT10_corr_all.feather")

df_corr.reset_index().rename(columns={"index": "Datum"}).to_feather(out_fp)

print(f"Saved → {out_fp}")



