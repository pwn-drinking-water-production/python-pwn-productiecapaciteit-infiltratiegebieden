import logging
import os
from datetime import timedelta, datetime
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from strang_analyse_fun2 import (
    get_config,
    get_false_measurements,
)

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


class strangWeerstand(object):
    def __init__(
        self,
        df_a_leiding,
        df_a_filter,
        df_a_wvp,
        druk_limiet=-7.5,
        **kwargs,
    ):
        self.__dict__.update(kwargs)
        self._df_leiding = df_a_leiding
        self._df_filter = df_a_filter
        self._df_wvp = df_a_wvp
        self.druk_limiet = druk_limiet  # onderdruk vacuumsysteem
        self.test_dates = {
            # 'test2009': datetime(2009, 8, 18),
            "test2015": datetime(2015, 6, 30),
            "test2021": datetime(2021, 6, 14),
        }

    @property
    def lei(self):
        return self._df_leiding

    @property
    def fil(self):
        return self._df_filter

    @property
    def wvp(self):
        return self._df_wvp

    def get_leiding_schoonmaak_dates(self):
        return self.lei.datum

    def get_wel_schoonmaak_dates(self):
        return self.fil.datum

    def get_schoonmaak_dates(self):
        return np.unique(
            np.concatenate(
                (
                    self.get_leiding_schoonmaak_dates()[1:],
                    self.get_wel_schoonmaak_dates()[1:],
                )
            )
        )

    def add_leiding_schoonmaak(self, date, method="mean"):
        date = pd.to_datetime(date)
        slope = self.lei.slope.iloc[-1]
        offset = self.lei.leiding.a_na_projectie(date, method=method)
        self.lei.loc[len(self.lei)] = dict(
            datum=date, slope=slope, offset=offset, gewijzigd=pd.Timestamp.now()
        )
        pass

    def add_wel_schoonmaak(self, date, method="mean"):
        date = pd.to_datetime(date)
        slope = self.fil.slope.iloc[-1]
        offset = self.fil.wel.a_na_projectie(date, method=method)

        self.fil.loc[len(self.fil)] = dict(
            datum=date, slope=slope, offset=offset, gewijzigd=pd.Timestamp.now()
        )
        pass

    def lim_vac(self, index):
        a_leiding = self.lei.leiding.a_model(index)
        a_filter = self.fil.wel.a_model(index) / self.nput
        a_wvp = self.wvp.wvp.a_model(index)

        c_abc = self.hpand - self.hverbindingvacuum - self.druk_limiet
        discr = (a_filter + a_wvp) ** 2 - (4 * a_leiding * c_abc)
        return (-(a_filter + a_wvp) - discr**0.5) / (2 * a_leiding)

    def lim_luchthap(self, index):
        a_wvp = self.wvp.wvp.a_model(index)
        a_filter = self.fil.wel.a_model(index) / self.nput
        return (self.hluchthappen - self.hpand) / (a_wvp + a_filter)

    def lim_verblijf(self, index):
        return pd.Series(data=self.nput * self.Qlim_bio_per_put, index=index)

    def lims(self, index):
        return pd.DataFrame(
            {
                "Pompcapaciteit": self.Qpomp,
                "Vacuumsysteem": self.lim_vac(index),
                "Luchthappen": self.lim_luchthap(index),
                "Verblijftijd": self.lim_verblijf(index),
            }
        )

    def lim_flow_min(self, index, deltah_veilig=1.5):
        a_wvp = self.wvp.wvp.a_model(index)
        flow_min = -(self.hpand - self.hmaaiveld + deltah_veilig) / a_wvp
        flow_min[flow_min < 0.0] = 0
        return flow_min

    def capaciteit(self, index):
        return self.lims(index).min(axis=1)

    def capaciteit_cat(self, index):
        lims = self.lims(index)
        values = lims.idxmin(axis=1)
        categories = lims.columns

        return pd.Categorical(values=values, categories=categories)

    def get_schoonmaakscenario(self, date_clean, leiding=True, wel=True):
        strang2 = deepcopy(self)

        if leiding:
            strang2.add_leiding_schoonmaak(date_clean)

        if wel:
            strang2.add_wel_schoonmaak(date_clean)

        return strang2

    def capaciteit_schoonmaak(self, date_clean, date_test, leiding=True, wel=True):
        strang2 = self.get_schoonmaakscenario(date_clean, leiding=leiding, wel=wel)
        return strang2.capaciteit(date_test)

    def capaciteit_effect_schoonmaak(
        self, date_clean, date_test, leiding=True, wel=True
    ):
        cap_na_schoonmaak = self.capaciteit_schoonmaak(
            date_clean, date_test, leiding=leiding, wel=wel
        )

        return cap_na_schoonmaak - self.capaciteit(date_test)

    def capaciteit_effect_schoonmaak_cat(
        self, date_clean, date_test, leiding=True, wel=True
    ):
        strang2 = self.get_schoonmaakscenario(date_clean, leiding=leiding, wel=wel)
        return strang2.capaciteit_cat(date_test)

    def plot_lims(self, index, date_clean, ax=None, deltah_veilig=1.5):
        lims = self.lims(index)
        for ilim, lim in enumerate(lims):
            ax.plot(index, lims[lim], lw=0.8, label=lim, c=f"C{ilim}")

        cap = self.capaciteit(index)
        nlims = lims.columns.size
        ax.plot(index, cap, lw=2, label="Zonder schoonmaak", c=f"C{nlims}", ls="--")

        index_na = index[index >= date_clean]
        cap_na_schoonmaak = self.capaciteit_schoonmaak(
            date_clean, index_na, leiding=True, wel=True
        )
        ax.plot(
            index_na,
            cap_na_schoonmaak,
            lw=2,
            label="Na schoonmaak",
            c=f"C{nlims + 1}",
            ls="--",
        )

        flow_min = self.lim_flow_min(index, deltah_veilig=deltah_veilig)
        ax.plot(
            index, flow_min, label="Minimale inzet", lw=2, c=f"C{nlims + 2}", ls="--"
        )

        dates = weerstand.test_dates.values()
        flows = [weerstand.__dict__[key] for key in weerstand.test_dates.keys()]
        ax.scatter(dates, flows, label="Gemeten", c="black")

        self.plot_schoonmaak(ax, [date_clean], label="Schoonmaak moment")
        self.plot_schoonmaak(ax, self.get_schoonmaak_dates(), label="")

        ax.legend(fontsize="small", loc="upper left", title="Capaciteit")
        ax.set_ylabel("Capaciteit strang (m$^3$/h)")

        self.plot_lim_cat(
            date_clean,
            index,
            ax=ax,
            y1=0.05,
            y2=0.095,
            leiding=False,
            wel=False,
            info=None,
        )
        self.plot_lim_cat(
            date_clean,
            index_na,
            ax=ax,
            y1=0.0,
            y2=0.045,
            leiding=True,
            wel=True,
            legend=False,
        )

        months = mdates.MonthLocator((1, 4, 7, 10))
        ax.xaxis.set_minor_locator(months)

        ax2 = ax.twinx()
        ylim = ax.get_ylim()
        ax2.set_ylim((ylim[0] / self.nput, ylim[1] / self.nput))
        ax2.set_ylabel("Capaciteit put (m$^3$/h)")
        pass

    def plot_capaciteit_effect_schoonmaak(
        self, date_clean, date_test, ax=None, y0=0, thick=0.08, legend=True
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5), gridspec_kw=gridspec_kw)

        effect_leiding = self.capaciteit_effect_schoonmaak(
            date_clean, date_test, leiding=True, wel=False
        )
        effect_wel = self.capaciteit_effect_schoonmaak(
            date_clean, date_test, leiding=False, wel=True
        )
        effect_som = self.capaciteit_effect_schoonmaak(
            date_clean, date_test, leiding=True, wel=True
        )

        # grove schatting van contributie van schoonmaak leidingen/putten
        # Leidingweerstanden gaan kwadratisch en tellen dus niet lekker op
        # strang_leiding + strang_wel != strang_som
        frac_lei = effect_leiding / (effect_leiding + effect_wel) * effect_som
        frac_wel = effect_wel / (effect_leiding + effect_wel) * effect_som

        ax.fill_between(date_test, y1=frac_lei, y2=y0, label="Leiding")
        ax.fill_between(date_test, y1=effect_som, y2=frac_lei, label="Put")
        ax.plot(date_test, effect_som, c="black", lw=0.8, label="")

        self.plot_schoonmaak(ax, [date_clean], label="Schoonmaak moment")
        self.plot_schoonmaak(ax, self.get_schoonmaak_dates(), label="")
        ax.legend(fontsize="small", loc="upper left", title="Effect schoonmaak")
        ylim = ax.get_ylim()
        ax.set_ylim((ylim[1] - (1 + thick) * (ylim[1] - ylim[0]), ylim[1]))
        ax.set_ylabel("Extra capaciteit tgv schoonmaak (m$^3$/h)")

        years = mdates.YearLocator()  # every year
        ax.xaxis.set_major_locator(years)
        months = mdates.MonthLocator((1, 4, 7, 10))
        ax.xaxis.set_minor_locator(months)

        self.plot_lim_cat(
            date_clean,
            date_test,
            ax=ax,
            y1=0.0,
            y2=thick,
            leiding=True,
            wel=True,
            # info="",
        )
        ax2 = ax.twinx()
        ylim = ax.get_ylim()
        ax2.set_ylim((ylim[0] / self.nput, ylim[1] / self.nput))
        ax2.set_ylabel("Extra capaciteit tgv schoonmaak put (m$^3$/h)")

        return ax

    @staticmethod
    def plot_schoonmaak(ax, dates, label="Schoonmaak moment"):
        # Helper
        ax.vlines(
            dates,
            ls=":",
            lw=5,
            ymin=0,
            ymax=1,
            linewidth=2,
            color="red",
            transform=ax.get_xaxis_transform(),
            label=label,
        )

    def plot_lim_cat(
        self,
        date_clean,
        date_test,
        ax=None,
        y1=0.0,
        y2=1.0,
        leiding=True,
        wel=True,
        info=None,
        legend=True,
    ):
        # Helper
        # plots limit category to twinx axis

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5), gridspec_kw=gridspec_kw)

        ax2 = ax.twinx()
        transform = ax2.get_xaxis_transform()
        lim_cat = self.capaciteit_effect_schoonmaak_cat(
            date_clean, date_test, leiding=leiding, wel=wel
        )

        for icat, cat in enumerate(lim_cat.categories):
            where = lim_cat == cat
            ax2.fill_between(
                date_test,
                y1=y1,
                y2=y2,
                where=where,
                color=f"C{icat}",
                edgecolor="None",
                label=cat,
                transform=transform,
            )

        if info is not None:
            xlim = ax2.get_xlim()
            xc = (xlim[0] + xlim[1]) / 2
            yc = (y1 + y2) / 2

            ax2.text(
                xc,
                yc,
                info,
                fontsize=10,
                backgroundcolor="white",
                va="center",
                ha="center",
                transform=transform,
            )

        if legend:
            ax2.legend(fontsize="small", loc="lower left", title="Limieten")

        ax2.set_axis_off()
        return ax


w_all = dict()

for strang, c in config.iterrows():
    df_a_filter = pd.read_excel(filterweerstand_fp, sheet_name=strang)
    df_a_leiding = pd.read_excel(leidingweerstand_fp, sheet_name=strang)
    df_a_wvp = pd.read_excel(wvpweerstand_fp, sheet_name=strang, index_col=0).squeeze(
        "columns"
    )

    weerstand = strangWeerstand(df_a_leiding, df_a_filter, df_a_wvp, **c.to_dict())
    w_all[strang] = weerstand

lims = pd.DataFrame({k: v.capaciteit(index) for k, v in w_all.items()})
pd.DataFrame({k: v.capaciteit(index) for k, v in w_all.items()}).plot.area()

fig_path = os.path.join(res_folder, f"Capaciteit Som - {strang}.png")
plt.savefig(fig_path, dpi=300)
print("hoi")
