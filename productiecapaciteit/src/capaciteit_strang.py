from datetime import timedelta, datetime
from pprint import pformat
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from productiecapaciteit.src.weerstand_pandasaccessors import LeidingResistanceAccessor
from productiecapaciteit.src.weerstand_pandasaccessors import WellResistanceAccessor
from productiecapaciteit.src.weerstand_pandasaccessors import WvpResistanceAccessor
from productiecapaciteit.src.strang_analyse_fun2 import visc_ratio

gridspec_kw = {
    "left": 0.07,
    "bottom": 0.12,
    "right": 0.92,
    "top": 0.88,
    "wspace": 0.2,
    "hspace": 0.2,
}


# def get_config(fn):
#     config = pd.read_excel(fn).set_index("Unnamed: 0").T
#     config = config.loc[:, config.columns.notna()]
#     return config


class strangWeerstand(object):
    def __init__(
        self,
        df_a_leiding,
        df_a_filter,
        df_a_wvp,
        druk_limiet=-7.5,
        temp_opwarming=None,
        **kwargs,
    ):
        self.__dict__.update(kwargs)
        self._df_leiding = df_a_leiding.copy()
        self._df_filter = df_a_filter.copy()
        self._df_wvp = df_a_wvp.copy()
        self.druk_limiet = druk_limiet  # onderdruk vacuumsysteem
        self.temp_opwarming = temp_opwarming
        self.test_dates = {
            # 'test2009': datetime(2009, 8, 18),
            "test2015": datetime(2015, 6, 30),
            "test2021": datetime(2021, 6, 14),
        }

        if self.temp_opwarming is not None:
            # visc_ratio(15, temp_ref=8) = 0.82
            mean_temp = self._df_wvp.temp_mean
            mean_temp_new = mean_temp + temp_opwarming
            ratio = visc_ratio(mean_temp_new, temp_ref=mean_temp)

            # adjust filter
            self._df_filter.offset *= ratio
            self._df_filter.slope *= ratio

            # adjust wvp
            self._df_wvp.temp_mean += temp_opwarming

    def __repr__(self, width=100):
        s = list()

        sorted_dict = dict(sorted(self.__dict__.items()))
        for k in ["_df_leiding", "_df_filter", "_df_wvp"]:
            si = f"MODEL CONFIG: {k.split('_')[-1]}\n"
            si += pformat(self.__dict__[k], width=width)
            s.append(si)
            del sorted_dict[k]

        s.append("PARAMETERS:\n" + pformat(sorted_dict, width=width))

        object_methods = [method_name for method_name in dir(self)
                          if callable(getattr(self, method_name))
                          and method_name[0] != "_"]
        si = "METHODS:\n - " + "\n - ".join(object_methods)
        s.append(si)

        return ("\n" + width * "-" + "\n").join(s)

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
        index = pd.DatetimeIndex(index)
        return pd.Series(data=self.nput * self.Qlim_bio_per_put, index=index)

    def lims(self, index, use_lims=None):
        out = pd.DataFrame(
            {
                "Pompcapaciteit": self.Qpomp,
                "Vacuumsysteem": self.lim_vac(index),
                "Luchthappen": self.lim_luchthap(index),
                "Verblijftijd": self.lim_verblijf(index),
            },
            index=pd.DatetimeIndex(index)
        )
        use_lims = out.columns if use_lims is None else use_lims
        return out[use_lims]

    def lim_flow_min(self, index, deltah_veilig=1.5):
        a_wvp = self.wvp.wvp.a_model(index)
        flow_min = -(self.hpand - self.hmaaiveld + deltah_veilig) / a_wvp
        flow_min[flow_min < 0.0] = 0
        return flow_min

    def lim_flow_min_schoonmaak(self, index, date_clean, leiding=True, wel=True):
        strang2 = self.get_schoonmaakscenario(date_clean, leiding=leiding, wel=wel)
        return strang2.lim_flow_min(index, deltah_veilig=1.5)

    def capaciteit(self, index, use_lims=None):
        return self.lims(index, use_lims=use_lims).min(axis=1)

    def capaciteit_cat(self, index, use_lims=None):
        lims = self.lims(index, use_lims=use_lims)
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

    def lims_schoonmaak(self, date_clean, date_test, leiding=True, wel=True):
        strang2 = self.get_schoonmaakscenario(date_clean, leiding=leiding, wel=wel)
        return strang2.lims(date_test)

    def capaciteit_schoonmaak(self, date_clean, date_test, leiding=True, wel=True):
        return self.lims_schoonmaak(date_clean, date_test, leiding=leiding, wel=wel).min(axis=1)

    def capaciteit_effect_schoonmaak(
        self, date_clean, date_test, leiding=True, wel=True
    ):
        cap_na_schoonmaak = self.capaciteit_schoonmaak(
            date_clean, date_test, leiding=leiding, wel=wel
        )

        return cap_na_schoonmaak - self.capaciteit(date_test)

    def report_capaciteit_effect_schoonmaak(
        self, date_clean, date_goal
    ):
        """
        Investigate the effect a schoonmaak on date `date_clean` on the capacity on date `date_goal`.
        """
        effect_leiding = self.capaciteit_effect_schoonmaak(
            date_clean, date_goal, leiding=True, wel=False
        )
        effect_wel = self.capaciteit_effect_schoonmaak(
            date_clean, date_goal, leiding=False, wel=True
        )
        effect_som = self.capaciteit_effect_schoonmaak(
            date_clean, date_goal, leiding=True, wel=True
        )

        # grove schatting van contributie van schoonmaak leidingen/putten
        # Leidingweerstanden gaan kwadratisch en tellen dus niet lekker op
        # strang_leiding + strang_wel != strang_som
        ratio_lei = effect_leiding / (effect_leiding + effect_wel)
        ratio_wel = effect_wel / (effect_leiding + effect_wel)
        frac_lei = ratio_lei * effect_som
        frac_wel = ratio_wel * effect_som
        out = {
            "leiding": effect_leiding,
            "wel": effect_wel,
            "som": effect_som,
            "ratio_wel": ratio_wel,
            "ratio_lei": ratio_lei,
            "frac_wel": frac_wel,
            "frac_lei": frac_lei
        }
        return effect_som, out

    def capaciteit_effect_schoonmaak_cat(
        self, date_clean, date_test, leiding=True, wel=True
    ):
        strang2 = self.get_schoonmaakscenario(date_clean, leiding=leiding, wel=wel)
        return strang2.capaciteit_cat(date_test)

    def plot_lims(self, index, date_clean, ax=None, deltah_veilig=1.5):
        index_voor = index[index < date_clean]
        index_na = index[index >= date_clean]

        #### Capacity
        # Before cleaning
        # show = index < date_clean
        lims_voor = self.lims(index_voor)
        lims_na = self.lims(index_na)
        for ilim, lim in enumerate(lims_voor):
            ax.plot(index_voor, lims_voor[lim], lw=0.8, label=lim, c=f"C{ilim}")
            ax.plot(index_na, lims_na[lim], lw=0.8, c=f"C{ilim}", alpha=0.3)

        cap = self.capaciteit(index_voor)
        nlims = lims_voor.columns.size
        ax.plot(index_voor, cap, lw=2, label="Max. inzet zonder schoonmaak", c=f"C{nlims}", ls=":")

        # Take into account the cleaning
        lims_na = self.lims_schoonmaak(date_clean, index_na, leiding=True, wel=True)
        for ilim, lim in enumerate(lims_na):
            ax.plot(index_na, lims_na[lim], lw=0.8, c=f"C{ilim}")

        cap = self.capaciteit_schoonmaak(
            date_clean, index_na, leiding=True, wel=True
        )
        ax.plot(index_na, cap, lw=2, label="Max. inzet na schoonmaak", c=f"C{nlims + 1}", ls=":")

        #### Minimal flow
        # Before cleaning
        flow_min_voor = self.lim_flow_min(index_voor, deltah_veilig=deltah_veilig)
        flow_min_na = self.lim_flow_min(index_na, deltah_veilig=deltah_veilig)
        ax.plot(
            index_voor, flow_min_voor, label="Min. inzet voor schoonmaak", lw=1, c=f"C{nlims + 2}", ls="--"
        )
        ax.plot(
            index_na, flow_min_na, lw=2, c=f"C{nlims + 2}", ls="--", alpha=0.3
        )

        # After cleaning
        flow_min_na = self.lim_flow_min_schoonmaak(index_na, date_clean, leiding=True, wel=True)
        ax.plot(
            index_na, flow_min_na, label="Min. inzet na schoonmaak", lw=1, c=f"C{nlims + 3}", ls="--"
        )

        dates = self.test_dates.values()
        flows = [self.__dict__[key] for key in self.test_dates.keys()]
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
            info="Voor schoonmaak",
            legend=True
        )
        self.plot_lim_cat(
            date_clean,
            index_na,
            ax=ax,
            y1=0.0,
            y2=0.045,
            leiding=True,
            wel=True,
            info="Na schoonmaak",
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
        self, date_clean, date_test, date_goal, ax=None, y0=0, thick=0.08, legend=True,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5), gridspec_kw=gridspec_kw)

        effect_som, effect_dict = self.report_capaciteit_effect_schoonmaak(date_clean, date_test)

        ax.fill_between(date_test, y1=effect_dict["frac_lei"], y2=y0, label="Leiding")
        ax.fill_between(date_test, y1=effect_som, y2=effect_dict["frac_lei"], label="Put")
        ax.plot(date_test, effect_som, c="black", lw=0.8, label="")

        self.plot_schoonmaak(ax, [date_clean], label="Schoonmaak moment")
        self.plot_schoonmaak(ax, self.get_schoonmaak_dates(), label="")

        string = date_goal.strftime(f"%Y-%m-%d: {effect_som[date_goal]:.0f} m$^3$/h extra")
        transform = ax.get_xaxis_transform()
        ax.text(
            date_goal,
            0.5,
            string,
            rotation=90,
            fontsize="small",
            va="center",
            ha="left",
            transform=transform,
            bbox=dict(alpha=0.5, facecolor="white", linewidth=0),
        )
        ax.vlines(
            date_goal,
            ls=":",
            lw=2,
            ymin=0,
            ymax=1,
            linewidth=1,
            color="C5",
            transform=ax.get_xaxis_transform(),
        )

        if legend:
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
            info="Na schoonmaak",
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
                # backgroundcolor="white",
                va="center",
                ha="center",
                transform=transform,
            )

        if legend:
            ax2.legend(fontsize="small", loc="lower left", title="Limieten", ncol=2)

        ax2.set_axis_off()
        return ax
