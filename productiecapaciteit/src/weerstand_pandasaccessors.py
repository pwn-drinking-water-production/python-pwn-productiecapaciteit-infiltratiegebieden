import numpy as np
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("wel")
class WellResistanceAccessor:
    """De eerste datum is het begin van je tijdreeks.
    Andere datums zijn die van de werkzaamheden"""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        if (
            "datum" not in obj.columns
            or "offset" not in obj.columns
            or "slope" not in obj.columns
        ):
            raise AttributeError("Must have 'datum' and 'offset' and 'slope'.")

        # all dates are sorted
        assert np.all(obj.datum.diff()[1:] > pd.Timedelta(days=1))

        # all slopes are negative
        assert np.all(obj.slope <= 0)

        # all offsets are negative
        assert np.all(obj.offset <= 0)
        pass

    @property
    def datum(self):
        return self._obj.datum.values

    @property
    def slope(self):
        return self._obj.slope.values

    @property
    def offset(self):
        return self._obj.offset.values

    @property
    def a_voor(self):
        # a_voor bestaat niet voor eerste datum
        dt = (self.datum[1:] - self.datum[:-1]) / pd.Timedelta(days=1)

        a_voor = np.concatenate(
            ([np.nan], dt * self.slope[:-1] + self.offset[:-1]), axis=0
        )
        return a_voor

    @property
    def a_na(self):
        return self.offset

    @property
    def a_effect(self):
        return self.a_na / self.a_voor

    def a_voor_projectie(self, datum_projectie):
        return self.a_na[-1] + self.slope[-1] * (
            pd.Timestamp(datum_projectie) - self.datum[-1]
        ) / pd.Timedelta(days=1)

    def a_na_projectie(self, datum_projectie, method="mean", reductie=None):
        if reductie is None:
            assert method == "mean"
            if self.a_effect.size > 1:
                reductie = np.nanmean(self.a_effect[1:])
            else:
                print("Geen schoonmaken die putweerstand doen afnemen")
                reductie = np.nan

        return self.a_voor_projectie(datum_projectie) * reductie

    def a_model(self, index):
        d_offset = pd.Series(index=index, data=0.0)
        d_slope = pd.Series(index=index, data=0.0)
        d_days_since_wzh = pd.Series(index=index, data=0.0)

        datums = self.datum.copy()

        if index[0] < datums[0]:
            datums[0] = index[0]

        if index[-1] > datums[-1]:
            datums = np.concatenate((datums, index[[-1]]))

        for offset, slope, datum, start, end in zip(
            self.offset, self.slope, self.datum, datums[:-1], datums[1:]
        ):
            d_offset[start:end] = offset
            d_slope[start:end] = slope
            d_days_since_wzh[start:end] = (
                d_days_since_wzh[start:end].index - datum
            ) / pd.Timedelta(days=1)

        # Add only an offset to times before first datum
        d_slope[: datums[0]] = 0.0

        return d_offset + d_slope * d_days_since_wzh

    def dp_voor(self, flow):
        # Flow per put
        return self.a_voor * flow

    def dp_na(self, flow):
        # Flow per put
        return self.a_na * flow

    def dp_projectie_voor(self, datum_projectie, flow):
        # Flow per put
        return self.a_voor_projectie(datum_projectie) * flow

    def dp_projectie_na(self, datum_projectie, flow, method="mean", reductie=None):
        # Flow per put
        return (
            self.a_na_projectie(datum_projectie, method=method, reductie=reductie)
            * flow
        )

    def dp_model(self, index, flow):
        # Flow per put
        # Verlaging: er komen positieve waarden uit
        return self.a_model(index) * flow

    def plot_werkzh(self, ax, dates_in_excel=None):
        if dates_in_excel is not None:
            ax.vlines(
                dates_in_excel[1:],
                ls="-",
                lw=5,
                ymin=0,
                ymax=1,
                linewidth=4,
                color="gold",
                transform=ax.get_xaxis_transform(),
                label="Werkzh volgens Excel",
            )

        ax.vlines(
            self.datum[1:],
            ls=":",
            lw=2,
            ymin=0,
            ymax=1,
            linewidth=2,
            color="C5",
            transform=ax.get_xaxis_transform(),
            label="Werkzh meegenomen in model",
        )
        self.plot_effect(ax)
        pass

    def plot_effect(self, ax, yc=0.02):
        transform = ax.get_xaxis_transform()

        for date, effect in zip(self.datum[1:], self.a_effect[1:]):
            string = f"{(1 - effect) * 100:.0f}% reductie"
            ax.text(
                date,
                yc,
                string,
                rotation=90,
                fontsize='small',
                va="bottom",
                ha="center",
                transform=transform,
                bbox=dict(alpha=0.5, facecolor="white", linewidth=0)
            )


@pd.api.extensions.register_series_accessor("wvp")
class WvpResistanceAccessor:
    """De eerste datum is het begin van je tijdreeks.
    Andere datums zijn die van de werkzaamheden"""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        req_keys = [
            "offset",
            "offset_datum",
            "slope",
            "temp_mean",
            "temp_delta",
            "temp_ref",
            "time_offset",
            "method",
        ]
        if not all([i in obj for i in req_keys]):
            raise AttributeError(f"Must have {' and '.join(req_keys)}.")

        # all slopes are negative
        assert np.all(obj.slope <= 0)

        # all offsets are negative
        assert np.all(obj.offset <= 0)
        assert obj.method in ("sin", "Niet"), "Method not supported"
        assert obj.temp_delta >= 0.
        assert (obj.temp_mean >= 0.) and (obj.temp_mean <= 30.)
        pass

    @property
    def offset(self):
        return self._obj.offset

    @property
    def offset_datum(self):
        return self._obj.offset_datum

    @property
    def slope(self):
        return self._obj.slope

    @property
    def temp_min(self):
        return self._obj.temp_mean - self._obj.temp_delta

    @property
    def temp_max(self):
        return self._obj.temp_mean + self._obj.temp_delta
    
    @property
    def temp_mean(self):
        return self._obj.temp_mean
    
    @property
    def temp_delta(self):
        return self._obj.temp_delta

    @property
    def temp_ref(self):
        return self._obj.temp_ref

    @property
    def time_offset(self):
        return self._obj.time_offset

    @property
    def method(self):
        return self._obj.method

    def temp_model(self, index):
        if self.method == "sin":
            # temp = delta * sin((t - offset) * 2 * pi / 365) + mean
            year = pd.Categorical(index.year, ordered=True)
            start_year = year.rename_categories(
                pd.to_datetime(year.categories, format="%Y")
            )
            end_year = year.rename_categories(
                pd.to_datetime(year.categories.astype(str) + "1231", format="%Y%m%d")
            )
            nday_year = end_year.map(lambda x: x.dayofyear)
            dt_year = index - start_year.to_numpy()
            temp_data = (
                    self.temp_delta
                    * np.sin(
                (dt_year / pd.Timedelta("1D") - self.time_offset)
                * 2
                * np.pi
                / nday_year
            )
                    + self.temp_mean
            )
            temp_df = pd.Series(data=temp_data, index=index, name="wvp_model_temp")
            return temp_df

        else:
            AssertionError("Method not supported")

        pass

    def model_viscratio(self, index):
        """Bij 20degC -> 0.8, bij 5degC -> 1.2"""
        if self.method == "Niet":
            return pd.Series(data=1, index=index, name="wvp_model_viscratio")

        else:
            temp_aquifer = self.temp_model(index)
            return self.visc_ratio(temp_aquifer, temp_ref=self.temp_ref).rename(
                "wvp_model_viscratio"
            )

    def viscratio(self, index, temp_wvp):
        """Bij 20degC -> 0.8, bij 5degC -> 1.2"""
        data = self.visc_ratio(temp_wvp, temp_ref=self.temp_ref)
        return pd.Series(data=data, index=index, name="wvp_viscratio")

    def a_model_reftemp(self, index):
        """Weerstand bij referentie temp"""
        dt = (index - self.offset_datum) / pd.Timedelta('1D')
        a = self.offset + self.slope * dt
        return pd.Series(data=a, index=index, name="wvp_model_a at 12degC")

    def a_model(self, index, temp_wvp=None):
        wvp_model_a12 = self.a_model_reftemp(index)

        if temp_wvp is None:
            # Use model for wvp temperature
            wvp_visc_ratio = self.model_viscratio(index)
        else:
            # Use measurements for wvp temperature
            wvp_visc_ratio = self.viscratio(index, temp_wvp)

        return wvp_model_a12 * wvp_visc_ratio

    def dp_model(self, index, flow, temp_wvp=None):
        a = self.a_model(index, temp_wvp=temp_wvp)
        return a * flow

    @staticmethod
    def visc_ratio(temp, temp_ref=12.0):
        visc_ref = (
            1 + 0.0155 * (temp_ref - 20.0)
        ) ** -1.572  # / 1000  removed the division because we re taking a ratio.
        visc = (1 + 0.0155 * (temp - 20.0)) ** -1.572  # / 1000
        return visc / visc_ref


@pd.api.extensions.register_dataframe_accessor("leiding")
class LeidingResistanceAccessor:
    """De eerste datum is het begin van je tijdreeks.
    Andere datums zijn die van de werkzaamheden"""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        if (
            "datum" not in obj.columns
            or "offset" not in obj.columns
            or "slope" not in obj.columns
        ):
            raise AttributeError("Must have 'datum' and 'offset' and 'slope'.")

        # all dates are sorted
        assert np.all(obj.datum.diff()[1:] > pd.Timedelta(days=1))

        # all slopes are negative
        assert np.all(obj.slope <= 0)

        # all offsets are negative
        assert np.all(obj.offset <= 0)
        pass

    @property
    def datum(self):
        return self._obj.datum.values

    @property
    def slope(self):
        return self._obj.slope.values

    @property
    def offset(self):
        return self._obj.offset.values

    @property
    def a_voor(self):
        # a_voor bestaat niet voor eerste datum
        dt = (self.datum[1:] - self.datum[:-1]) / pd.Timedelta(days=1)

        a_voor = np.concatenate(
            ([np.nan], dt * self.slope[:-1] + self.offset[:-1]), axis=0
        )
        return a_voor

    @property
    def a_na(self):
        return self.offset

    @property
    def a_effect(self):
        return self.a_na / self.a_voor

    def a_voor_projectie(self, datum_projectie):
        return self.a_na[-1] + self.slope[-1] * (
            pd.Timestamp(datum_projectie) - self.datum[-1]
        ) / pd.Timedelta(days=1)

    def a_na_projectie(self, datum_projectie, method="mean", reductie=None):
        if reductie is None:
            assert method == "mean"
            if self.a_effect.size > 1:
                reductie = np.nanmean(self.a_effect[1:])
            else:
                print("Geen schoonmaken die leidingweerstand doen afnemen")
                reductie = np.nan

        return self.a_voor_projectie(datum_projectie) * reductie

    def a_model(self, index):
        d_offset = pd.Series(index=index, data=0.0)
        d_slope = pd.Series(index=index, data=0.0)
        d_days_since_wzh = pd.Series(index=index, data=0.0)

        datums = self.datum.copy()

        if index[0] < datums[0]:
            datums[0] = index[0]

        if index[-1] > datums[-1]:
            datums = np.concatenate((datums, index[[-1]]))

        for offset, slope, datum, start, end in zip(
            self.offset, self.slope, self.datum, datums[:-1], datums[1:]
        ):
            d_offset[start:end] = offset
            d_slope[start:end] = slope
            d_days_since_wzh[start:end] = (
                d_days_since_wzh[start:end].index - datum
            ) / pd.Timedelta(days=1)

        # Add only an offset to times before first datum
        d_slope[: datums[0]] = 0.0

        return d_offset + d_slope * d_days_since_wzh

    def dp_voor(self, flow):
        return self.a_voor * flow**2

    def dp_na(self, flow):
        return self.a_na * flow**2

    def dp_projectie_voor(self, datum_projectie, flow):
        return self.a_voor_projectie(datum_projectie) * flow**2

    def dp_projectie_na(self, datum_projectie, flow, method="mean", reductie=None):
        return (
            self.a_na_projectie(datum_projectie, method=method, reductie=reductie)
            * flow**2
        )

    def dp_model(self, index, flow):
        # Verlaging: er komen positieve waarden uit
        return self.a_model(index) * flow**2

    def plot_werkzh(self, ax, dates_in_excel=None):
        if dates_in_excel is not None:
            ax.vlines(
                dates_in_excel[1:],
                ls="-",
                lw=5,
                ymin=0,
                ymax=1,
                linewidth=4,
                color="gold",
                transform=ax.get_xaxis_transform(),
                label="Werkzh volgens Excel",
            )

        ax.vlines(
            self.datum[1:],
            ls=":",
            lw=2,
            ymin=0,
            ymax=1,
            linewidth=2,
            color="C5",
            transform=ax.get_xaxis_transform(),
            label="Werkzh meegenomen in model",
        )
        self.plot_effect(ax)
        pass

    def plot_effect(self, ax, yc=0.02):
        transform = ax.get_xaxis_transform()

        for date, effect in zip(self.datum[1:], self.a_effect[1:]):
            string = f"{(1 - effect) * 100:.0f}% reductie"
            ax.text(
                date,
                yc,
                string,
                rotation=90,
                fontsize='small',
                va="bottom",
                ha="center",
                transform=transform,
                bbox=dict(alpha=0.5, facecolor="white", linewidth=0)
            )

