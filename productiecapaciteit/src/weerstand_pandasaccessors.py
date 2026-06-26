import warnings

import numpy as np
import pandas as pd

from productiecapaciteit.src.wvp_transient_funs import (
    build_multiwell_geometry,
    infer_lower_timestep,
    objective,
)


@pd.api.extensions.register_dataframe_accessor("wel")
class WellResistanceAccessor:
    """De eerste datum is het begin van je tijdreeks.
    Andere datums zijn die van de werkzaamheden
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        if "datum" not in obj.columns or "offset" not in obj.columns or "slope" not in obj.columns:
            raise AttributeError("Must have 'datum' and 'offset' and 'slope'.")

        # all dates are sorted
        assert np.all(obj.datum.diff()[1:] > pd.Timedelta(days=1))

        # all slopes are negative
        assert np.all(obj.slope <= 0)

        # all offsets are negative
        assert np.all(obj.offset <= 0)

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

        a_voor = np.concatenate(([np.nan], dt * self.slope[:-1] + self.offset[:-1]), axis=0)
        return a_voor

    @property
    def a_na(self):
        return self.offset

    @property
    def a_effect(self):
        return self.a_na / self.a_voor

    def a_voor_projectie(self, datum_projectie):
        return self.a_na[-1] + self.slope[-1] * (pd.Timestamp(datum_projectie) - self.datum[-1]) / pd.Timedelta(days=1)

    def a_na_projectie(self, datum_projectie, method="mean", reductie=None):
        if reductie is None:
            assert method == "mean"
            if self.a_effect.size > 1:
                reductie = np.nanmean(self.a_effect[1:])
            else:
                print("Geen schoonmaken die putweerstand doen afnemen => geen voorspelde reductie")
                reductie = 1.0

        return self.a_voor_projectie(datum_projectie) * reductie

    def a_model(self, index):
        index = pd.DatetimeIndex(index)

        d_offset = pd.Series(index=index, data=0.0)
        d_slope = pd.Series(index=index, data=0.0)
        d_days_since_wzh = pd.Series(index=index, data=0.0)

        datums = self.datum.copy()

        datums[0] = min(index[0], datums[0])

        if index[-1] > datums[-1]:
            datums = np.concatenate((datums, index[[-1]]))

        for offset, slope, datum, start, end in zip(
            self.offset, self.slope, self.datum, datums[:-1], datums[1:], strict=False
        ):
            d_offset[start:end] = offset
            d_slope[start:end] = slope
            d_days_since_wzh[start:end] = (d_days_since_wzh[start:end].index - datum) / pd.Timedelta(days=1)

        # Add only an offset to times before first datum
        d_slope[: datums[0]] = 0.0

        return d_offset + d_slope * d_days_since_wzh

    def add_zero_effect_dates(self, dates):
        # add dates with zero effect. inplace not possible
        dates = pd.Index(dates)
        dates_new = list(filter(lambda x: x not in self.datum, dates))
        nnew = len(dates_new)

        if nnew == 0:
            return self._obj

        offsets_new = self.a_model(dates_new).values
        slopes_new = np.tile(self.slope[[0]], nnew)
        gewijzigd_new = np.tile([pd.Timestamp.now()], nnew)

        df_new = pd.DataFrame({
            "datum": dates_new,
            "offset": offsets_new,
            "slope": slopes_new,
            "gewijzigd": gewijzigd_new,
        })

        values_new = np.insert(self._obj.values, 0, values=df_new.values, axis=0)
        pandas_obj = pd.DataFrame(
            data=values_new,
            columns=self._obj.columns,
        ).sort_values("datum", ignore_index=True)

        self._validate(pandas_obj)
        return pandas_obj

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
        return self.a_na_projectie(datum_projectie, method=method, reductie=reductie) * flow

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

    def plot_effect(self, ax, yc=0.02):
        transform = ax.get_xaxis_transform()

        for date, effect in zip(self.datum[1:], self.a_effect[1:], strict=False):
            string = f"{(1 - effect) * 100:.0f}% reductie"
            ax.text(
                date,
                yc,
                string,
                rotation=90,
                fontsize="small",
                va="bottom",
                ha="center",
                transform=transform,
                bbox=dict(alpha=0.5, facecolor="white", linewidth=0),
            )


@pd.api.extensions.register_series_accessor("wvp")
class WvpResistanceAccessor:
    """De eerste datum is het begin van je tijdreeks.
    Andere datums zijn die van de werkzaamheden
    """

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

        if not np.all(obj.slope <= 0):
            raise ValueError("WVP slopes must be nonpositive")
        if not np.all(obj.offset <= 0):
            raise ValueError("WVP offsets must be nonpositive")
        if obj.method not in ("sin", "Niet"):
            raise ValueError("Method not supported")
        if obj.temp_delta < 0.0:
            raise ValueError("temp_delta must be nonnegative")
        if not 0.0 <= obj.temp_mean <= 30.0:
            raise ValueError("temp_mean must be between 0 and 30 degC")

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
        index = pd.DatetimeIndex(index)

        if self.method == "sin":
            # temp = delta * sin((t - offset) * 2 * pi / 365) + mean
            year = pd.Categorical(index.year, ordered=True)
            start_year = year.rename_categories(pd.to_datetime(year.categories, format="%Y"))
            end_year = year.rename_categories(pd.to_datetime(year.categories.astype(str) + "1231", format="%Y%m%d"))
            nday_year = end_year.map(lambda x: x.dayofyear, na_action="ignore").astype(float)
            dt_year = index - start_year.to_numpy()
            temp_data = (
                self.temp_delta * np.sin((dt_year / pd.Timedelta("1D") - self.time_offset) * 2 * np.pi / nday_year)
                + self.temp_mean
            )
            temp_df = pd.Series(data=temp_data, index=index, name="wvp_model_temp")
            return temp_df

        raise ValueError("Method not supported")

    def model_viscratio(self, index):
        """Bij 20degC -> 0.8, bij 5degC -> 1.2"""
        if self.method == "Niet":
            return pd.Series(data=1, index=index, name="wvp_model_viscratio")

        temp_aquifer = self.temp_model(index)
        return self.visc_ratio(temp_aquifer, temp_ref=self.temp_ref).rename("wvp_model_viscratio")

    def viscratio(self, index, temp_wvp):
        """Bij 20degC -> 0.8, bij 5degC -> 1.2"""
        data = self.visc_ratio(temp_wvp, temp_ref=self.temp_ref)
        return pd.Series(data=data, index=index, name="wvp_viscratio")

    def a_model_reftemp(self, index):
        """Weerstand bij referentie temp"""
        index = pd.DatetimeIndex(index)
        dt = (index - self.offset_datum) / pd.Timedelta("1D")
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


@pd.api.extensions.register_series_accessor("wvpt")
class WvpTransientResistanceAccessor(WvpResistanceAccessor):
    """Transient WVP resistance accessor using physical Hantush parameters."""

    REFERENCE_TRANSMISSIVITY_KEYS = (
        "kD_ref_m2_per_d",
        "kD_ref_slope_m2_per_d_per_d",
        "kD_ref_datum",
    )
    TEMPERATURE_KEYS = (
        "temperature_mean_degC",
        "temperature_delta_degC",
        "temperature_ref_degC",
        "temperature_time_offset_d",
        "temperature_method",
    )
    PHYSICAL_KEYS = (
        "well_radius_m",
        "storage_coefficient",
        "leakage_resistance_d",
    )

    @staticmethod
    def _validate(obj):
        req_keys = (
            WvpTransientResistanceAccessor.REFERENCE_TRANSMISSIVITY_KEYS
            + WvpTransientResistanceAccessor.TEMPERATURE_KEYS
            + WvpTransientResistanceAccessor.PHYSICAL_KEYS
        )
        if not all(key in obj for key in req_keys):
            raise AttributeError(f"Must have {' and '.join(req_keys)}.")

        values = np.array(
            [
                obj["kD_ref_m2_per_d"],
                obj["kD_ref_slope_m2_per_d_per_d"],
                obj["well_radius_m"],
                obj["storage_coefficient"],
                obj["leakage_resistance_d"],
            ],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError("Transient WVP coefficients must be finite")
        temperature_values = np.array(
            [
                obj["temperature_mean_degC"],
                obj["temperature_delta_degC"],
                obj["temperature_ref_degC"],
                obj["temperature_time_offset_d"],
            ],
            dtype=float,
        )
        if not np.isfinite(temperature_values).all():
            raise ValueError("Transient WVP temperature coefficients must be finite")
        if float(obj["kD_ref_m2_per_d"]) <= 0.0:
            raise ValueError("kD_ref_m2_per_d must be positive")
        if not np.all(values[2:] > 0.0):
            raise ValueError("Transient WVP physical coefficients must be positive")
        if obj["temperature_method"] not in ("sin", "Niet"):
            raise ValueError("WVPT temperature_method not supported")
        if obj["temperature_delta_degC"] < 0.0:
            raise ValueError("temperature_delta_degC must be nonnegative")
        if not 0.0 <= obj["temperature_mean_degC"] <= 30.0:
            raise ValueError("temperature_mean_degC must be between 0 and 30 degC")
        if float(obj["storage_coefficient"]) > 1.0:
            warnings.warn(
                "storage_coefficient is greater than 1; check the transient WVP coefficients",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def well_radius_m(self):
        return float(self._obj["well_radius_m"])

    @property
    def storage_coefficient(self):
        return float(self._obj["storage_coefficient"])

    @property
    def leakage_resistance_d(self):
        return float(self._obj["leakage_resistance_d"])

    @property
    def kD_ref_m2_per_d(self):
        return float(self._obj["kD_ref_m2_per_d"])

    @property
    def kD_ref_slope_m2_per_d_per_d(self):
        return float(self._obj["kD_ref_slope_m2_per_d_per_d"])

    @property
    def kD_ref_datum(self):
        return pd.Timestamp(self._obj["kD_ref_datum"])

    @property
    def temp_mean(self):
        return self._obj["temperature_mean_degC"]

    @property
    def temp_min(self):
        return self.temp_mean - self.temp_delta

    @property
    def temp_max(self):
        return self.temp_mean + self.temp_delta

    @property
    def temp_delta(self):
        return self._obj["temperature_delta_degC"]

    @property
    def temp_ref(self):
        return self._obj["temperature_ref_degC"]

    @property
    def time_offset(self):
        return self._obj["temperature_time_offset_d"]

    @property
    def method(self):
        return self._obj["temperature_method"]

    @property
    def alpha(self):
        return (self.well_radius_m**2 * self.storage_coefficient / 4.0) ** 0.5

    @property
    def beta(self):
        return (1.0 / (self.leakage_resistance_d * self.storage_coefficient)) ** 0.5

    @staticmethod
    def _as_1d_float_array(name, values, index):
        index = pd.DatetimeIndex(index)
        if isinstance(values, pd.Series):
            arr = values.reindex(index).to_numpy(dtype=float)
        else:
            arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = np.full(index.size, float(arr), dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a scalar or 1D array, got shape {arr.shape}")
        if arr.size != index.size:
            raise ValueError(f"{name} must have length {index.size}, got {arr.size}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return arr

    def _multiwell_geometry(
        self,
        nput,
        dx_tussenputten,
        r_mirrorwel,
        target_well_index=None,
    ):
        return build_multiwell_geometry(
            dx_tussenputten,
            r_mirrorwel,
            nput,
            target_well_index=target_well_index,
            distance_scale=1.0 / self.well_radius_m,
            include_self=True,
            self_distance=1.0,
        )

    def temp_model(self, index):
        return super().temp_model(index).rename("wvpt_model_temp")

    def model_viscratio(self, index):
        return super().model_viscratio(index).rename("wvpt_model_viscratio")

    def viscratio(self, index, temp_wvp):
        return super().viscratio(index, temp_wvp).rename("wvpt_viscratio")

    def kD_ref_model(self, index):
        index = pd.DatetimeIndex(index)
        dt = (index - self.kD_ref_datum) / pd.Timedelta("1D")
        kd_ref = self.kD_ref_m2_per_d + self.kD_ref_slope_m2_per_d_per_d * dt
        return pd.Series(data=kd_ref, index=index, name="wvpt_model_kD_ref")

    def kD_model(
        self,
        index,
        temp_wvp=None,
    ):
        """Return kD corrected from reference temperature to modeled or measured temperature."""
        index = pd.DatetimeIndex(index)
        kd_reference_temp = self.kD_ref_model(index).to_numpy(dtype=float)

        if temp_wvp is None:
            viscosity_ratio = self.model_viscratio(index).to_numpy(dtype=float)
        else:
            viscosity_ratio = self.viscratio(index, temp_wvp).to_numpy(dtype=float)
        if not np.isfinite(viscosity_ratio).all() or np.any(viscosity_ratio <= 0.0):
            raise ValueError("WVP viscosity ratio must be finite and positive")

        kd = kd_reference_temp / viscosity_ratio
        if not np.isfinite(kd).all() or np.any(kd <= 0.0):
            raise ValueError("Calibrated kD must be finite and positive")
        return pd.Series(data=kd, index=index, name="wvpt_model_kD")

    # The steady instantaneous-resistance helpers inherited from WvpResistanceAccessor
    # are meaningless for a transient model: resistance depends on the full flow history,
    # not a single timestamp. Reject them so callers reach for dp_model instead.
    @staticmethod
    def _transient_not_implemented(name):
        msg = f"WVPT is transient; {name} has no instantaneous form. Use dp_model(...) for transient drawdown."
        raise NotImplementedError(msg)

    def a_model(self, *args, **kwargs):
        self._transient_not_implemented("a_model")

    def a_model_reftemp(self, *args, **kwargs):
        self._transient_not_implemented("a_model_reftemp")

    def resistance_model(self, *args, **kwargs):
        self._transient_not_implemented("resistance_model")

    def resistance_model_reftemp(self, *args, **kwargs):
        self._transient_not_implemented("resistance_model_reftemp")

    def dp_model(
        self,
        index,
        flow,
        nput,
        dx_tussenputten,
        r_mirrorwel,
        temp_wvp=None,
        target_well_index=None,
        initial_condition="steady",
        frac_step_max=0.95,
        tmax_days_cap=None,
        max_workers=None,
        integration_method="kd_grid",
        n_gauss=32,
        max_gauss_step_days=0.5,
        n_per_step=8,
        near_steps=3,
        hantush_method="variable_kd",
    ):
        index = pd.DatetimeIndex(index)
        flow = self._as_1d_float_array("flow", flow, index)
        multiwell, multiwell_counts = self._multiwell_geometry(
            nput,
            dx_tussenputten,
            r_mirrorwel,
            target_well_index=target_well_index,
        )
        nput = float(nput)
        if nput <= 0.0:
            raise ValueError(f"nput must be positive, got {nput}")

        pextra = {
            "index": index,
            "Q_obs": flow / nput * 24.0,
            "kD": self.kD_model(index, temp_wvp=temp_wvp).to_numpy(dtype=float),
            "dt_lower": infer_lower_timestep(index),
            "multiwell_contains_r_self": True,
            "multiwell": multiwell,
            "multiwell_counts": multiwell_counts,
            "frac_step_max": frac_step_max,
            "initial_condition": initial_condition,
            "tmax_days_cap": tmax_days_cap,
            "max_workers": max_workers,
            "integration_method": integration_method,
            "n_gauss": n_gauss,
            "max_gauss_step_days": max_gauss_step_days,
            "n_per_step": n_per_step,
            "near_steps": near_steps,
            "hantush_method": hantush_method,
        }
        drawdown = objective([self.alpha, self.beta], return_result=True, **pextra)
        if not np.isfinite(drawdown).all():
            raise ValueError("Transient WVP drawdown contains NaN or infinite values")
        return pd.Series(data=-drawdown, index=index, name="wvpt_model_dp")


@pd.api.extensions.register_dataframe_accessor("leiding")
class LeidingResistanceAccessor:
    """De eerste datum is het begin van je tijdreeks.
    Andere datums zijn die van de werkzaamheden
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        if "datum" not in obj.columns or "offset" not in obj.columns or "slope" not in obj.columns:
            raise AttributeError("Must have 'datum' and 'offset' and 'slope'.")

        # all dates are sorted
        assert np.all(obj.datum.diff()[1:] > pd.Timedelta(days=0))

        # all slopes are negative
        assert np.all(obj.slope <= 0)

        # all offsets are negative
        assert np.all(obj.offset <= 0)

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

        a_voor = np.concatenate(([np.nan], dt * self.slope[:-1] + self.offset[:-1]), axis=0)
        return a_voor

    @property
    def a_na(self):
        return self.offset

    @property
    def a_effect(self):
        return self.a_na / self.a_voor

    def a_voor_projectie(self, datum_projectie):
        return self.a_na[-1] + self.slope[-1] * (pd.Timestamp(datum_projectie) - self.datum[-1]) / pd.Timedelta(days=1)

    def a_na_projectie(self, datum_projectie, method="mean", reductie=None):
        if reductie is None:
            assert method == "mean"
            if self.a_effect.size > 1:
                reductie = np.nanmean(self.a_effect[1:])
            else:
                print("Geen schoonmaken die leidingweerstand doen afnemen => geen voorspelde reductie")
                reductie = 1.0

        return self.a_voor_projectie(datum_projectie) * reductie

    def a_model(self, index):
        index = pd.DatetimeIndex(index)
        d_offset = pd.Series(index=index, data=0.0)
        d_slope = pd.Series(index=index, data=0.0)
        d_days_since_wzh = pd.Series(index=index, data=0.0)

        datums = self.datum.copy()

        datums[0] = min(index[0], datums[0])

        if index[-1] > datums[-1]:
            datums = np.concatenate((datums, index[[-1]]))

        for offset, slope, datum, start, end in zip(
            self.offset, self.slope, self.datum, datums[:-1], datums[1:], strict=False
        ):
            d_offset[start:end] = offset
            d_slope[start:end] = slope
            d_days_since_wzh[start:end] = (d_days_since_wzh[start:end].index - datum) / pd.Timedelta(days=1)

        # Add only an offset to times before first datum
        d_slope[: datums[0]] = 0.0

        return d_offset + d_slope * d_days_since_wzh

    def add_zero_effect_dates(self, dates):
        # add dates with zero effect. inplace not possible
        dates = pd.Index(dates)
        dates_new = list(filter(lambda x: x not in self.datum, dates))
        nnew = len(dates_new)

        if nnew == 0:
            return self._obj

        offsets_new = self.a_model(dates_new).values
        slopes_new = np.tile(self.slope[[0]], nnew)
        gewijzigd_new = np.tile([pd.Timestamp.now()], nnew)

        df_new = pd.DataFrame({
            "datum": dates_new,
            "offset": offsets_new,
            "slope": slopes_new,
            "gewijzigd": gewijzigd_new,
        })

        values_new = np.insert(self._obj.values, 0, values=df_new.values, axis=0)
        pandas_obj = pd.DataFrame(
            data=values_new,
            columns=self._obj.columns,
        ).sort_values("datum", ignore_index=True)

        self._validate(pandas_obj)
        return pandas_obj

    def dp_voor(self, flow):
        return self.a_voor * flow**2

    def dp_na(self, flow):
        return self.a_na * flow**2

    def dp_projectie_voor(self, datum_projectie, flow):
        return self.a_voor_projectie(datum_projectie) * flow**2

    def dp_projectie_na(self, datum_projectie, flow, method="mean", reductie=None):
        return self.a_na_projectie(datum_projectie, method=method, reductie=reductie) * flow**2

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

    def plot_effect(self, ax, yc=0.02):
        transform = ax.get_xaxis_transform()

        for date, effect in zip(self.datum[1:], self.a_effect[1:], strict=False):
            string = f"{(1 - effect) * 100:.0f}% reductie"
            ax.text(
                date,
                yc,
                string,
                rotation=90,
                fontsize="small",
                va="bottom",
                ha="center",
                transform=transform,
                bbox=dict(alpha=0.5, facecolor="white", linewidth=0),
            )
