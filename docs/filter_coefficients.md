# Filter coefficients: units, temperature and calibration

The existing workbook stores signed hourly coefficients:

```text
gws0 - gws1 = a_hour(t) * q_sensor_m3h
a_hour(t) = offset_j + slope * days_since_datum_j
```

`gws0` is internal head and `gws1` outside-filter head, in the same metre datum.
Offsets have units h/m²; slopes have units h/m²/day. The normalized fitter estimates
one shared nonpositive slope and a nonpositive offset for each cleaning period.
It fits the native observations, not daily averages or smoothed ratios.

## Solver interface

```python
from productiecapaciteit.src.filter_calibration import (
    filter_resistance,
    split_filter_intervals,
)

split_edges = split_filter_intervals(forcing_edges, coefficients)
rf = filter_resistance(
    coefficients,
    split_edges[1:],
    temperature=extracted_water_temperature_at_endpoints,
    side="left",
)
```

`rf` is nonnegative resistance in day/m² for solver flow in m³/day:
`Rf_day = -a_hour/24`. The equivalent accessor methods are
`coefficients.wel.resistance_model(...)` and
`coefficients.wel.split_intervals(forcing_edges)`. Importing
`filter_calibration` registers these accessors.

Forcing edges include the initial instant. After splitting an interval, the caller
must carry its original piecewise-constant flow onto both subintervals; splitting
does not create new measured values. `side="left"` uses the pre-cleaning coefficient
at the endpoint of the interval ending at cleaning. `side="right"` uses the new
offset at that instant. Splitting alone is insufficient without this one-sided
evaluation. Aquifer state is continuous through cleaning; the algebraic filter
loss can jump.

`evaluate_filter_coefficient` returns the signed stored coefficient without
viscosity. Before the first datum it holds the first offset constant. Existing
`wel.a_model` and `wel.dp_model` retain their historical behavior and do **not**
automatically apply normalized temperature metadata. Use `resistance_model` for
temperature-aware hydraulics.

## Persisted temperature semantics

Metadata is repeated in ordinary dataframe/Excel columns, not `DataFrame.attrs`.

| Column | Meaning |
| --- | --- |
| `temperature_semantics` | `legacy_empirical` or `reference_normalized` |
| `temperature_ref_degC` | Required common reference for normalized rows; undefined for legacy rows |
| `temperature_channel` | Explicit extracted-water channel used in the normalized fit |
| `temperature_policy` | `native_no_interpolation` for normalized fits |
| `coefficient_units`, `slope_units` | `h/m2`, `h/m2/day` |
| `sensor_id` | Head-sensor well identity |
| `head_provenance` | `direct` or `reconstructed` |
| `inside_head_channel`, `outside_head_channel` | Channels used for the head difference |
| `flow_denominator_provenance` | `measured_individual:<channel>` or `effective_equal_flow:<channel>/nput=<n>` |
| `fit_periods` | JSON array of inclusive ISO timestamp pairs |
| `upstream_fit_dependencies` | JSON list of all upstream fit identities and periods |
| `fit_method` | Objective/algorithm used, distinguishing the historical empirical fit |
| `observation_support`, `n_observations` | Native observation support and usable fitted count |

A workbook without `temperature_semantics` remains a legacy empirical workbook.
Its evaluation emits `LegacyFilterCoefficientWarning` and **does not** apply a
second viscosity correction, even when temperature is supplied. It must not be
advertised as reference-normalized. Mixed/unknown semantics and incomplete
normalized reference metadata are errors.

Normalized evaluation computes
`Rf_day(t) = -a_ref_hour(t)/24 * visc_ratio(T_extracted(t), T_ref)`, using the
existing `WvpResistanceAccessor.visc_ratio` helper. Cooler water increases
resistance. The reference temperature is persisted, not assumed from old files.
Temperature must be finite and in [0, 40] °C. Series are timestamp-aligned; scalar
temperature explicitly prescribes a constant. Missing, invalid or misaligned
temperature raises `ExtractedWaterTemperatureError`. There is no implicit
interpolation, extrapolation or substitution of canal-bed temperature. Any
upstream interpolation policy must be declared separately by its caller.

## Opt-in native-resolution fitting

```python
from productiecapaciteit.src.filter_calibration import fit_normalized_filter

coefficients = fit_normalized_filter(
    native_observations,
    coefficient_dates,
    temperature_column="T_extracted",  # audited extracted-water channel
    sensor_id="well-17",
    head_provenance="direct",
    upstream_fit_dependencies=[],  # explicitly no upstream fits
    sensor_flow_column="q_sensor",
    reference_temperature=12.0,
)
```

The fitter uses the signed observations
`a_ref = (gws0-gws1)/(q_sensor*visc_ratio)`. Negative flow is supported. Zero flow,
nonfinite heads and nonfinite flow are excluded before temperature validation.
It refits **both** the shared slope and cleaning offsets in reference-temperature
space by bounded linear least squares on raw residuals. It never uses the
temperature-uncorrected configured `put_a_slope`. A rank-deficient selection
raises an identifiability error rather than inventing a slope.

If individual measured flow is unavailable, explicitly pass
`sensor_flow_column=None`, `allow_effective_equal_flow=True` and a positive integer
`nput`. The fit then uses `Q/nput` and records an **effective** coefficient.
A declared but missing individual-flow channel is an error, not permission to
fall back. In particular:

```text
R_effective / R_true = nput * q_sensor / Q
```

This unequal-flow bias survives viscosity normalization. Such calibration needs
held-out head comparison and sensitivity assessment before transfer to a model
with solved unequal flows.

Pass `fit_periods=[(start, end), ...]` to restrict and record fitting windows.
Declare every upstream dependency with its own fitting periods, for example
`{"parameter_fit": "filter-v1", "fit_periods": [["2019-01-01", "2019-12-31"]]}`.
Reconstructed outside-filter heads are derived diagnostics, never independent
validation evidence. A record is held out only if it lies outside **all** relevant
upstream filter/aquifer fits, not just this fit's periods. Unknown legacy
dependencies are serialized as JSON `null`, not falsely asserted to be absent.
The metadata records provenance; it does not itself establish independent
scientific acceptance.

## Report

`python -m productiecapaciteit.reports.putfilter3` keeps the legacy empirical
workflow and configured slope. Importing the module performs no report I/O.

The opt-in route requires one selected strang and explicitly audited channels:

```powershell
python -m productiecapaciteit.reports.putfilter3 `
  --normalize-temperature --strang YOUR_STRANG `
  --temperature-column T_extracted --sensor-id well-17 `
  --sensor-flow-column q_sensor --head-provenance direct `
  --upstream-fit-dependencies-json '[]' --reference-temperature 12
```

For the explicit effective fallback, omit `--sensor-flow-column` and add
`--allow-effective-equal-flow`; the configured active count is recorded.
Normalized coefficients are saved to
`Filterweerstand_modelcoefficienten_reference.xlsx` and normalized plots have a
`_reference` suffix, so legacy readers cannot silently acquire new temperature
semantics. The plots compare native normalized observations (smoothed for display
only) with the reference-space model.
