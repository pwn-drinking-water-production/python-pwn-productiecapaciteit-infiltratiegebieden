# Causal flow-driven common-head wells

`productiecapaciteit.src.wvp_robin` solves an open, connected well manifold:

```text
h_a,i - H_w = Rf q_i
sum(q_i) = Q
```

Rates are **m³/day**, time is **days**, heads are **metres in a common datum**,
filter resistance is **day/m²**, and canal-bed resistance is **day/m**.
The signed hourly filter coefficient converts as `Rf = -a_hour / 24`.
`Rf` is shared by all wells, not fitted separately to their discharges.
Negative well flows are physical crossflow in this model, including at zero
total abstraction. There is no clipping, rescaling, equal-flow fallback or
non-return-valve approximation.

## Interfaces

- `solve_common_head(A, h_free, Q, Rf)` is the pure no-shore algebra.
- `IntervalResponse(index, kd, radii, storage, leakage)` in
  `wvp_transient_funs` supplies causal interval Green responses. `index` contains
  **N+1 edges for N interval rates**, including the initial instant.
- `boundary_geometry` reuses the report's Legendre nodes, shoreline offset,
  finite-radius distance clipping and cubic interpolation in log radius.
  Its geometry-only operators contain no flow, resistance or time history.
- `solve_transient` commits independent well and shore rates chronologically.
  Current shoreline unknowns are solved simultaneously with current well rates
  and common head. Well and total-flow equations are hard constraints.
  Shore collocation is arc-length-weighted least squares in the null space of
  the hard equations, with SVD rank checks and explicit head/flow scaling.
- `RobinSolution` stores compact source/boundary histories, ranks, conditions
  and physical residuals. `drawdown` evaluates arbitrary ordered times in days
  and chunks map points; map cells never enter the boundary solve.
- `stop_after` and `resume` support checkpoints on the **same prescribed
  coefficient grid**, with unchanged past forcing, geometry and initial field.
  Keeping that grid fixed avoids changing the PCHIP kD interpolation at restart.

The existing offline time-shape shoreline fit remains a reference calculation;
it is **not** used as a causal response matrix.

## Initialization, cleaning and numerical error

`initial="zero"` means zero pumping/drawdown prehistory. A steady start requires
an explicit pre-period total flow. It solves the steady well/shore boundary
problem at initial coefficients, then carries that solved field through later
kD changes. It never replaces it with a new steady field at every endpoint.
Optional initial filter, bed and background values are explicit inputs.

Callers must split intervals at cleaning dates. For right-endpoint collocation,
use the filter coefficient's **left limit**, so cleaning is not applied
retrospectively. The aquifer field is continuous at the cleaning instant;
algebraic internal heads and partitions can change. Between endpoints the
piecewise-constant rates need not satisfy a common head exactly; temporal
refinement is a separate acceptance check.

`IntervalResponse.interval_weights` is the adaptive-quadrature reference.
The default append-only history tree retains every interval through integrated
source moments. Distant blocks interpolate the cumulative-kD Green kernel at
16 Gauss nodes, using an embedded 8-node estimate; near/unresolved blocks
descend to direct interval quadrature. No tail is silently truncated.
The embedded estimate is **not a rigorous global error bound**, particularly
under cancellation. Set `history_rtol=history_atol=0` for the direct reference.
Compare heads, rates and residuals under tolerance and discretization refinement.

For a spatially overdetermined shore, a steady least-squares field is only
stationary to its spatial residual. Do not demand machine-precision stationarity
before shore convergence. Exact hard constraints and approximate boundary/model
agreement are different quantities.

## Filter fitting and interpretation

See [filter coefficients](filter_coefficients.md) for native-resolution,
reference-temperature fitting and legacy workbook compatibility. A normalized
fit estimates offsets **and slope** in reference-temperature space. Legacy
empirical coefficients do not receive another viscosity correction.
An effective `Q/n` fit is not evidence that wells actually had equal flows:
`R_effective/R_true = n*q_sensor/Q`, even after viscosity normalization.

## Validation and limitations

Run in a separate environment, never the user's environment:

```powershell
uv venv ..\env-hydraulics --python 3.11
uv pip install --python ..\env-hydraulics\Scripts\python.exe -e ".[test]" matplotlib geopandas
..\env-hydraulics\Scripts\python.exe -m pytest tests
```

`dawacotools` is pinned to the published Python-3.8+-compatible revision
`31720fdabe6a0652a0a1b39dbc04c692a41791a6`; a current unpinned revision requires
Python 3.12 and conflicts with downstream's Python 3.11 support. The test-only
Pastas bound retains the `HantushWellModel.numpy_step` independent reference API.

Tests cover analytic partitions and resistance limits, signed shutdown crossflow,
nonzero-history weighted KKT constraints, independent variable-kD quadrature,
steady and zero starts, checkpoint continuation, accelerated-path exercise,
an image-well canal convergence limit, and normalized coefficient provenance.
The downstream synthetic integration independently checks raw-distance KKT/maps,
cleaning-side resistance, infiltration grouping and interval-mean observations.

Real-data acceptance additionally needs verified well identities/counts,
sensor/temperature identities, **all** upstream fitting periods, held-out direct
heads, initial-condition and spatial/temporal sensitivity, and record-specific
resource measurements. Prescribed changing canal level plus pumping-only
drawdown is not an independently consistent transient background-head solution.
Inherited aquifer coefficients and calibration-derived Q95/0.5 m bed assumptions
remain provisional. The old lumped series baseline/growth is not added again;
an unresolved borehole/skin contribution remains a limitation. Common pipe loss
is a diagnostic only, never a filter or map-cell loss.
