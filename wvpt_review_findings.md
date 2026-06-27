# Transient WVP implementation review findings

Date: 2026-06-24

Multiple GPT-5.5/xhigh review agents inspected the transient WVP implementation from the angles of physics/math, code efficiency and leanness, meaningfulness of tests, consistency with existing accessors, and general correctness. No blockers were found. The findings below are grouped and de-duplicated, while preserving nits.

## Major findings

| Area | Finding | Why it matters | Suggested action |
|---|---|---|---|
| Physics/math | `kD_model` solves `kD` at reference resistance, then divides by viscosity ratio. Hantush resistance is nonlinear in `kD`. | Seasonal/measured temperature `.wvpt` steady state may not match `.wvp`. | Either solve `kD` against temperature-corrected resistance per timestamp, or explicitly document/test the current assumption. |
| Physics/math | `dx_mirrorwell` is doubled internally as canal distance, but the name may imply image-well distance. | Boundary/image wells may be placed twice too far away if semantics are misunderstood. | Confirm semantics; rename/document or remove doubling. Add exact geometry tests. |
| Robustness | Default `leakage_resistance_d` bounds include infeasible values; reviewers reproduced failures for real strangen. | Optimizer can abort when `solve_steady_multiwell_kd` cannot bracket `kD`. | Use feasible per-strang bounds, wider/dynamic `kD` bounds, or return finite penalty residuals. |
| Robustness | One failed strang aborts the full report and the workbook is only written at the end. | Late failure loses successful fits. | Catch/log per-strang failures and persist successful sheets safely. |
| Performance | `kD(t)` root solve runs per timestamp per optimizer evaluation. | Potentially very slow for 24 strangen times thousands of rows. | Cache/interpolate inverse resistance to `kD`; precompute geometry, viscosity, and resistance. |
| Performance | Hantush multiwell path allocates large arrays per term and creates thread pools per call. | Memory/CPU blow-ups are possible, especially at high leakage with `tmax_days_cap=None`. | Cap `tmax`, stream-sum terms, reuse executor/precomputed grids, constrain leakage. |
| API consistency | `.wvpt.dp_model` requires geometry args, unlike `.wvp.dp_model(index, flow, temp_wvp=None)`. | Less accessor-compatible and easier to misuse. | Consider keeping `dp_model` steady-compatible and naming the transient method separately, or make geometry configurable/optional. |
| Data alignment | Flow `Series` inputs are converted to numpy without reindexing. | Reordered Series can silently produce wrong drawdown. | If flow is a Series, reindex to model index before conversion. |
| Dependencies | `weerstand_pandasaccessors.py` now imports SciPy-backed transient helpers, but `scipy` is not declared in `pyproject.toml`. | Importing ordinary accessors may fail in a clean install. | Add SciPy dependency or lazy-import transient helpers. |
| Workbook design | The report writes transient-only sheets, not accessor-ready combined `.wvpt` sheets. | Direct workbook-to-accessor use is not possible without the loader. | Keep the separate workbook if intentional, but document the loader; or write combined sheets. |
| Tests | Several tests use production helpers to generate expected values. | Shared bugs can pass roundtrips. | Add anchored numeric fixtures/reference vectors. |
| Tests | No `nput > 1` unit-conversion test. | Could miss `Q / nput * 24` bugs. | Add explicit multiwell/per-well flow tests. |
| Tests | Missing non-reference temperature/viscosity test. | Would catch the main `kD` temperature concern. | Add seasonal/measured-temperature steady-equivalence regression. |
| Tests | Synthetic leakage fit uses the production path to generate observations. | Circular physics bugs may pass. | Add fixed synthetic data or independently generated reference data. |
| Tests | Pastas test skips if Pastas is absent, but Pastas is not in test dependencies. | External physics validation may not run in CI. | Add Pastas to test extras or replace with fixed reference vectors. |
| Code drift | `src/wvp_transient.py` still has a second transient implementation with hard-coded constants. | Duplicate logic will drift from `.wvpt` and the report. | Refactor prototype to call shared `.wvpt` helpers or retire it. |
| Git hygiene | New report/tests are untracked; Excel lock file is untracked. | Review/CI may miss files; lock file should not be committed. | Add intended files, remove/ignore `~$*.xlsx`. |

## Minor findings and nits

| Severity | Finding | Suggested action |
|---|---|---|
| minor | `model_std` uses population std (`np.std`, `ddof=0`), while the existing report uses pandas sample std. | Use pandas/sample std if comparability matters. |
| minor | `resample_transient_observations` does not check positive drawdown or minimum two timestamps. | Validate before Hantush. |
| minor | 12h mean values are labeled right and modeled as step values at that timestamp. | Define forcing interval convention; consider left/midpoint labels. |
| minor | `least_squares` is used for one scalar with no `max_nfev`. | Consider bounded scalar optimization or evaluation cap. |
| minor | Workbook overwrite is not atomic and fails poorly if Excel has it open. | Write a temp file, then replace; catch lock errors. |
| minor | Stored leakage bounds in coefficient sheets are not reused by fitting. | Read them or remove the metadata. |
| minor | `combine_wvp_coefficients` can create duplicate labels like `model_std` and `gewijzigd`. | Namespace transient metadata or de-duplicate intentionally. |
| minor | `read_series_workbook` returns `{}` silently for missing files. | Add `required=True` for steady/filter inputs. |
| minor | NaN/inf validation is incomplete in geometry/steady solve helpers. | Add explicit finite checks. |
| minor | Plots omit `kD(t)`, flow, bounds, number of observations, and multiwell counts. | Add diagnostics to plot/log/workbook. |
| minor | `main(strangen="IK102")` treats the string as characters. | Normalize a single string to `[strangen]`. |
| minor | Logging clears handlers without closing and may propagate. | Close handlers and set `propagate=False`. |
| nit | `steady_multiwell_resistance_from_kd` docstring units are confusing. | Clarify coefficient units for total `m3/h` input. |
| nit | `storage_coefficient > 1` passes validation. | Consider warning/range check. |
| nit | `WvpResistanceAccessor.temp_model` may emit a pandas FutureWarning; helper already uses `na_action="ignore"`. | Mirror helper behavior. |
| nit | Existing accessor validation uses `assert`, which can be disabled. | Prefer explicit exceptions eventually. |
| nit | `WvpResistanceAccessor.temp_model` constructs `AssertionError` without raising. | Change to `raise AssertionError` or `ValueError`. |
| nit | `objective` uses `print` diagnostics. | Use logging or returned diagnostics. |
| nit | Existing `reports/wvpweerstand.py` ends with `print("hoi")`. | Remove when touching that report. |
| nit | Project-level ruff currently fails due removed rules `S320` and `UP038`. | Clean config separately. |

## Positive confirmations

- `alpha` and `beta` formulas match the physical plan.
- Existing signed WVP convention is mostly preserved: positive drawdown, negative `dp_model`.
- Total `m3/h` to per-well `m3/d` conversion appears correct.
- The report is conceptually lean: one fitted physical parameter.
- Tests already cover sign inversion, zero flow, constant-flow steady initial condition, image-well qualitative behavior, workbook roundtrip, and synthetic fit recovery.

## Suggested first solve batch

1. Temperature/`kD` consistency.
2. Infeasible leakage bounds and optimizer penalty behavior.
3. Per-strang failure handling and safe persistence.
4. SciPy dependency/import strategy.
5. Missing non-reference-temperature and `nput > 1` tests.

## Interactive triage decisions

Date: 2026-06-24

The findings below were triaged interactively. Items marked **Solve in next session** are the selected scope for the follow-up implementation session.

### Solve in next session

| ID | Finding | Decision notes |
|---|---|---|
| `leakage-bounds` | Default `leakage_resistance_d` bounds include infeasible values and can abort optimization. | Make fitting robust when candidate leakage values cannot bracket a valid `kD`. |
| `per-strang-failure-persistence` | One failed strang aborts the full report and the workbook is only written at the end. | Catch/log failures per strang and persist successful fits. |
| `scipy-dependency` | SciPy-backed transient helpers are imported by accessors but `scipy` is not declared in `pyproject.toml`. | Add `scipy` to `pyproject.toml` dependencies. |
| `nput-temp-tests` | Missing `nput > 1` and non-reference-temperature tests. | Add `nput > 1` flow-conversion coverage and non-reference-temperature behavior coverage. |
| `series-alignment` | Flow `Series` inputs are converted to numpy without reindexing to the model index. | Reindex Series inputs to the model index before conversion. |
| `mirrorwell-semantics` | `dx_mirrorwell` is doubled internally as canal distance but the name may imply image-well distance. | Rename to `r_mirrorwel` and keep using doubled boundary distances. |
| `workbook-design` | Report writes transient-only sheets, not accessor-ready combined `.wvpt` sheets. | Write combined accessor-ready `.wvpt` coefficient sheets. |
| `test-reference-fixtures` | Several tests use production helpers to generate expected values and Pastas can skip. | Add Pastas to test dependencies so external Hantush validation is not silently skipped. |
| `atomic-workbook-write` | Workbook overwrite is not atomic and fails poorly if Excel has it open. | Include atomic workbook writes and clear Excel-lock failure behavior. |
| `minor-nits` | Miscellaneous minor/nit findings from the review. | Include all minor/nit cleanup in the next session. |

### Needs discussion before solving

| ID | Finding | Decision notes |
|---|---|---|
| `temp-kd-consistency` | `kD_model` solves `kD` at reference resistance and then divides by viscosity ratio, although Hantush resistance is nonlinear in `kD`. | Discuss the physical interpretation before changing behavior. |
| `dp-model-api` | `.wvpt.dp_model` requires geometry args unlike `.wvp.dp_model`. | Discuss accessor API compatibility before implementation. |

### Deferred

| ID | Finding | Decision notes |
|---|---|---|
| `performance-kd-cache` | `kD(t)` root solve runs per timestamp per optimizer evaluation. | Defer larger optimization until correctness and robustness settle. |
| `hantush-memory` | Hantush multiwell path can allocate large arrays per term and create thread pools per call. | Defer larger optimization until correctness and robustness settle. |
| `src-wvp-transient-drift` | `src/wvp_transient.py` duplicates transient logic with hard-coded constants. | Defer until accessor/report behavior is stable. |
