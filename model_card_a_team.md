# Model Card: a_team DE electricity-load forecaster ("Das A Team")

This card describes the forecasting **system** that team `a_team` deploys for the
DDMO/Numerische-Mathematik live challenge (24-hour German electricity-load
forecast). It follows the
[Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook)
taxonomy.

> **Scope note.** The bundled `spotforecast2-safe` library ships its own
> *library* model card. That card explicitly assigns the duties of a high-risk
> deployment (EU AI Act Art. 9–15) to the **integrator**. This document is that
> integrator-side card: it describes `a_team`'s concrete tuned deployment, not
> the library. Facts here are taken from `14_team_4_submission.ipynb` and its
> executed run for target day **2026-06-22**.

---

## 1. Model Details

| Field | Value |
| --- | --- |
| System name | `a_team` DE load forecaster |
| Leaderboard identity | `a_team` (display "Das A Team"); `a_team_entsoe` when `include_entsoe_forecast_load=True` — see §4 |
| Team members (GitHub) | obecher, Math1s0, JannhTH, MarkDT551, Kradid655 |
| Type | Recursive multi-step LightGBM forecaster, SpotOptim-tuned, built on `spotforecast2` (`MultiTask`) + `spotforecast2-safe` |
| Base library | `spotforecast2-safe` (deterministic feature engineering + recursive forecaster wrappers); tuning via `spotoptim` |
| Developed by | team `a_team`, TH Köln — course "Numerische Mathematik" (Prof. Bartz-Beielstein) |
| Derived from | chapter 14 reference `team_4` submission (professor's template) |
| Language / runtime | Python 3.13.13 (notebook kernel `.venv`), CPU-only |
| License context | inherits the base library's AGPL-3.0-or-later terms; coursework artefact |

The pipeline itself performs no learning; LightGBM does. The `a_team` variant is
a *configuration and hardening* of the reference pipeline, not a new model
family (see §4).

### Software versions

Environment: **Python 3.13.13** (notebook kernel `.venv`), CPU-only. The
dependency set is governed by the course's shared `uv.lock`. Each version below
is tagged with its evidence source in this repository.

| Package | Version | Source |
| --- | --- | --- |
| `spotforecast2-safe` | ≥ 16.4.0 (corruption policy; providers ≥ 15.7, adjacency ≥ 15.9, healing ≥ 16.1/16.2) | notebook (minimum requirement) |
| `spotoptim` | ≥ 0.12.7 (log-scale integer-dimension fix, §4) | notebook (minimum requirement) |
| `pandas` | 3.0.2 | parquet metadata (repo cache) |
| `pyarrow` / Arrow C++ | 24.0.0 | parquet metadata (repo cache) |
| `spotforecast2` | 5.1.1 | course reference env |
| `lightgbm` | 4.6.0 | course reference env |
| `numpy` | 2.4.6 | course reference env |
| `scikit-learn` | 1.9.0 | course reference env |
| `shap` | 0.52.0 | course reference env |
| `entsoe-py` | 0.8.0 | course reference env |

`pandas` / `pyarrow` were read directly from the repository's cached parquet
artefacts (the baseline `pipeline.py` cache — see the artefact note in §4). The
"course reference env" pins are those recorded in the professor's `team_4`
reproducibility bundle (2026-06-07) — the shared course environment `a_team`
builds on; they satisfy the notebook's minimum requirements above.

---

## 2. Intended Use and Scope

**Intended use.** Produce a validated 24-hour-ahead hourly forecast of the German
(DE bidding-zone) total electricity load for the challenge leaderboard, one
submission per target day, scored against ENTSO-E ground truth.

**In scope.** DE total load, hourly resolution, 24 h horizon, target day = the
day after the run.

**Out of scope.** Other countries/bidding zones, sub-hourly operational
dispatch, probabilistic/interval forecasts (the model emits point forecasts
only), and any use as a safety-critical control signal without independent
system-level validation.

---

## 3. How to Reproduce / Get Started

The forecast is produced by running the notebook end-to-end (cells must run top
to bottom; each mutates `team4_mt` in place):

```python
# after prepare_data / outliers / impute / build_exogenous_features:
team4_mt.run_task_spotoptim(search_space=team4_search_space, show=False)
future = team4_mt.results["spotoptim"]["Actual Load"]["future_pred"]
y0 = future.loc[TOMORROW_UTC:LAST_TARGET]     # the 24 h submitted forecast
```

A standalone, historically replayable variant of the same pipeline ships with
the **lecture material** as `lecture/scripts/team4_submit.py` (download →
coverage guards → PACF lag selection → `ConfigEntsoe` → `MultiTask(spotoptim)` →
`y_0` → submission CSV → validate → optional PR). It is **not** part of this
folder.

### Live vs. archive forecast — reproducibility (CR-2)

Two distinct profiles exist and must not be conflated:

| Profile | Settings | Reproducibility |
| --- | --- | --- |
| **Operational** (produced the submitted forecast) | parallel SpotOptim `n_jobs_spotoptim=-1`, 100 trials; weather/COVID fetched live | **not** bit-reproducible (parallel scheduling + non-pinned live archives) |
| **Deterministic archive** (for audit) | serial SpotOptim `n_jobs_spotoptim=None`, `random_state=42`, frozen snapshot | bit-reproducible on a fixed architecture |

The **submitted** leaderboard forecast used the operational profile and is
therefore **not** bit-reproducible. For an auditable copy, re-run with
`n_jobs_spotoptim=None`, seed 42, and a frozen data snapshot. Weather and COVID
providers fetch historical archive data live; offline they degrade gracefully
(`on_exog_provider_failure="skip"`) and the result then deviates.

> **Known gap.** A full reproducibility bundle (frozen ENTSO-E snapshot +
> `uv.lock` + deterministic reference CSV + `SHA256SUMS`), like the reference
> `team_4` package, does not yet exist for `a_team`.

---

## 4. Technical Specification

### Task and model family

Recursive multi-step forecasting of a univariate target ("Actual Load") from its
own lags, rolling-window features, cyclic calendar encodings, and day-ahead
exogenous regressors. Base estimator: `LGBMRegressor` inside a
`ForecasterRecursive`; over the live horizon the model is applied recursively,
feeding its own predictions back as lag/window inputs.

### Custom forecaster factory (`team4_lgbm_factory`)

The defining `a_team` change vs. the reference: **anchored rolling-mean windows
at 72 h / 168 h / 720 h** and the **default L2 objective** (the stock factory
uses a single 72 h mean). Rationale — the documented **2026-06-05 flat-forecast
incident**: short 24 h windows and an L1 objective collapse into self-feedback
over the ~28 h live recursion (at a 28 h horizon a 24 h window is 100 %
prediction-fed; a 168 h window only ~17 %), flattening the forecast. Long
anchored windows stay grounded in observed history.

### Configuration (`ConfigEntsoe` — `team4_config`)

| Item | Value |
| --- | --- |
| Seasonal `Period` encodings | daily, weekly, monthly, quarterly, yearly (5) |
| Candidate lags (`lags_consider`) | PACF-selected `key_lags = [1, 2, 3, 15, 24, 25, 168, 169]`, warm-started |
| Search space | custom `team4_search_space`; every lag candidate carries the weekly anchor 167/168; linear `n_estimators` range |
| `train_size` | 2 years (avoids the 2022/23 energy-crisis regime) |
| Live horizon (`predict_size`) | `LIVE_N_STEPS` = 33 (last 24 sliced as `y_0`) |
| CV geometry | `cv_block_size=24`, `refit_size=7`, `number_folds=10` → 70-fold rolling-origin backtest over 1680 h |
| Tuning budget | `n_trials_spotoptim=100`, `n_initial_spotoptim=50`, `n_jobs_spotoptim=-1` |
| Exogenous features | `include_weather_windows`, `include_holiday_features` (state="NW") + `include_holiday_adjacency_features` (Brückentag / day before/after), `poly_features_degree=2` capped at `max_poly_features=40` |
| Day-ahead / static providers | `entsoe_wind_forecast`, `entsoe_solar_forecast`, `entsoe_net_load`, `entsoe_day_ahead_price`, `covid_infection_rate` (all leakage-clean: published D-1 or static) |
| ENTSO-E `Forecasted Load` | **not** used as a feature (`include_entsoe_forecast_load=False` → identity `a_team`) |
| Data-quality policy | value-sanity QC (intra-hour range > 8 GW or adjacent step > 6 GW flagged); `target_corruption_policy="truncate"`; provider healing `exog_max_gap_hours=3`, `exog_max_tail_gap_hours=48` |
| Seed | `random_state=42` |

### Tuned result (target day 2026-06-22 run)

- Training window: **2024-06-21 14:00 → 2026-06-21 14:00 UTC**; `end_train`
  (last complete hour) = 2026-06-21 14:00 UTC.
- Feature pipeline: +5 provider columns, 40 of 5995 polynomial interactions
  kept; combined exog shape (39217, 174); **166 exogenous features selected**;
  merged training frame (39184, 167).
- SpotOptim best lags: **`[1, 2, 3, 23, 24, 25, 47, 48, 167, 168, 169, 336]`**
  (the extended + two-week-lag candidate).
- Artefacts (Art. 12 record-keeping): saved model
  `<cache_home>/models/ddmo-live-team4/ddmo-live-team4_Actual Load_spotoptim_20260621_165512.joblib`;
  tuning-results JSON
  `<cache_home>/tuning_results/ddmo-live-team4_Actual Load_spotoptim_20260621_165446.json`.
- SpotOptim best hyperparameters:

  | Hyperparameter | Value |
  | --- | --- |
  | num_leaves | 484 |
  | max_depth | 24 |
  | learning_rate | 0.0460 |
  | n_estimators | 1744 |
  | bagging_fraction | 0.507 |
  | feature_fraction | 0.549 |
  | reg_alpha | 1.254 |
  | reg_lambda | 8.333 |
  | random_state | 42 |

  *(These are the tuned values for one specific run; they are re-selected on
  every operational run and are not fixed defaults.)*

> **Note on artefact availability.** The **submitted forecast CSV is durably and
> publicly recorded** in the `bartzbeielstein/challenge-leaderboard` git history
> (the leaderboard is public) — that public repository, not this folder, is the
> authoritative submission record. The tuned model and tuning-results JSON live
> in the external `<cache_home>` and are local-only. The `_cache/` directory
> shipped in this folder belongs to the separate **baseline** `pipeline.py` run
> (2024 ENTSO-E data, executed 2026-05-04), used only for the scale reference in
> §7 — not the advanced run documented here. For a fully self-contained local
> package, add the tuned model, the tuning-results JSON, and the advanced-run
> log; the submitted CSV need not be copied — cite its leaderboard-git location.

### Leakage guards (CR-3)

The realised `Actual Load` / ENTSO-E `Forecasted Load` columns must never enter
the training frame, the selected exogenous set, or the fitted model — asserted
fail-loud so a regression breaks the run rather than biasing the model. Only
lagged history and day-ahead/static priors are admissible.

---

## 5. Interfaces and Runtime

- **Input:** the merged interim ENTSO-E frame (`interim/energy_load.csv`, 15-min
  DE load, aggregated to hourly by mean) plus the day-ahead provider side-tables
  and live weather/COVID archives.
- **Output:** a 24-row CSV with columns `timestamp_utc`
  (`YYYY-MM-DDTHH:MM:SSZ`) and `forecast_mw`; values strictly positive, no NaN,
  first step = target day 00:00 UTC.
- **Runtime:** CPU-only, Python 3.13.13. Operational run ~5–10 min (parallel
  SpotOptim); deterministic replay is much slower (serial).
- **Persistence:** the tuned forecaster is saved as a `.joblib` model under the
  configured cache home.

---

## 6. Data and Operational Design Domain (ODD)

| Condition | Valid range | Outside the range |
| --- | --- | --- |
| Target | DE total load, hourly, regular monotonic UTC index | error / unreliable |
| Coverage freshness | last published actual within the feed-lag tolerance; interior gaps guarded | fail-loud abort |
| Frontier hour | only hours with all quarter-hour samples published may anchor the recursion | truncated to last sound hour |
| Target sanity | intra-hour range ≤ 8 GW, adjacent step ≤ 6 GW (DST week may need exemption) | flagged; `truncate` policy retracts `end_train` |
| Exogenous providers | complete on the training window (bounded healing ≤ 3 h interior, ≤ 48 h tail) | provider dropped (`skip`), fewer features |
| Training regime | post-energy-crisis (2 y window) | crisis-era demand deliberately excluded |

**Data source & attribution.** Load, day-ahead renewable-forecast and
day-ahead-price inputs are ENTSO-E Transparency Platform data (DE bidding zone,
<https://transparency.entsoe.eu/>), reused under its free data-reuse terms;
weather from Open-Meteo (no key); COVID incidence from the bundled RKI vintage.

**Known limitations.** Forecast accuracy is bounded by LightGBM and the training
data; concept drift, weather-forecast error, holidays, and DST transitions
degrade it. Point forecasts only — no calibrated uncertainty. Offline runs lose
the weather/COVID providers and deviate from the reference.

---

## 7. Evaluation

**Target-day accuracy (2026-06-22).** ENTSO-E has since published the actual load
for the target day, so the forecast can now be scored against the 24 published
hourly actuals (mean load ≈ 54,547 MW):

| Metric | Value |
| --- | --- |
| MAE | **2337 MW** (leaderboard scoring metric) |
| RMSE | ≈ 2415 MW |
| MAPE | ≈ 4.4 % |

These are the figures for a **single target day**; accuracy varies from day to
day, so they are not representative of the model's general performance. They
supersede the render-time `metrics_future = {mae ≈ 46.6 MW}`, a **placeholder**
emitted before the actuals existed ("MAE/MAPE figures render as zero placeholders
until the actuals arrive") that must never be read as accuracy.

**In context.** The companion baseline pipeline in this repo (`pipeline.py`, a
simpler stock-LightGBM model on the same ENTSO-E data, run 2026-05-04, recorded
in `_cache/provenance.json`) measured **MAE = 2673.4 MW** on its November eval
and a **rolling-backtest mean MAE = 2024.8 MW**. The `a_team` target-day
**MAE = 2337 MW** sits squarely in that realistic range for DE load, confirming a
genuine, correctly-scaled forecast — not the flat-forecast failure mode of §4.
*(The baseline is a separate model evaluated on a different period; its numbers
anchor the scale, they are not a like-for-like comparison.)*

**How this system's accuracy is actually established:**

1. **70-fold rolling-origin backtest** — SpotOptim minimises the pooled MAE of
   24-step-ahead forecasts over the most recent 1680 h (70 days). This is the
   validation signal that selects the hyperparameters and lags above. *(The
   pooled backtest MAE is retained in the tuning-results JSON referenced in §4.)*
2. **Live leaderboard score** — each daily submission is scored against ENTSO-E
   actuals once they publish, on the public
   `bartzbeielstein/challenge-leaderboard`.
3. **Baseline comparison** — the forecast is overlaid against ENTSO-E's own
   day-ahead `Forecasted Load` (warn-only shape check), used as an independent
   reference the model never trains on.

**Caveat.** The backtest cannot see the live-recursion self-feedback failure
mode (every fold restarts from observed history and scores only 24 steps); this
is exactly why the anchored-window / weekly-anchored-lag hardening in §4 was
introduced operationally rather than caught by CV.

---

## 8. Model Transparency

Point forecasts, no native uncertainty. The model is white-box: LightGBM split-
and gain-importance are available, and the notebook additionally computes SHAP
attributions and a top-feature-importance figure (coloured by feature family).
Lags and the anchored window means dominate; weather, calendar, holiday, and the
day-ahead providers contribute the exogenous signal.

---

## 9. Operation: Monitoring and Response

- **Monitor:** input data quality (coverage freshness, interior gaps, sanity
  tripwire), provider availability, and daily leaderboard error vs. the ENTSO-E
  baseline.
- **Refit cadence:** every operational run re-tunes and refits on a fresh 2-year
  window (`refit_size=7` governs the backtest refit spacing).
- **Response:** on stale/corrupt data the guards abort or truncate rather than
  fabricate; on provider outage the pipeline degrades gracefully; a persistent
  accuracy regression vs. the ENTSO-E baseline is the signal to revisit the
  factory windows, lag space, or feature menu.
- **Operational security (public git):** submissions and any shared working tree
  are stored in public git, so no secret may be committed. The ENTSO-E API key is
  read only from the `ENTSOE_API_KEY` environment variable (fail-loud if unset)
  and is not hard-coded anywhere in the notebook or the companion scripts
  (`pipeline.py`, `stromverbrauch_vorhersage.py`); any key that was ever
  committed must be rotated at ENTSO-E.

---

## 10. Compliance Support (EU AI Act) and Code Rules

Claims here are deliberately scoped — the caveats are stated per row and in the
code-rule notes below.

| Obligation | Article | Addressed by | Scope / caveat |
| --- | --- | --- | --- |
| Risk management | Art. 9 | 70-fold backtest, deterministic seed, coverage/sanity guards | backtest self-feedback blind spot (§7) |
| Data governance | Art. 10 | leakage guards (CR-3), NaN discipline, `truncate` policy | `truncate`/`skip` are risk-accepted (below) |
| Technical documentation | Art. 11 | this card + notebook + run provenance (§4) + dependency versions (§1) | — |
| Record-keeping | Art. 12 | submitted CSVs immutably recorded in the **public** `challenge-leaderboard` git; notebook-with-outputs; structured JSON audit log (`_cache/logs/*.log`, `schema_version 1.0.0`) | model / tuning JSON are local-only (§4) |
| Transparency | Art. 13 | feature importance + SHAP (§8) | — |
| Accuracy / robustness | Art. 15 | tuned pipeline, backtest distribution, target-day MAE 2337 MW (§7) | scored on one leaderboard day |
| Cybersecurity (supply-chain) | Art. 15 | OpenSSF Scorecard runs in CI (`publish_results: true`), badge in README; License / Security-Policy / Dependency-Update-Tool / Vulnerabilities checks pass | aggregate 3.6/10 at time of writing; many Scorecard checks are N/A for a coursework repo (see Dependency Governance) |

**Code-rule scope, stated honestly:**

- **CR-1 (no dead code):** the `spotforecast2-safe` building blocks carry the
  library's ≥ 80 % coverage; `a_team`'s **own** code (`team4_lgbm_factory`,
  `team4_search_space`, value-sanity constants, leakage asserts) is exercised
  end-to-end on every render and backed by runtime asserts, but is **not yet
  covered by dedicated unit tests**.
- **CR-2 (determinism):** satisfied only by the **deterministic archive
  profile** (§3); the **submitted** forecast used the non-deterministic
  operational profile.
- **CR-3 (fail-safe):** strong fail-loud guards (leakage, coverage, sanity);
  `target_corruption_policy="truncate"`, `on_exog_provider_failure="skip"` and
  `on_weather_failure="skip"` are **documented, WARNING-logged, risk-accepted**
  deviations from a strict "always raise" reading (the conservative `"abort"`
  knob stays available).
- **CR-4 (minimal CVE surface):** the **safety-critical inference artefact**
  (the `.joblib` model + `spotforecast2-safe`) is deny-list-clean; the
  deny-listed packages live only in the offline tuning environment — see
  Dependency Governance below.

Full system-level certification remains the integrator's responsibility.

### Dependency Governance (CR-4)

The reference library card requires that plotly, matplotlib, optuna, spotoptim,
torch, and tensorflow stay out of the safety-critical package, and that "tuning
belongs in a separate workflow outside the safety-critical environment."
`a_team` respects this by **separating two environments**:

| Environment | Contents | Deny-list status |
| --- | --- | --- |
| Safety-critical inference | tuned `.joblib` model + `spotforecast2-safe` (+ its permissive deps) | clean |
| Offline development / tuning | the notebook: `spotoptim`, `optuna`, `matplotlib`, `plotly`, `shap` | uses deny-listed packages, but never ships in the deployed path |

> **Known gap.** An automated deny-list test against a committed `uv.lock` (the
> reference package's audit step 1) is not yet in place; this repository
> currently ships no lockfile.

### Supply-chain security posture (OpenSSF Scorecard)

The deployment repository is continuously scanned by **OpenSSF Scorecard** via a
GitHub Action (`publish_results: true`), so the score is independently
verifiable rather than self-asserted and refreshes on every push:

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Math1s0/numerische-mathematik-th-koeln/badge)](https://scorecard.dev/viewer/?uri=github.com/Math1s0/numerische-mathematik-th-koeln)

At the time of writing the aggregate score is **3.6/10**. As Scorecard targets
large open-source supply chains, several checks (Fuzzing, Signed-Releases,
Contributors, CII-Best-Practices, Packaging) are **not applicable** to a
single-team coursework repository and are expected to score zero. The checks that
*are* meaningful here and pass are License, Security-Policy,
Dependency-Update-Tool (Dependabot), and Vulnerabilities (no known-vulnerable
dependencies). This partially addresses the CR-4 dependency-governance gap noted
above by adding an automated, external supply-chain signal; a committed
`uv.lock` and an automated deny-list test remain the outstanding items.

Report: <https://scorecard.dev/viewer/?uri=github.com/Math1s0/numerische-mathematik-th-koeln>

---

## 11. Glossary

| Term | Meaning |
| --- | --- |
| PACF | partial autocorrelation function — used to pick `key_lags` |
| Anchored window | rolling-mean feature long enough (≥ 72 h) to stay grounded in observed history over the live recursion |
| `y_0` | the 24 hourly forecast values for the target day, sliced from the 33-step live forecast |
| Self-feedback | recursion regime where short-window features are computed from the model's own predictions |
| Brückentag | bridge day between a public holiday and the weekend; behaves like a partial holiday |

---

## 12. How to Audit

1. Confirm the leakage guard passes: `Forecasted Load` / `Actual` absent from
   the training frame, selected exogenous set, and fitted model.
2. Re-run the deterministic profile (serial SpotOptim, `random_state=42`) and
   compare the submission CSV against a stored reference.
3. Check the submission schema with the leaderboard's own
   `scripts/validate_submission.py`.
4. Verify the training window, `end_train`, and horizon against the coverage
   guards for the target day.
5. Inspect the tuning-results JSON for the selected lags/hyperparameters.
6. Cross-check the submitted forecast against the immutable public record in the
   `challenge-leaderboard` git history for the target day.
7. Review the automated OpenSSF Scorecard report
   (<https://scorecard.dev/viewer/?uri=github.com/Math1s0/numerische-mathematik-th-koeln>)
   for the repository's supply-chain hygiene; the README badge reflects the
   current aggregate score.

---

## 13. Authors and Contact

Team `a_team` ("Das A Team"), TH Köln — course "Numerische Mathematik"
(Prof. Thomas Bartz-Beielstein). GitHub handles: obecher, Math1s0,
JannhTH, MarkDT551, Kradid655. Built on `spotforecast2-safe`
(Bartz-Beielstein, AGPL-3.0-or-later).

---

## 14. Disclaimer and Liability

Provided as coursework, as is, without warranty. Forecasts are point estimates
for a leaderboard challenge and must not be used as a safety-critical control
signal without independent system-level validation. The reported target-day
error (§7) is a single-day score, not a guarantee of future accuracy.
