# Model Card: a_team DE electricity-load forecaster ("Das A Team")

This card describes the forecasting **system** that team `a_team` deploys for the
DDMO/Numerische-Mathematik live challenge (24-hour German electricity-load
forecast). It ships inside `a_team`'s **reproducibility package** (this folder)
and follows the
[Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook)
taxonomy.

> **Scope note.** The bundled `spotforecast2-safe` library ships its own
> *library* model card, which assigns the duties of a high-risk deployment (EU AI
> Act Art. 9–15) to the **integrator**. This document is that integrator-side
> card: it describes `a_team`'s tuned deployment as packaged here
> (`a_team_script.py` + the frozen data snapshot + the pinned environment), not
> the library. The example evaluation (§7) is the documented run for target day
> **2026-06-22**; re-running the package re-tunes for its own configured target
> day.

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
| Language / runtime | Python 3.13, CPU-only |
| License context | inherits the base library's AGPL-3.0-or-later terms; coursework artefact |

The pipeline itself performs no learning; LightGBM does. The `a_team` variant is
a *configuration and hardening* of the reference pipeline, not a new model
family (see §4).

### Software versions

The dependency set is pinned in this package's **committed `uv.lock`**
(`pyproject.toml` declares the constraints). Key versions:

| Package | Version | Source |
| --- | --- | --- |
| `spotforecast2-safe` | 21.1.0 | committed `uv.lock` |
| `spotoptim` | 1.0.0 | committed `uv.lock` |
| `spotforecast2` | 7.0.0 | committed `uv.lock` |
| `lightgbm` | 4.6.0 | committed `uv.lock` |
| `pandas` | 3.0.3 | committed `uv.lock` |
| `pyarrow` | 24.0.0 | committed `uv.lock` |
| `numpy` | 2.4.6 | committed `uv.lock` |
| `scikit-learn` | 1.9.0 | committed `uv.lock` |
| `shap` | 0.52.0 | committed `uv.lock` |
| `entsoe-py` | 0.8.0 | committed `uv.lock` |

`uv sync` installs exactly these pins. `spotforecast2-safe 21.1.0` and
`spotoptim 1.0.0` comfortably exceed the pipeline's feature floors
(`spotforecast2-safe ≥ 16.4.0` for the target-corruption policy and day-ahead
providers). Note `spotoptim 1.0.0` is **sequential-only** — see §3.

---

## 2. Intended Use and Scope

**Intended use.** Produce a validated 24-hour-ahead hourly forecast of the German
(DE bidding-zone) total electricity load for the challenge leaderboard, one
submission per target day, scored against ENTSO-E ground truth.

**In scope.** DE total load, hourly resolution, 24 h horizon, target day = the
day after the run.

**Out of scope.** Other countries/bidding zones, sub-hourly operational
dispatch, probabilistic/interval forecasts (point forecasts only), and any use
as a safety-critical control signal without independent system-level validation.

---

## 3. How to Reproduce / Get Started

This package is self-contained. With the bundled frozen ENTSO-E snapshot
(`./data/interim/`) and the pinned environment (`uv.lock`, `pyproject.toml`), the
forecast runs offline — no ENTSO-E API key or network required:

```powershell
uv sync                        # install the exact pinned versions from uv.lock
uv run python a_team_script.py
```

`a_team_script.py` is the notebook's code path without plots: registry check
(optional) → coverage guards → PACF lag selection → `ConfigEntsoe` →
`MultiTask(spotoptim)` → `y_0` → submission CSV. It sets
`SPOTFORECAST2_DATA=./data` and **skips any download whose target CSV already
exists**, so the bundled snapshot is used as-is. The ENTSO-E API key is read only
from the `ENTSOE_API_KEY` environment variable and is needed **only** if the
frozen CSVs are absent (never hard-coded — see §9). The submission CSV is written
to `./submissions/<team>/<date>.csv` (set `SUBMISSION_ROOT` to redirect).

### Determinism (CR-2)

`spotoptim 1.0.0` is **sequential-only** — the former `n_jobs_spotoptim`
parallelism was removed — so the surrogate search runs in a fixed order. With
`random_state=42`, the pinned `uv.lock` environment, and the frozen data
snapshot, a re-run reproduces the forecast on a fixed CPU architecture.

> **Two honest caveats.** (1) The date constants (`NOW_UTC`/`TODAY_UTC`/
> `TOMORROW_UTC`) currently follow the system clock; for a fixed-target
> bit-reproducible replay they must be pinned to the target day (and a reference
> CSV + `SHA256SUMS` added). (2) The original 2026-06-22 leaderboard forecast was
> produced earlier with a **parallel pre-1.0 spotoptim**, so it is not
> bit-identical to a fresh sequential re-run; LightGBM floating-point results
> also differ across CPU architectures.

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
| Live horizon (`predict_size`) | `LIVE_N_STEPS` (last 24 sliced as `y_0`) |
| CV geometry | `cv_block_size=24`, `refit_size=7`, `number_folds=10` → 70-fold rolling-origin backtest over 1680 h |
| Tuning budget | `n_trials_spotoptim=100`, `n_initial_spotoptim=50` (sequential — `spotoptim 1.0` dropped `n_jobs_spotoptim`) |
| Exogenous features | `include_weather_windows`, `include_holiday_features` (state="NW") + `include_holiday_adjacency_features` (Brückentag / day before/after), `poly_features_degree=2` capped at `max_poly_features=40` |
| Day-ahead / static providers | `entsoe_wind_forecast`, `entsoe_solar_forecast`, `entsoe_net_load`, `entsoe_day_ahead_price`, `covid_infection_rate` (all leakage-clean: published D-1 or static) |
| ENTSO-E `Forecasted Load` | **not** used as a feature (`include_entsoe_forecast_load=False` → identity `a_team`) |
| Data-quality policy | value-sanity QC (intra-hour range > 8 GW or adjacent step > 6 GW flagged); `target_corruption_policy="truncate"`; provider healing `exog_max_gap_hours=3`, `exog_max_tail_gap_hours=48` |
| Seed | `random_state=42` |

### Feature composition (2026-06-22 example run)

The build produced **174 exogenous columns**, of which **166 were selected for
training** (mutual-information pruning), plus the endogenous lags and rolling
windows. Families:

| Family | Count | What it is |
| --- | ---: | --- |
| weather_window | 90 | rolling weather statistics (e.g. `…_window_7D_mean/max`) |
| polynomial | 40 | degree-2 interactions, top 40 by mutual information |
| weather/other | 23 | base weather variables (temperature, humidity, pressure, wind, cloud cover) |
| cyclical/RBF | 12 | RBF calendar encodings from the 5 `Period`s |
| holiday | 4 | public holiday + Brückentag / day before / day after (NW) |
| entsoe_provider | 4 | wind / solar forecast, net load, day-ahead price |
| covid | 1 | `covid_infection_rate` |

Endogenous: SpotOptim-selected lags `[1, 2, 3, 23, 24, 25, 47, 48, 167, 168, 169,
336]` and anchored rolling-mean windows 72 h / 168 h / 720 h.

### Tuned result (documented 2026-06-22 run)

- Training window: **2024-06-21 14:00 → 2026-06-21 14:00 UTC**; `end_train`
  (last complete hour) = 2026-06-21 14:00 UTC.
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

  *(Tuned values for one specific run; they are re-selected on every run and are
  not fixed defaults. Re-running this package re-tunes on its own target day and
  may differ.)*

> **Artefacts.** The frozen ENTSO-E snapshot ships in `./data/` (interim + raw,
> 2022 → 2026). Running the script writes the tuned model, the tuning-results
> JSON, and logs under `./_cache/` (package-relative). The submitted forecast CSV
> is durably and publicly recorded in the `bartzbeielstein/challenge-leaderboard`
> git history — that public repo is the authoritative submission record.

### Leakage guards (CR-3)

The realised `Actual Load` / ENTSO-E `Forecasted Load` columns must never enter
the training frame, the selected exogenous set, or the fitted model — asserted
fail-loud so a regression breaks the run rather than biasing the model. Only
lagged history and day-ahead/static priors are admissible.

---

## 5. Interfaces and Runtime

- **Input:** the bundled interim ENTSO-E frame (`./data/interim/energy_load.csv`,
  15-min DE load aggregated to hourly by mean) plus the day-ahead provider
  side-tables (`renewable_forecast.csv`, `day_ahead_price.csv`); weather and
  COVID are fetched from public archives at run time (offline they degrade).
- **Output:** a 24-row CSV with columns `timestamp_utc`
  (`YYYY-MM-DDTHH:MM:SSZ`) and `forecast_mw`; values strictly positive, no NaN,
  first step = target day 00:00 UTC.
- **Runtime:** CPU-only, Python 3.13. Sequential SpotOptim (spotoptim 1.0).
- **Persistence:** the tuned forecaster is saved as a `.joblib` model under
  `./_cache/`.

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
the weather/COVID providers and deviate.

---

## 7. Evaluation

**Example target-day accuracy (2026-06-22).** For the documented run, ENTSO-E has
published the actual load, so the forecast can be scored against the 24 published
hourly actuals (mean load ≈ 54,547 MW):

| Metric | Value |
| --- | --- |
| MAE | **2337 MW** (leaderboard scoring metric) |
| RMSE | ≈ 2415 MW |
| MAPE | ≈ 4.4 % |

These are the figures for a **single target day**; accuracy varies from day to
day, so they are not representative of the model's general performance, and a
re-run of this package for its own target day will differ.

**In context.** For scale: a separate, simpler baseline (a stock-LightGBM model
on the same ENTSO-E data — **not part of this package**) measured
**MAE ≈ 2673 MW** on a November eval and a **rolling-backtest mean MAE ≈ 2025 MW**.
The `a_team` target-day **MAE = 2337 MW** sits squarely in that realistic range
for DE load, confirming a genuine, correctly-scaled forecast — not the
flat-forecast failure mode of §4. *(Different model and period; the numbers
anchor the scale, not a like-for-like comparison.)*

**How this system's accuracy is established:**

1. **70-fold rolling-origin backtest** — SpotOptim minimises the pooled MAE of
   24-step-ahead forecasts over the most recent 1680 h; this selects the
   hyperparameters and lags. *(The pooled backtest MAE is written to the
   tuning-results JSON under `./_cache/`.)*
2. **Live leaderboard score** — each daily submission is scored against ENTSO-E
   actuals on the public `bartzbeielstein/challenge-leaderboard`.
3. **Baseline comparison** — the forecast is overlaid against ENTSO-E's own
   day-ahead `Forecasted Load` (warn-only shape check), a reference the model
   never trains on.

**Caveat.** The backtest cannot see the live-recursion self-feedback failure
mode (each fold restarts from observed history and scores only 24 steps); this is
why the anchored-window / weekly-anchored-lag hardening in §4 was introduced
operationally rather than caught by CV.

---

## 8. Model Transparency

Point forecasts, no native uncertainty. The model is white-box: LightGBM split-
and gain-importance are available, and the source notebook additionally computes
SHAP attributions and a top-feature-importance figure by family. Lags and the
anchored window means dominate; weather, calendar, holiday, and the day-ahead
providers contribute the exogenous signal (§4 feature composition).

---

## 9. Operation: Monitoring and Response

- **Monitor:** input data quality (coverage freshness, interior gaps, sanity
  tripwire), provider availability, and daily leaderboard error vs. the ENTSO-E
  baseline.
- **Refit cadence:** every run re-tunes and refits on a fresh 2-year window.
- **Response:** on stale/corrupt data the guards abort or truncate rather than
  fabricate; on provider outage the pipeline degrades gracefully; a persistent
  accuracy regression vs. the ENTSO-E baseline is the signal to revisit the
  factory windows, lag space, or feature menu.
- **Operational security (public git):** this package goes to public git, so no
  secret is committed. The ENTSO-E API key is read only from the `ENTSOE_API_KEY`
  environment variable (fail-loud if a download is needed and it is unset) and is
  **not hard-coded** in `a_team_script.py`; any key that was ever committed
  elsewhere must be rotated at ENTSO-E.

---

## 10. Compliance Support (EU AI Act) and Code Rules

Claims here are deliberately scoped — the caveats are stated per row and in the
code-rule notes below.

| Obligation | Article | Addressed by | Scope / caveat |
| --- | --- | --- | --- |
| Risk management | Art. 9 | 70-fold backtest, deterministic seed, coverage/sanity guards | backtest self-feedback blind spot (§7) |
| Data governance | Art. 10 | leakage guards (CR-3), NaN discipline, `truncate` policy | `truncate`/`skip` are risk-accepted (below) |
| Technical documentation | Art. 11 | this card + `a_team_script.py` + run provenance (§4) + pinned `uv.lock` (§1) | — |
| Record-keeping | Art. 12 | submitted CSVs immutably recorded in the **public** `challenge-leaderboard` git; per-run model + tuning JSON + logs under `./_cache/` | — |
| Transparency | Art. 13 | LightGBM feature importance; SHAP in the source notebook (§8) | SHAP not produced by the package script |
| Accuracy / robustness | Art. 15 | tuned pipeline, backtest distribution, target-day MAE 2337 MW (§7) | scored on one leaderboard day |

**Code-rule scope, stated honestly:**

- **CR-1 (no dead code):** the `spotforecast2-safe` building blocks carry the
  library's ≥ 80 % coverage, and `a_team`'s **own** code (`team4_lgbm_factory`,
  `team4_search_space`, value-sanity constants, leakage asserts) is exercised
  end-to-end on every run with runtime asserts guarding each path; dedicated unit
  tests can be layered on top to lock this in.
- **CR-2 (determinism):** `spotoptim 1.0` is sequential-only, so with seed 42 +
  the pinned `uv.lock` + the frozen snapshot the run is reproducible on a fixed
  architecture; pinning the date constants and adding a reference CSV + checksums
  completes a fully fixed-target replay (§3).
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
| Offline development / tuning | this package's env: `spotoptim`, `optuna`, `matplotlib`, `plotly`, `shap`, `torch`, `xgboost` | uses deny-listed packages, but never ships in the deployed inference path |

> **Enforcing the deny-list.** This package commits a `uv.lock`, so the deny-list
> is now machine-checkable: a test can assert none of plotly / matplotlib /
> optuna / spotoptim / torch / tensorflow appear in the *inference* lockfile —
> the reference package's audit step 1.

---

## 11. Glossary

| Term | Meaning |
| --- | --- |
| PACF | partial autocorrelation function — used to pick `key_lags` |
| Anchored window | rolling-mean feature long enough (≥ 72 h) to stay grounded in observed history over the live recursion |
| `y_0` | the 24 hourly forecast values for the target day, sliced from the live forecast |
| Self-feedback | recursion regime where short-window features are computed from the model's own predictions |
| Brückentag | bridge day between a public holiday and the weekend; behaves like a partial holiday |

---

## 12. How to Audit

1. `uv sync` and confirm the pins match the committed `uv.lock` (§1).
2. `uv run python a_team_script.py` and confirm it runs offline against
   `./data/interim/` (no network / API key needed) and writes
   `./submissions/<team>/<date>.csv`.
3. Confirm the leakage guard passes: `Forecasted Load` / `Actual` absent from the
   training frame, selected exogenous set, and fitted model.
4. Inspect the tuning-results JSON under `./_cache/` for the selected
   lags / hyperparameters, and the model `.joblib`.
5. Cross-check the submitted forecast against the immutable public record in the
   `challenge-leaderboard` git history for the target day.

---

## 13. Authors and Contact

Team `a_team` ("Das A Team"), TH Köln — course "Numerische Mathematik"
(Prof. Thomas Bartz-Beielstein). GitHub handles: obecher, Math1s0, JannhTH,
MarkDT551, Kradid655. Built on `spotforecast2-safe` (Bartz-Beielstein,
AGPL-3.0-or-later).

---

## 14. Disclaimer and Liability

Provided as coursework, as is, without warranty. Forecasts are point estimates
for a leaderboard challenge and must not be used as a safety-critical control
signal without independent system-level validation. The reported target-day
error (§7) is a single-day score, not a guarantee of future accuracy.
