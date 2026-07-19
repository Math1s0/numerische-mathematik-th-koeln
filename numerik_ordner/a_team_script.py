
# %% Cell 1
#| label: team4-registry
#| echo: true
from pathlib import Path
import os
import yaml

# Optional registry check against a local challenge-leaderboard clone. Set
# CHALLENGE_LEADERBOARD_DIR to enable it; skipped when the file is absent so the
# reproducibility package runs standalone, without the leaderboard repo.
_lb_dir = os.environ.get("CHALLENGE_LEADERBOARD_DIR")
teams_yml = (Path(_lb_dir) / "teams.yml") if _lb_dir else Path("teams.yml")
if teams_yml.exists():
    teams = yaml.safe_load(teams_yml.read_text())["teams"]
    for tid in ("a_team", "a_team_entsoe"):
        team = next((t for t in teams if t["id"] == tid), None)
        assert team is not None, f"{tid} is not in challenge-leaderboard/teams.yml"
        print(f"{tid:<14} display_name: {team['display_name']:<24} "
              f"github: {', '.join(team['github_handles'])}")
else:
    print("teams.yml not found; skipping registry check "
          "(set CHALLENGE_LEADERBOARD_DIR to enable).")

# %% Cell 2
#| label: team4-imports
#| echo: true
import logging, os
from pathlib import Path
import pandas as pd

# Parallel SpotOptim tuning (`n_jobs_spotoptim`, @sec-team4-config) runs the
# surrogate search across CPU cores via a process pool. Under a Jupyter kernel
# --- which is how Quarto executes this chapter --- the platform-default
# "spawn" start method cannot re-import the notebook's __main__, so every
# worker evaluation fails ("All initial design evaluations failed"). Switching
# to "fork" (the Linux default; also available on macOS) lets workers inherit
# the already-imported modules and the objective, so the pool works at render
# time. This call must run before any pool is created. It is harmless where
# "fork" is already the default; on a platform without "fork" the guard falls
# through and you should set `n_jobs_spotoptim=None` (sequential) instead.
import multiprocessing as _mp
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
try:
    _mp.set_start_method("fork", force=True)
except (RuntimeError, ValueError):
    pass

logging.basicConfig(level=logging.WARNING)

# Isolate this chapter's storage from chapter 13 -----------------------------
# download_new_data() writes the merged ENTSO-E mirror to
# <SPOTFORECAST2_DATA>/interim/energy_load.csv, and MultiTask persists tuned
# models, tuning results, and logs under <SPOTFORECAST2_CACHE>. Both default to
# a single shared location in $HOME, so chapters 13 and 14 would otherwise read
# and write the *same* interim download. Pinning per-chapter homes here — before
# any download or MultiTask call — gives this chapter its own download, cache,
# and log directory. The package reads both env vars at call time via
# get_data_home() / get_cache_home(), so setting them now is sufficient.
# Portable, package-relative homes: the frozen ENTSO-E snapshot ships in ./data,
# and tuned models / logs are written under ./_cache next to this script.
_HERE = Path(__file__).resolve().parent
os.environ["SPOTFORECAST2_DATA"]  = str(_HERE / "data")
os.environ["SPOTFORECAST2_CACHE"] = str(_HERE / "_cache")

from lightgbm import LGBMRegressor
from spotforecast2_safe.data import Period
from spotforecast2_safe.downloader.entsoe import (
    download_new_data,
    download_renewable_forecast,
    download_day_ahead_price,
)
from spotforecast2_safe.data.fetch_data import get_data_home
from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import RollingFeatures
from spotforecast2.tasks.task_entsoe import (
    entsoe_data_loader,
    entsoe_test_data_loader,
)
from spotforecast2.multitask import MultiTask

# %% Cell 3
#| label: team4-constants
#| echo: true
NOW_UTC = pd.Timestamp.now(tz="UTC")
TODAY_UTC = NOW_UTC.normalize()
YESTERDAY_UTC = TODAY_UTC - pd.Timedelta(days=1)
TOMORROW_UTC = TODAY_UTC + pd.Timedelta(days=1)

# The earliest available timestamp: 2014-12-31 23:00 UTC
# START_DOWNLOAD = "201501010000"
START_DOWNLOAD = "202201010000"
# Fetch through *tomorrow* so the interim file lands two things: (a) the most
# current ENTSO-E actuals already published today --- which shrinks the
# recursive bridge and horizon --- and (b) ENTSO-E's own day-ahead
# `Forecasted Load` for the target day, fed to the model as the
# `entsoe_forecasted_load` feature (@sec-team4-exog) and also drawn in the
# comparison plot in @sec-team4-vs-entsoe. Future timestamps simply
# carry no Actual Load yet, so the training cutoff and horizon are unchanged.
END_DOWNLOAD = (TOMORROW_UTC + pd.Timedelta(days=1)).strftime("%Y%m%d%H%M")

print(f"NOW_UTC       = {NOW_UTC}")
print(f"TODAY_UTC     = {TODAY_UTC}")
print(f"TOMORROW_UTC  = {TOMORROW_UTC}  (target day = y_0)")
print(f"END_DOWNLOAD  = {END_DOWNLOAD}  (covers tomorrow for ENTSO-E's day-ahead forecast)")

# %% Cell 4
#| label: team4-download
#| echo: true

import os
from spotforecast2_safe.downloader.entsoe import download_new_data

# The API key is read from the environment only — never hard-coded. It is only
# needed when the frozen CSVs under ./data/interim are absent; with the bundled
# snapshot present the download is skipped and no key / network is required.
api_key = os.environ.get("ENTSOE_API_KEY")

_energy_csv = get_data_home() / "interim" / "energy_load.csv"
if _energy_csv.exists():
    print(f"using frozen snapshot: {_energy_csv}")
else:
    assert api_key, "ENTSOE_API_KEY must be set in the environment to download the data."
    download_new_data(
        api_key=api_key,
        country_code="DE",
        start=START_DOWNLOAD,
        end=END_DOWNLOAD,
        force=True,  # bypass the 24 h cooldown so every render pulls a fresh mirror
        keep_forecast_future=True,
        timeout=60,  # bound stalled reads: a dead socket once hung a render forever
    )

# %% Cell 5
interim_csv = get_data_home() / "interim" / "energy_load.csv"
interim = pd.read_csv(interim_csv, index_col=0, parse_dates=True)
interim.index = pd.to_datetime(interim.index, utc=True)

required_last = TODAY_UTC - pd.Timedelta(hours=1)
if interim.index.max() < required_last:
    raise RuntimeError(
        f"ENTSO-E coverage is stale: last interim row is {interim.index.max()} "
        f"but the chapter requires at least {required_last}. "
        "Re-run download_new_data(..., force=True) or check the ENTSO-E API."
    )

# ENTSO-E publishes the German actual load with a normal lag of a few hours,
# so the actuals legitimately trail `now` (especially early in the UTC day).
# Guard only against a genuinely stale feed: the last published actual must lie
# within MAX_ACTUAL_LAG of now. Raise the tolerance if your feed lags more.
MAX_ACTUAL_LAG = pd.Timedelta(hours=36)
last_actual = interim["Actual Load"].dropna().index.max()
if last_actual < NOW_UTC - MAX_ACTUAL_LAG:
    lag_h = int((NOW_UTC - last_actual) / pd.Timedelta(hours=1))
    raise RuntimeError(
        f"Actual Load is stale: last published actual is {last_actual}, "
        f"{lag_h} h before now ({NOW_UTC}); tolerance is "
        f"{int(MAX_ACTUAL_LAG / pd.Timedelta(hours=1))} h. "
        "Re-run download_new_data(..., force=True) or wait for ENTSO-E to "
        "publish more recent actuals."
    )

# The two edge checks above cannot see a hole in the middle of the feed: the
# index stays fresh (Forecasted Load) and the last actual stays recent even
# when a publication outage wiped out a full day in between (observed
# 2026-06-02). Scan the recent window the lag features depend on and fail
# loudly on any oversized interior gap between consecutive published actuals.
GAP_SCAN_DAYS = 28
MAX_ACTUAL_GAP = pd.Timedelta(hours=12)
recent_actuals = (
    interim["Actual Load"]
    .loc[NOW_UTC - pd.Timedelta(days=GAP_SCAN_DAYS):]
    .dropna()
)
actual_gaps = recent_actuals.index.to_series().diff()
oversized = actual_gaps[actual_gaps > MAX_ACTUAL_GAP]
if not oversized.empty:
    gap_list = ", ".join(
        f"{end - width} → {end} ({width})" for end, width in oversized.items()
    )
    raise RuntimeError(
        f"Actual Load has interior gaps wider than {MAX_ACTUAL_GAP} within "
        f"the last {GAP_SCAN_DAYS} days: {gap_list}. ENTSO-E may have "
        "published late; re-run download_new_data(..., force=True) once the "
        "late actuals are published, and make sure no stale partial pull in raw/ "
        "is masking them (current spotforecast2-safe merges raw pulls "
        "NaN-safely: the newest non-missing value wins)."
    )
print(f"interim CSV last row: {interim.index.max()}")
print(interim["Actual Load"].dropna().tail(6))

# %% Cell 6
#| label: team4-entsoe-predictions
#| echo: true
predictions_entsoe = interim["Forecasted Load"].resample("h").mean().dropna()
predictions_entsoe.name = "Forecasted Load"

print(f"predictions_entsoe: {len(predictions_entsoe)} hourly values "
      f"| {predictions_entsoe.index.min()} → {predictions_entsoe.index.max()}")
print(predictions_entsoe.tail(6))

# %% Cell 7
#| label: team4-native-resolution
#| echo: true
native_step = interim.index.to_series().diff().dropna().mode().iloc[0]
print(f"interim rows           : {len(interim)}")
print(f"native cadence (modal) : {native_step}   <- 15-min ENTSO-E DE Actual Load")
print(interim[["Actual Load"]].dropna().head(4))

# %% Cell 8
# New day-ahead exogenous side-tables (sf2-safe >= 15.7.0). Each writes its own
# namespaced interim file (interim/renewable_forecast.csv, interim/day_ahead_price.csv);
# the `Actual Load` / `Forecasted Load` schema in interim/energy_load.csv is untouched.
# These feed the ENTSO-E renewable/net-load/price providers enabled in @sec-team4-config.
# They are day-ahead values (published D-1), so they are leakage-clean at forecast time.

from spotforecast2_safe.downloader.entsoe import download_renewable_forecast
_renewable_csv = get_data_home() / "interim" / "renewable_forecast.csv"
if not _renewable_csv.exists():
    assert api_key, "ENTSOE_API_KEY must be set to download the renewable forecast."
    download_renewable_forecast(
        api_key=api_key, country_code="DE", start=START_DOWNLOAD, end=END_DOWNLOAD, force=True,
        timeout=60,
    )

# %% Cell 9
#| label: team4-renewable-inspect
#| echo: true
renewable_csv = get_data_home() / "interim" / "renewable_forecast.csv"
renewable = pd.read_csv(renewable_csv, index_col=0, parse_dates=True)
renewable.index = pd.to_datetime(renewable.index, utc=True)
print(f"features : {list(renewable.columns)}")
print(f"rows     : {len(renewable):,} (freq {pd.infer_freq(renewable.index[:100])})")
print(f"begin    : {renewable.index.min()}")
print(f"end      : {renewable.index.max()}  (covers the target day y_0)")
print(f"gaps     : {renewable.isna().sum().to_dict()}")

# %% Cell 10
from spotforecast2_safe.downloader.entsoe import download_day_ahead_price
_price_csv = get_data_home() / "interim" / "day_ahead_price.csv"
if not _price_csv.exists():
    assert api_key, "ENTSOE_API_KEY must be set to download the day-ahead price."
    download_day_ahead_price(
        api_key=api_key, country_code="DE_LU", start=START_DOWNLOAD, end=END_DOWNLOAD, force=True,
        timeout=60,
    )

# %% Cell 11
#| label: team4-price-inspect
#| echo: true
price_csv = get_data_home() / "interim" / "day_ahead_price.csv"
price = pd.read_csv(price_csv, index_col=0, parse_dates=True)
price.index = pd.to_datetime(price.index, utc=True)
print(f"features : {list(price.columns)}")
print(f"rows     : {len(price):,} (freq {pd.infer_freq(price.index[:100])} -> {pd.infer_freq(price.index[-100:])})")
print(f"begin    : {price.index.min()}")
print(f"end      : {price.index.max()}  (covers the target day y_0)")
print(f"gaps     : {price.isna().sum().to_dict()}")

# %% Cell 12
#| label: team4-cutoff
#| echo: true
LAST_TARGET = TOMORROW_UTC + pd.Timedelta(hours=23)

# Frontier completeness guard: only an hour with all of its quarter-hour
# samples published may anchor the recursion. The expected count per hour is
# derived from the feed's own cadence (15 min for DE -> 4 samples/hour), so
# the guard also works unchanged on an hourly-cadence feed (1 sample/hour).
actual_15min = interim["Actual Load"].dropna()
CADENCE = actual_15min.index.to_series().diff().mode().iloc[0]
SAMPLES_PER_HOUR = int(pd.Timedelta(hours=1) / CADENCE)
samples_by_hour = actual_15min.resample("h").count()
LAST_FULL_HOUR = samples_by_hour[samples_by_hour >= SAMPLES_PER_HOUR].index.max()

FIRST_PRED = LAST_FULL_HOUR + pd.Timedelta(hours=1)
LIVE_N_STEPS = int((LAST_TARGET - FIRST_PRED).total_seconds() // 3600) + 1

# Value-sanity tripwire (2026-06-03/04 incident): complete but physically
# impossible 15-min actuals must not anchor the recursion. Thresholds are the
# clean-data q0.9999 rounded up (intra-hour range 7.3 -> 8 GW, adjacent step
# 4.1 -> 6 GW); the late-March DST transition week can reach ~12.6 GW ranges
# and would need a temporary exemption. Since sf2-safe 16.4.0 the rules live
# in the library and the SAME rules re-check the frame inside prepare_data
# via the target_qc_* knobs (@sec-team4-config) — one implementation, one
# mental model. This chapter runs policy="truncate" (default since
# 2026-06-05): training retracts to the last sound hour and the recursion
# bridges the gap on published day-ahead exog, fabricating nothing. Gate A
# of the incident forensics showed that aborting buys no better data later
# (ENTSO-E never corrects this class); "abort" stays available as the
# conservative alternative — flip the constant below.
from spotforecast2_safe.preprocessing import apply_target_corruption_policy

MAX_INTRAHOUR_RANGE_MW = 8_000
MAX_ADJ_STEP_MW = 6_000
QC_WINDOW_DAYS = 3
TARGET_CORRUPTION_POLICY = "truncate"   # "abort" = conservative alternative

# Diagnostics use the un-dropped series so that diffs across publication
# gaps show as NaN, matching the library's time-adjacency semantics.
qc_recent = interim["Actual Load"].loc[
    actual_15min.index.max() - pd.Timedelta(days=QC_WINDOW_DAYS) :
]
qc_hourly = qc_recent.resample("h")
qc_range = (qc_hourly.max() - qc_hourly.min()).dropna()
qc_step = qc_recent.diff().abs().dropna()

print(f"QC window            : last {QC_WINDOW_DAYS} days of 15-min actuals")
print(f"intra-hour range     : max {qc_range.max():,.0f} MW (limit {MAX_INTRAHOUR_RANGE_MW:,} MW)")
print(f"adjacent-slot step   : max {qc_step.max():,.0f} MW (limit {MAX_ADJ_STEP_MW:,} MW)")

# Preview on a copy — the AUTHORITATIVE policy run happens inside
# prepare_data via the target_qc_* knobs (@sec-team4-config). Under "abort"
# this call raises TargetCorruptionError; under "truncate" it reports what
# prepare_data will cut (WARNING-logged there, Art. 12 record-keeping).
_, qc_report = apply_target_corruption_policy(
    interim[["Actual Load"]],
    targets=["Actual Load"],
    policy=TARGET_CORRUPTION_POLICY,
    range_mw=MAX_INTRAHOUR_RANGE_MW,
    step_mw=MAX_ADJ_STEP_MW,
    window_days=QC_WINDOW_DAYS,
    max_heal_hours=0,
    anchor_zone_hours=168,
    cutoff=None,
    logger=logging.getLogger("team4-qc"),
)
if qc_report.fired:
    LAST_SOUND_HOUR = qc_report.first_flagged_hour - pd.Timedelta(hours=1)
    print(f"value-sanity QC      : {qc_report.n_flagged_hours} corrupt hour(s) in "
          f"{len(qc_report.spans)} span(s) -> policy {qc_report.action!r}")
    print(f"last sound hour      : {LAST_SOUND_HOUR}  (prepare_data retracts "
          f"data_end here and auto-extends predict_size to reach {LAST_TARGET})")
else:
    print(f"value-sanity QC      : PASS (no corruption in the last "
          f"{QC_WINDOW_DAYS} days)")

print(f"feed cadence         : {CADENCE}  ({SAMPLES_PER_HOUR} samples/hour)")
print(f"last published sample: {actual_15min.index.max()}")
print(f"end_train (inclusive): {LAST_FULL_HOUR}  (last complete hour)")
print(f"first forecast step  : {FIRST_PRED}")
print(f"last forecast step   : {LAST_TARGET}")
print(f"predict_size         : {LIVE_N_STEPS}  (slice last 24 for y_0)")

# %% Cell 14
#| label: team4-training-data-print
#| echo: true
train_view = interim.loc[:LAST_FULL_HOUR, ["Actual Load"]].dropna()
# Reuse predictions_entsoe (@sec-team4-entsoe-predictions), sliced to the
# training window. visualize_ts_plotly overlays one trace per dict entry but
# requires the plotted column to exist in every frame, so the forecast is
# renamed onto the shared "Actual Load" column; the dict key supplies its
# legend label.
entsoe_view = predictions_entsoe.loc[:LAST_FULL_HOUR].rename("Actual Load").to_frame()
print(f"training view: {len(train_view)} rows "
      f"| {train_view.index.min()} → {train_view.index.max()}")
print(f"ENTSO-E forecast view: {len(entsoe_view)} rows "
      f"| {entsoe_view.index.min()} → {entsoe_view.index.max()}")

# %% Cell 17
#| label: team4-acf
#| echo: true
import numpy as np
from spotforecast2.stats.autocorrelation import calculate_lag_autocorrelation

# Hourly Actual Load over the training side only (mean of each hour's four
# quarter-hour values), matching the grid prepare_data() will build later.
acf_series = interim.loc[:LAST_FULL_HOUR, "Actual Load"].resample("h").mean().dropna()
acf = calculate_lag_autocorrelation(acf_series, n_lags=200)

# Data-driven key lags. The 95% white-noise band is ±1.96/sqrt(N); a lag is
# "significant" when its correlation falls outside it. We rank by the PARTIAL
# autocorrelation (PACF), which removes the correlation already carried by
# shorter lags and so flags only the lags with *direct* predictive value ---
# the standard Box-Jenkins rule.
conf = 1.96 / np.sqrt(len(acf_series))
significant = acf[acf["partial_autocorrelation_abs"] > conf]
key_lags = sorted(significant.nlargest(8, "partial_autocorrelation_abs")["lag"].astype(int))
if not key_lags:                       # degenerate fallback (e.g. very short series)
    key_lags = [1, 2, 24, 168]

print(f"N = {len(acf_series)}  |  95% significance band = ±{conf:.4f}")
print(f"significant PACF lags (|pacf| > band): {len(significant)} of {len(acf)}")
print(f"key_lags (top {len(key_lags)} by |PACF|, in lag order): {key_lags}")
# %% Cell 18
#| label: team4-factory
#| echo: true
def team4_lgbm_factory(config, *, weight_func=None, target=None):
    """LightGBM recursive forecaster with anchored level windows (>= 72 h).

    Default L2 objective; window features deliberately exclude the 24-h
    scale --- short windows turn into pure self-feedback over the live
    recursion and flatten the forecast (observed 2026-06-05).
    """
    del target
    return ForecasterRecursive(
        estimator=LGBMRegressor(random_state=config.random_state, verbose=-1),
        lags=config.lags_consider[-1],
        window_features=[  # one instance per feature: keeps generated names unique
            RollingFeatures(stats="mean", window_sizes=config.window_size),  # 72 h, as stock
            RollingFeatures(stats="mean", window_sizes=24 * 7),
            RollingFeatures(stats="mean", window_sizes=24 * 30),
        ],
        weight_func=weight_func,
    )

print("team4_lgbm_factory ready: L2 objective, anchored window means (72/168/720 h)")

# %% Cell 19
#| label: team4-config
#| echo: true
TRAIN_SIZE = pd.Timedelta(days=365 * 2)
PREDICT_SIZE = 24
REFIT_SIZE = 7
NUMBER_FOLDS = 10
DELTA_VAL = pd.Timedelta(hours=PREDICT_SIZE * REFIT_SIZE * NUMBER_FOLDS)
# Gap-penalty zone (hours) for weighted imputation, decoupled from the
# factory's rolling-feature windows (72/168/720 h, @sec-team4-factory). A
# narrower zone reduces the "all sample weights zero -> uniform weighting"
# warning on short CV folds.
IMPUTATION_WINDOW_SIZE = 24

ADVANCED_PERIODS = [
    Period(name="daily",     n_periods=12, column="hour",      input_range=(1, 24)),
    Period(name="weekly",    n_periods=7,  column="dayofweek", input_range=(0, 6)),
    Period(name="monthly",   n_periods=12, column="month",     input_range=(1, 12)),
    Period(name="quarterly", n_periods=4,  column="quarter",   input_range=(1, 4)),
    Period(name="yearly",    n_periods=12, column="dayofyear", input_range=(1, 365)),
]

team4_config = ConfigEntsoe(
    country_code="DE",
    data_filename="interim/energy_load.csv",
    targets=["Actual Load"],
    agg_weights=[1.0],
    bounds=None,
    data_loader=entsoe_data_loader,
    test_data_loader=entsoe_test_data_loader,
    forecaster_factory=team4_lgbm_factory,
    periods=ADVANCED_PERIODS,
    lags_consider=key_lags,
    train_size=TRAIN_SIZE,
    end_train_default=LAST_FULL_HOUR.isoformat(),
    delta_val=DELTA_VAL,
    predict_size=LIVE_N_STEPS,
    cv_block_size=PREDICT_SIZE,
    refit_size=REFIT_SIZE,
    number_folds=NUMBER_FOLDS,
    imputation_window_size=IMPUTATION_WINDOW_SIZE,
    n_trials_optuna=15,
    n_trials_spotoptim=100,
    n_initial_spotoptim=50,
    #n_jobs_spotoptim=-1,
    warm_start_lags=True,
    include_weather_windows=True,
    include_holiday_features=True,
    include_holiday_adjacency_features=True,  # Brückentag + day before/after holiday (sf2-safe >= 15.9.0)
    poly_features_degree=2,
    max_poly_features=40,
    state="NW",
    random_state=42,
    on_weather_failure="skip",
    # New exogenous providers (sf2-safe >= 15.7.0), each gated by one flag and
    # appended by the live MultiTask pipeline (@sec-team4-exog). All are
    # day-ahead or static published vintages, hence leakage-clean (CR-3).
    include_entsoe_forecast_load=False,       # ENTSO-E day-ahead Forecasted Load (near-oracle D-1 prior)
    include_entsoe_renewable_forecast=True,  # day-ahead wind + solar generation forecast
    include_entsoe_net_load=True,            # forecasted load - (wind + solar) forecast
    include_entsoe_day_ahead_price=True,     # DE/LU day-ahead spot price
    include_covid_infection_rate=True,       # bundled RKI national 7-day incidence (lockdown-level proxy)
    on_exog_provider_failure="skip",         # degrade gracefully if a side-table is short of the full window
    exog_max_gap_hours=3,                    # heal interior side-table pinholes of up to 3 h (sf2-safe >= 16.1.0)
    exog_max_tail_gap_hours=48,              # hold the unpublishable day-ahead frontier tail (sf2-safe >= 16.2.0)
    exog_provider_window="train",            # validate providers only on [start_train, cov_end], not all history
    # Target-side corruption policy (sf2-safe >= 16.4.0): the value-sanity
    # tripwire's rules (@sec-team4-cutoff), re-checked at 15-min cadence
    # inside prepare_data — this is the authoritative run. With "truncate"
    # (the chapter default since 2026-06-05) a corrupt frontier retracts
    # data_end to the last sound hour and auto-extends predict_size, so the
    # forecast still reaches the target day on sound published history;
    # "abort" (the library default) is the conservative alternative.
    target_qc_range_mw=MAX_INTRAHOUR_RANGE_MW,
    target_qc_step_mw=MAX_ADJ_STEP_MW,
    target_qc_window_days=QC_WINDOW_DAYS,
    target_corruption_policy=TARGET_CORRUPTION_POLICY,
)
team4_config.data_frame_name = "ddmo-live-team4"
print("team4_config ready:", team4_config.data_frame_name,
      "| predict_size=", team4_config.predict_size,
      "| n_trials_spotoptim=", team4_config.n_trials_spotoptim,
      #"| n_jobs_spotoptim=", team4_config.n_jobs_spotoptim,
      "| number_folds=", team4_config.number_folds)

# %% Cell 20
#| label: team4-search-space
#| echo: true
team4_search_space = {
    "estimator__num_leaves": (8, 1024),
    "estimator__max_depth": (3, 32),
    "estimator__learning_rate": (0.0001, 0.3, "log10"),
    # Linear integer range, deliberately NOT (10, 5000, "log10"): SpotOptim
    # < 0.12.7 int-cast log-transformed bounds, so an int+log10 dimension
    # collapsed to the decade exponents {10, 100, 1000, 10000} — four values
    # only, the last of which silently EXCEEDED the declared cap (observed
    # 2026-06-05: the tuner picked 10000 trees from a "5000" bound; fixed
    # upstream in spotoptim 0.12.7). The linear range stays: transparent and
    # optimizer-version independent.
    "estimator__n_estimators": (100, 5000),
    "estimator__bagging_fraction": (0.5, 1.0),
    "estimator__feature_fraction": (0.5, 1.0),
    "estimator__reg_alpha": (0.001, 10.0),
    "estimator__reg_lambda": (0.001, 10.0),
    # Every candidate carries the weekly anchor 167/168: lags beyond the live
    # horizon keep reading observed history during the recursion. Anchor-free
    # stock candidates ("24", "48", ...) are deliberately excluded.
    "lags": [
        "[1, 2, 3, 11, 12, 22, 23, 24, 47, 48, 167, 168]",   # stock 12-lag
        "[1, 2, 11, 12, 23, 24, 167, 168]",                   # stock 8-lag
        "[1, 2, 24, 48, 167, 168]",                           # compact + weekly anchor
        "[1, 2, 23, 24, 47, 48, 167, 168]",                   # cycle neighbours + weekly
        str(sorted(set(key_lags) | {1, 2, 24, 48, 168})),     # PACF picks ∪ canonical family
        "[1, 2, 3, 23, 24, 25, 47, 48, 167, 168, 169, 336]",  # extended + 2-week lag
    ],
}
anchored = all(("167" in c) or ("168" in c) for c in team4_search_space["lags"])
print(f"{len(team4_search_space['lags'])} lag candidates "
      f"(+ key_lags via warm start) | all weekly-anchored: {anchored}")

# %% Cell 22
#| label: team4-prepare
#| echo: true
team4_mt = MultiTask(team4_config, task="spotoptim")
team4_mt.prepare_data()

endo = team4_mt.df_pipeline
print(f"df_pipeline shape: {endo.shape}")
print(f"index range      : {endo.index.min()} → {endo.index.max()}")
print(f"cadence          : {endo.index.to_series().diff().dropna().mode().iloc[0]}  (hourly mean of the 15-min actuals)")
print(endo["Actual Load"].describe())

# %% Cell 25
#| label: team4-outlier-bounds
#| echo: true
OUTLIER_IQR_K = 5

s = team4_mt.df_pipeline["Actual Load"].dropna()
med = s.median()
iqr = s.quantile(0.75) - s.quantile(0.25)
low = max(0.0, med - OUTLIER_IQR_K * iqr)
high = med + OUTLIER_IQR_K * iqr
team4_mt.config.bounds = [(low, high)]

n_outside = int(((s < low) | (s > high)).sum())
print(f"median             : {med:,.0f} MW")
print(f"IQR                : {iqr:,.0f} MW")
print(f"derived bounds (K={OUTLIER_IQR_K}): [{low:,.0f}, {high:,.0f}] MW")
print(f"history points out : {n_outside} of {len(s)}")

# %% Cell 26
#| label: team4-detect-outliers
#| echo: true
team4_mt.detect_outliers()

n_removed = int(
    (team4_mt.df_pipeline["Actual Load"].isna()
     & team4_mt.df_pipeline_original["Actual Load"].notna()).sum()
)
nan_to_impute = int(team4_mt.df_pipeline["Actual Load"].isna().sum())
imputed_idx = team4_mt.df_pipeline.index[team4_mt.df_pipeline["Actual Load"].isna()]
# Split genuine interior gaps from the unpublished trailing tail (@sec-team4-availability):
# anything at or before end_train (LAST_FULL_HOUR) is a real gap; anything after it is
# simply the recent tail ENTSO-E has not published yet, which end_train excludes from y_train.
interior_idx = imputed_idx[imputed_idx <= LAST_FULL_HOUR]
tail_idx = imputed_idx[imputed_idx > LAST_FULL_HOUR]
contamination = getattr(team4_mt.config, "contamination", None)

print(f"outliers removed   : {n_removed}")
print(f"NaN to impute next : {nan_to_impute}")
print(f"  interior gaps    : {len(interior_idx)}  (real gaps in observed range)")
print(f"  trailing tail    : {len(tail_idx)}  (unpublished, after end_train)")
print(f"contamination      : {contamination}")

# %% Cell 28
#| label: team4-impute
#| echo: true
team4_mt.impute()

nan_after = int(team4_mt.df_pipeline["Actual Load"].isna().sum())
print(f"NaN before impute: {nan_to_impute}")
print(f"NaN after impute : {nan_after}  (must be 0)")

# %% Cell 30
#| label: team4-build-exog
#| echo: true
team4_mt.build_exogenous_features()

exog_all = team4_mt.exogenous_features
# print the full list of exogenous features for manual inspection, then count by family
print("exogenous features:")
for col in exog_all.columns:
    print(f"  {col}")
    
print(f"exogenous_features shape : {exog_all.shape}")
print(f"selected for training    : {len(team4_mt.exog_feature_names)} features")

def family_of(col):
    c = col.lower()
    if c.startswith("covid"):
        return "covid"
    if c.startswith("entsoe_"):
        return "entsoe_provider"
    if "holiday" in c or "brueckentag" in c:
        return "holiday"
    if "poly" in c:
        return "polynomial"
    if "window" in c:
        return "weather_window"
    if any(k in c for k in ("sin", "cos", "rbf")):
        return "cyclical/RBF"
    if c.startswith("lag_") or c.startswith("lag"):
        return "lag"
    return "weather/other"

families = {}
for col in exog_all.columns:
    families.setdefault(family_of(col), []).append(col)

counts = pd.DataFrame(
    sorted(((fam, len(cols)) for fam, cols in families.items()),
           key=lambda kv: kv[1], reverse=True),
    columns=["family", "count"],
)
print(counts.to_string(index=False))

# %% Cell 31
#| label: team4-weather-window
#| echo: true
archive_cutoff = NOW_UTC - pd.Timedelta(days=5)
past_days = (NOW_UTC.normalize() - archive_cutoff.normalize()).days + 1

print(f"now (render)       : {NOW_UTC}")
print(f"archive reaches    : {archive_cutoff}   (Open-Meteo archive lags ~5 days)")
print(f"forecast past_days : {past_days}         (bridges the 5-day archive lag)")
print(f"ffill tolerance    : 24 h                (longer gaps are refused, not filled)")

# %% Cell 36
#| label: team4-train
#| echo: true
team4_mt.run_task_spotoptim(search_space=team4_search_space, show=False)

future = team4_mt.results["spotoptim"]["Actual Load"]["future_pred"]
print(f"raw forecast: {len(future)} hourly steps "
      f"| {future.index.min()} → {future.index.max()}")

y0 = future.loc[TOMORROW_UTC:LAST_TARGET]
print(f"y_0 slice   : {len(y0)} hourly steps "
      f"| {y0.index.min()} → {y0.index.max()}")

# %% Cell 38
#| label: team4-load-model
#| echo: true
fc = team4_mt.results["spotoptim"]["Actual Load"]["forecaster"]

X_tr, y_int = fc.create_train_X_y(
    y=team4_mt.data_with_exog["Actual Load"],
    exog=team4_mt.data_with_exog[team4_mt.exog_feature_names],
)
assert list(X_tr.columns) == list(fc.estimator.feature_name_), \
    "design-matrix columns do not match the fitted estimator's feature names"

feat_names = list(fc.estimator.feature_name_)
print(f"trained features: {len(feat_names)}")

# Data-governance check (CR-3). team_4 now *uses* ENTSO-E's day-ahead priors
# as features --- the provider columns `entsoe_forecasted_load`,
# `entsoe_wind_forecast`, `entsoe_solar_forecast`, `entsoe_net_load`,
# `entsoe_day_ahead_price` (plus the static `covid_infection_rate`). Those are
# admissible: each is published on D-1 (or, for COVID, a static vintage), so it
# is available at forecast time and cannot leak the target day. What must NEVER
# enter the model is a *realised* quantity --- the raw interim `Actual Load` /
# `Forecasted Load` columns (the model sees only the aligned day-ahead provider
# copies and the lagged history) or the legacy realised `Actual`. Assert that,
# fail-loud, so a future regression breaks the render rather than biasing the
# model.
RAW_INTERIM_COLS = {"Actual Load", "Forecasted Load", "Actual"}
assert not (RAW_INTERIM_COLS & set(team4_mt.exog_feature_names)), \
    "a raw interim column leaked into the selected exogenous features!"
assert not (RAW_INTERIM_COLS & set(feat_names)), \
    "a raw interim column leaked into the fitted model's features!"

# The day-ahead / static priors the model is allowed to use (report-only: with
# on_exog_provider_failure='skip' a provider whose side-table is still gappy on
# the validated window after bounded healing is dropped, so this set lists the
# ones that actually made it in).
day_ahead_priors = sorted(
    c for c in feat_names if c.startswith("entsoe_") or c == "covid_infection_rate"
)
print(f"day-ahead / static priors used as features: {day_ahead_priors}")
print("data-governance check passed: priors are day-ahead/static; "
      "no realised target-day column reached the model")

fam_counts = {}
for col in feat_names:
    fam_counts[family_of(col)] = fam_counts.get(family_of(col), 0) + 1
for fam, n in sorted(fam_counts.items(), key=lambda kv: kv[1], reverse=True):
    print(f"  {fam:<16}: {n}")

# %% Cell 42
#| label: team4-vs-entsoe
#| echo: true
# Reuse predictions_entsoe (@sec-team4-entsoe-predictions) --- already on the
# hourly grid --- restricted to y_0's 24-hour window.
entsoe_fc = predictions_entsoe.reindex(y0.index)
overlap = entsoe_fc.dropna().index
print(f"ENTSO-E day-ahead hours available on target day: {len(overlap)} / {len(y0)}")
if len(overlap) > 0:
    mad = float((y0.loc[overlap] - entsoe_fc.loc[overlap]).abs().mean())
    print(f"mean |team_4 - ENTSO-E| over the overlap: {mad:,.0f} MW")
else:
    print("ENTSO-E has not published a day-ahead forecast for the target day "
          "in the local cache yet; showing team_4 only.")
# %% Cell 43
#| label: team4-shape-check
#| echo: true
SHAPE_MIN_CORR = 0.6     # profile agreement: curves rise and fall together
SHAPE_MIN_RANGE = 0.5    # amplitude: forecast range >= half the reference range

reference, ref_name = entsoe_fc.dropna(), "ENTSO-E day-ahead"
if len(reference) < 12:  # fallback: same weekday one week earlier
    week_ago = interim["Actual Load"].resample("h").mean().reindex(
        y0.index - pd.Timedelta(hours=168))
    week_ago.index = week_ago.index + pd.Timedelta(hours=168)
    reference, ref_name = week_ago.dropna(), "actuals one week earlier"

if len(reference) < 12:
    print("shape check skipped: no reference profile available in the cache.")
else:
    common = y0.index.intersection(reference.index)
    r = float(y0.loc[common].corr(reference.loc[common]))
    ref_range = float(reference.loc[common].max() - reference.loc[common].min())
    range_ratio = float((y0.max() - y0.min()) / ref_range) if ref_range > 0 else float("nan")
    print(f"shape check vs. {ref_name} ({len(common)} h): "
          f"r = {r:.2f} (min {SHAPE_MIN_CORR}), "
          f"range ratio = {range_ratio:.2f} (min {SHAPE_MIN_RANGE})")
    if r < SHAPE_MIN_CORR or range_ratio < SHAPE_MIN_RANGE:
        print("=" * 72)
        print("WARNING: forecast shape implausible -- the daily profile does not")
        print(f"track the {ref_name} reference. Inspect @sec-team4-vs-entsoe")
        print("before pushing this submission (warn-only by design, Art. 14).")
        print("=" * 72)
    else:
        print("shape check passed: daily profile and amplitude look plausible.")

# %% Cell 44
#| label: team4-submission-write
#| echo: true
assert len(y0) == 24,           f"expected 24 hourly steps for y_0, got {len(y0)}"
assert y0.index.min() == TOMORROW_UTC, (
    f"first step {y0.index.min()} != TOMORROW {TOMORROW_UTC}"
)
assert (y0 > 0).all(),          "non-positive forecast value --- spec requires > 0"
assert y0.notna().all(),        "NaN in forecast --- spec forbids"

# Team identity follows the feature set (@sec-team4-identity): with ENTSO-E's
# day-ahead Forecasted Load as a model input, submit as team_4_entsoe;
# without it, as team_4.
TEAM_ID = "a_team_entsoe" if team4_config.include_entsoe_forecast_load else "a_team"
print(f"TEAM_ID = {TEAM_ID}  "
      f"(include_entsoe_forecast_load={team4_config.include_entsoe_forecast_load})")
# Write the submission next to this script (./submissions/<team>/<date>.csv);
# set SUBMISSION_ROOT to a challenge-leaderboard clone to write there instead.
_sub_root = Path(os.environ.get("SUBMISSION_ROOT", str(_HERE)))
SUBMISSION_DIR = _sub_root / "submissions" / TEAM_ID
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
submission_path = SUBMISSION_DIR / f"{TOMORROW_UTC.date().isoformat()}.csv"

submission_df = pd.DataFrame({
    "timestamp_utc": y0.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "forecast_mw":   y0.round(2).values,
})
submission_df.to_csv(submission_path, index=False)
print(f"wrote {submission_path}")
print(f"  rows: {len(submission_df)}")
print(f"  range: {submission_df.forecast_mw.min():.1f}–{submission_df.forecast_mw.max():.1f} MW")
submission_df.head(3)

# %% Cell 45
#| label: team4-validate
#| echo: true
import subprocess
_lb_dir = os.environ.get("CHALLENGE_LEADERBOARD_DIR")
lb_repo = Path(_lb_dir) if _lb_dir else None
if (lb_repo is not None
        and (lb_repo / "scripts" / "validate_submission.py").exists()
        and submission_path.is_relative_to(lb_repo)):
    result = subprocess.run(
        [
            "uv", "run", "python", "scripts/validate_submission.py",
            "--path", str(submission_path.relative_to(lb_repo)),
            "--skip-deadline",
        ],
        cwd=lb_repo,
        capture_output=True,
        text=True,
    )
    print("exit code:", result.returncode)
    print("stdout   :", result.stdout.strip())
    if result.stderr.strip():
        print("stderr   :", result.stderr.strip())
else:
    print("leaderboard validator not run (set CHALLENGE_LEADERBOARD_DIR and write "
          "the CSV inside that clone to enable schema validation).")