"""
Sicherheitskritische Zeitreihenprognose – vollständige Pipeline
Basiert auf den Vorlesungsnotebooks 01–10
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURATION  (CR-2: alle Konstanten fixiert – keine "jetzt"-Werte)
# ─────────────────────────────────────────────────────────────────────────────

DATA_START   = "202401010000"                            # 2024-01-01 00:00 UTC
DATA_END     = "202412312300"                            # 2024-12-31 23:00 UTC
COUNTRY      = "DE"
TRAIN_END    = pd.Timestamp("2024-10-31 23:00", tz="UTC")
EVAL_END     = pd.Timestamp("2024-11-30 23:00", tz="UTC")
RANDOM_STATE = 2026

SCRIPT_DIR = Path(__file__).parent.resolve()
CACHE      = SCRIPT_DIR / "_cache"
SHARED     = CACHE / "shared"
DATA_HOME  = CACHE / "spotforecast2_data"

for _d in [CACHE, SHARED, DATA_HOME]:
    _d.mkdir(parents=True, exist_ok=True)

# Muss gesetzt sein BEVOR spotforecast2_safe geladen wird
os.environ["SPOTFORECAST2_DATA"] = str(DATA_HOME)


# ─────────────────────────────────────────────────────────────────────────────
# KAPITEL 01 – AUDIT-PROTOKOLL
# ─────────────────────────────────────────────────────────────────────────────

from spotforecast2_safe.manager.logger import setup_logging

log_dir = CACHE / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logger, log_path = setup_logging(level=logging.INFO, log_dir=log_dir)
logger.info("Vorlesungs-Lauf beginnt: Lastprognose-Pipeline (Kapitel 01).")

print(f"[01] Log-Datei: {log_path.name if log_path else '<nur stdout>'}")

if log_path is not None and log_path.exists():
    print("--- erste Zeile des Protokolls ---")
    print(log_path.read_text().splitlines()[0])


# ─────────────────────────────────────────────────────────────────────────────
# KAPITEL 02 – ENTSO-E DATEN
# ─────────────────────────────────────────────────────────────────────────────

from spotforecast2_safe.downloader.entsoe import download_new_data
from spotforecast2_safe.data.fetch_data import fetch_data, get_data_home

print("\n[02] Lade Lastdaten von ENTSO-E …")

interim = get_data_home() / "interim" / "energy_load.csv"

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "948c9100-84da-4aac-b891-f3f8f4a4de46")

if interim.exists():
    print(f"    Cache gefunden: {interim}")
else:
    logging.getLogger("spotforecast2_safe.downloader.entsoe").setLevel(logging.INFO)
    download_new_data(
        api_key=ENTSOE_API_KEY,
        country_code=COUNTRY,
        start=DATA_START,
        end=DATA_END,
        force=True,
    )

df = fetch_data(filename=str(interim))
df.index = pd.to_datetime(df.index, utc=True)
load_col = next(c for c in df.columns if "Actual" in c and "Load" in c)
y = df[load_col].astype(float).rename("load")

if y.index.inferred_freq != "h":
    y = y.resample("h").mean()

print(f"    Anzahl Zeitstempel: {len(y):,}")
print(f"    Fehlende Werte:     {int(y.isna().sum())}")
print(f"    Zeitfenster: {y.index[0].date()} bis {y.index[-1].date()}")

# Plot 1: Jahresübersicht
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(y.index, y.values, linewidth=0.5, color="steelblue")
axes[0].set_title("Jahresübersicht: Stromlast Deutschland 2024")
axes[0].set_ylabel("Last [MW]")
axes[0].grid(True, alpha=0.3)

# Plot 2: Beispielwoche
woche = y.loc["2024-02-05":"2024-02-12"]
axes[1].plot(woche.index, woche.values, color="darkorange")
axes[1].set_title("Beispielwoche: 5.–12. Februar 2024")
axes[1].set_ylabel("Last [MW]")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "02_lastdaten.png", dpi=120)
plt.close()
print("    Plot gespeichert: 02_lastdaten.png")

y.to_frame().to_parquet(SHARED / "load_de_2024.parquet")
logger.info("Kapitel 02: Lastdaten geladen und gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# KAPITEL 03 – DATENLÜCKEN
# ─────────────────────────────────────────────────────────────────────────────

from spotforecast2_safe import LinearlyInterpolateTS

print("\n[03] Datenlücken behandeln …")
print(f"    {len(y):,} Stunden geladen, {int(y.isna().sum())} NaN.")

# Künstliche Lücke für Demo-Zwecke
y_with_gap = y.copy()
y_with_gap.iloc[120:123] = np.nan
print(f"    Künstliche NaNs eingefügt: {int(y_with_gap.isna().sum())}")

# Modus 'raise' (Standard) – interpoliert Mittellücken, bricht bei Rändern ab
y_clean = LinearlyInterpolateTS().fit_transform(y_with_gap)
print(f"    NaN nach Interpolation (raise): {int(y_clean.isna().sum())}")

# Demo: Randlücke → CR-3 Fail-safe-Abbruch
y_edge = y.copy()
y_edge.iloc[0:2] = np.nan
try:
    LinearlyInterpolateTS(on_missing="raise").fit_transform(y_edge)
except ValueError as exc:
    print(f"    Erwarteter Fail-safe-Abbruch (Randlücke): {str(exc)[:80]}…")

# Modus 'ffill_bfill'
y_filled = LinearlyInterpolateTS(on_missing="ffill_bfill").fit_transform(y_edge)
print(f"    NaN nach 'ffill_bfill': {int(y_filled.isna().sum())}")

# Modus 'passthrough'
y_pt = LinearlyInterpolateTS(on_missing="passthrough").fit_transform(y_edge)
print(f"    NaN nach 'passthrough': {int(y_pt.isna().sum())}")

# Plot: Vor / Nach Interpolation
region = slice(118, 126)
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(y_with_gap.iloc[region].index, y_with_gap.iloc[region].values,
             "o-", color="tomato", label="Mit Lücke")
axes[0].set_title("Vor Interpolation (künstliche Lücke bei Index 120–122)")
axes[0].set_ylabel("Last [MW]")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(y_clean.iloc[region].index, y_clean.iloc[region].values,
             "o-", color="seagreen", label="Bereinigt")
axes[1].set_title("Nach LinearlyInterpolateTS (on_missing='raise')")
axes[1].set_ylabel("Last [MW]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "03_datenluecken.png", dpi=120)
plt.close()
print("    Plot gespeichert: 03_datenluecken.png")

y_clean.to_frame("load").to_parquet(SHARED / "load_de_2024_clean.parquet")
logger.info("Kapitel 03: Datenlücken behandelt, bereinigter Datensatz gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# KAPITEL 04 – TRAIN / TEST-SPLIT
# ─────────────────────────────────────────────────────────────────────────────

print("\n[04] Chronologischer Train/Test-Split …")

y_train = y_clean.loc[:TRAIN_END]
y_eval  = y_clean.loc[TRAIN_END + pd.Timedelta(hours=1) : EVAL_END]

print(f"    Training:   {y_train.index[0].date()} – {y_train.index[-1].date()} ({len(y_train):,} h)")
print(f"    Evaluation: {y_eval.index[0].date()} – {y_eval.index[-1].date()} ({len(y_eval):,} h)")

# Plot: Split-Übersicht
y_reserve = y_clean.loc[EVAL_END + pd.Timedelta(hours=1):]
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(y_train.index,   y_train.values,   color="steelblue",  lw=0.5, label="Training (Jan–Okt)")
ax.plot(y_eval.index,    y_eval.values,    color="darkorange", lw=0.5, label="Evaluation (Nov)")
ax.plot(y_reserve.index, y_reserve.values, color="gray",       lw=0.5, label="Reserve (Dez)")
ax.axvline(TRAIN_END, color="black", linestyle="--", lw=1)
ax.axvline(EVAL_END,  color="black", linestyle="--", lw=1)
ax.set_title("Chronologischer Train / Eval / Reserve-Split 2024")
ax.set_ylabel("Last [MW]")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "04_split.png", dpi=120)
plt.close()
print("    Plot gespeichert: 04_split.png")

y_train.to_frame("load").to_parquet(SHARED / "y_train.parquet")
y_eval.to_frame("load").to_parquet(SHARED / "y_eval.parquet")
(SHARED / "split_meta.json").write_text(json.dumps({
    "train_end": TRAIN_END.isoformat(),
    "eval_end":  EVAL_END.isoformat(),
    "freq":      "h",
    "tz":        "UTC",
}, indent=2))
logger.info("Kapitel 04: Train/Eval-Split gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# KAPITEL 05 – KALENDER UND WETTER
# ─────────────────────────────────────────────────────────────────────────────

from spotforecast2_safe import Period, ExogBuilder
from spotforecast2_safe.weather import WeatherService

print("\n[05] Kalender- und Wetter-Features …")

periods = [
    Period(name="hour",      n_periods=12, column="hour",      input_range=(0, 23)),
    Period(name="dayofweek", n_periods=4,  column="dayofweek", input_range=(0,  6)),
    Period(name="month",     n_periods=12, column="month",     input_range=(1, 12)),
]

builder  = ExogBuilder(periods=periods, country_code=COUNTRY)
exog_cal = builder.build(y_train.index.min(), y_eval.index.max())
print(f"    Kalender-Merkmale: {exog_cal.shape[0]:,} Stunden, {exog_cal.shape[1]} Spalten")

# Plot: RBF-Kodierung für Tagesstunden
hour_cols = [c for c in exog_cal.columns if c.startswith("hour_")]
fig, ax = plt.subplots(figsize=(12, 4))
x = np.arange(24)
for col in hour_cols:
    ax.plot(x, exog_cal[col].iloc[:24].values, alpha=0.7)
ax.set_title("Zyklische Kodierung: 12 RBF-Funktionen für Tagesstunden")
ax.set_xlabel("Stunde des Tages")
ax.set_ylabel("RBF-Aktivierung")
ax.set_xticks(range(24))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "05_rbf.png", dpi=120)
plt.close()
print("    Plot gespeichert: 05_rbf.png")

# Wetterdaten (Open-Meteo, Frankfurt am Main)
weather_cache = CACHE / "weather_de_2024.parquet"
ws = WeatherService(
    latitude=50.110924,
    longitude=8.682127,
    cache_path=weather_cache,
    use_forecast=False,
)
weather = ws.get_dataframe(
    start=y_train.index.min(),
    end=y_eval.index.max(),
    freq="h",
    fill_missing=False,           # CR-3: kein stilles Auffüllen
)
print(f"    Wetter-Frame: {weather.shape[0]:,} Stunden, {weather.shape[1]} Spalten")
print(f"    NaN-Werte:    {int(weather.isna().sum().sum())} (sollte 0 sein)")

# Plot: Last und Temperatur (Mitte März)
temp_col = next((c for c in weather.columns if "temperature" in c.lower()), weather.columns[0])
maerz = slice("2024-03-11", "2024-03-18")
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(y_train.loc[maerz].index, y_train.loc[maerz].values, color="steelblue")
axes[0].set_ylabel("Last [MW]")
axes[0].set_title("Mitte März 2024: Stromlast und Lufttemperatur")
axes[0].grid(True, alpha=0.3)
axes[1].plot(weather.loc[maerz].index, weather.loc[maerz, temp_col].values, color="tomato")
axes[1].set_ylabel("Temperatur [°C]")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "05_wetter_last.png", dpi=120)
plt.close()
print("    Plot gespeichert: 05_wetter_last.png")

# Merkmalsmatrizen zusammenführen
exog = exog_cal.join(weather, how="inner")
exog = exog.loc[y_train.index.min() : y_eval.index.max()]
print(f"    Vereinigte Merkmals-Matrix: {exog.shape[0]:,} Stunden, {exog.shape[1]} Spalten")
print(f"    NaN-Werte: {int(exog.isna().sum().sum())}")

exog.to_parquet(SHARED / "exog.parquet")
logger.info("Kapitel 05: Kalender + Wetter-Features erstellt und gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# KAPITEL 06 – FORECASTER TRAINING
# ─────────────────────────────────────────────────────────────────────────────

from spotforecast2_safe import ForecasterRecursiveLGBM
from lightgbm import LGBMRegressor

print("\n[06] Forecaster trainieren …")

model = ForecasterRecursiveLGBM(
    iteration=0,
    lags=168,              # 7 Tage × 24 Stunden
    periods=periods,
    country_code=COUNTRY,
    random_state=RANDOM_STATE,
)

# CR-2: drei LightGBM-Optionen für bitweise Reproduzierbarkeit
model.forecaster.estimator = LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    n_jobs=-1,
    verbose=-1,
    deterministic=True,            # CR-2
    force_col_wise=True,           # CR-2
    random_state=RANDOM_STATE,     # CR-2
)

print(f"    Schätzer:     {type(model.forecaster.estimator).__name__}")
print(f"    Anzahl Bäume: {model.forecaster.estimator.n_estimators}")
print(f"    Lags:         {model.forecaster.lags.size}")
print(f"    Seed:         {model.random_state}")

t0 = time.perf_counter()
model.fit(y=y_train, exog=exog.loc[y_train.index])
elapsed = time.perf_counter() - t0
print(f"    Training abgeschlossen in {elapsed:.1f} Sekunden.")
print(f"    Trainings-Index: {model.forecaster.training_range_[0]} … {model.forecaster.training_range_[1]}")

# Feature-Importance-Plot
try:
    importance = model.forecaster.get_feature_importances()
    top15 = importance.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top15["feature"].tolist()[::-1], top15["importance"].tolist()[::-1], color="steelblue")
    ax.set_title("Feature Importance (Top 15, Split-Anzahl)")
    ax.set_xlabel("Splits")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "06_feature_importance.png", dpi=120)
    plt.close()
    print("    Plot gespeichert: 06_feature_importance.png")
except Exception as exc:
    print(f"    Feature-Importance-Plot übersprungen: {exc}")

model_path = CACHE / "forecaster.pkl"
with model_path.open("wb") as fh:
    pickle.dump(model, fh)
print(f"    Modell gespeichert: {model_path}")
logger.info("Kapitel 06: Modell trainiert und gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# KAPITEL 07 – VORHERSAGE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[07] 24-Stunden-Vorhersage erzeugen …")

y_eval_24h  = y_eval.iloc[:24]
last_window = y_train.iloc[-168:].copy()
last_window.index = pd.DatetimeIndex(last_window.index, freq="h")

y_pred = model.forecaster.predict(
    steps=24,
    last_window=last_window,
    exog=exog.loc[y_eval_24h.index],
)
y_pred.name  = "forecast"
y_pred.index = y_eval_24h.index
print(f"    Vorhersage erzeugt: {len(y_pred)} Stunden")

# Wochen-Persistenz-Baseline
y_baseline = y_clean.shift(freq=pd.Timedelta(hours=168)).loc[y_eval_24h.index]

# Plot: Vorhersage vs. Tatsächlich vs. Baseline
kontext = y_train.iloc[-24:]
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(kontext.index,     kontext.values,       color="steelblue",  lw=1.5, label="Training (letzte 24 h)")
ax.plot(y_eval_24h.index,  y_eval_24h.values,    color="steelblue",  lw=1.5, linestyle=":", label="Tatsächlich")
ax.plot(y_pred.index,      y_pred.values,        color="seagreen",   lw=2,   label="Modell-Vorhersage")
ax.plot(y_baseline.index,  y_baseline.values,    color="purple",     lw=1.5, linestyle="--", label="Wochen-Persistenz")
ax.axvline(TRAIN_END + pd.Timedelta(hours=1), color="black", linestyle="--", lw=1)
ax.set_title("Vorhersage vs. Tatsächlich vs. Wochen-Persistenz (1. November 2024)")
ax.set_ylabel("Last [MW]")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "07_vorhersage.png", dpi=120)
plt.close()
print("    Plot gespeichert: 07_vorhersage.png")

y_pred.to_frame("forecast").to_parquet(SHARED / "y_pred_24h.parquet")
logger.info("Kapitel 07: 24-h-Vorhersage erzeugt und gespeichert.")


# ─────────────────────────────────────────────────────────────────────────────
# KAPITEL 08 – GENAUIGKEIT
# ─────────────────────────────────────────────────────────────────────────────

from spotforecast2_safe.manager.metrics import calculate_metrics

print("\n[08] Genauigkeitsmetriken …")

m    = calculate_metrics(actual=y_eval_24h, predicted=y_pred)
mae  = float(m["MAE"])
mse  = float(m["MSE"])
rmse = float(np.sqrt(mse))
mape = float(np.mean(np.abs((y_eval_24h.values - y_pred.values) / y_eval_24h.values)))

print(f"    MAE  = {mae:8.1f}   MW   (mittlere absolute Abweichung)")
print(f"    MSE  = {mse:10.1f}   MW²  (mittlere quadrierte Abweichung)")
print(f"    RMSE = {rmse:8.1f}   MW   (Wurzel aus MSE)")
print(f"    MAPE = {mape:8.2%}        (mittlere prozentuale Abweichung)")

# Baseline-Vergleich
mae_p  = float(np.abs(y_eval_24h.values - y_baseline.values).mean())
mape_p = float(np.mean(np.abs((y_eval_24h.values - y_baseline.values) / y_eval_24h.values)))
print(f"\n    {'':24} Modell       Wochen-Persistenz")
print(f"    {'MAE':24} {mae:8.1f} MW   {mae_p:8.1f} MW")
print(f"    {'MAPE':24} {mape:8.2%}     {mape_p:8.2%}")

# Plot: Residuenverteilung
residuen = y_eval_24h.values - y_pred.values
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(residuen, bins=12, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline( mae,  color="tomato",  linestyle="--", lw=2, label=f"MAE  = {mae:.0f} MW")
ax.axvline(-mae,  color="tomato",  linestyle="--", lw=2)
ax.axvline( rmse, color="orange",  linestyle="--", lw=2, label=f"RMSE = {rmse:.0f} MW")
ax.axvline(-rmse, color="orange",  linestyle="--", lw=2)
ax.set_title("Residuenverteilung der 24-h-Vorhersage")
ax.set_xlabel("Fehler [MW]")
ax.set_ylabel("Häufigkeit")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "08_metriken.png", dpi=120)
plt.close()
print("    Plot gespeichert: 08_metriken.png")
logger.info(f"Kapitel 08: MAE={mae:.1f} MW, RMSE={rmse:.1f} MW, MAPE={mape:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# KAPITEL 09 – BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

from spotforecast2_safe.model_selection import TimeSeriesFold, backtesting_forecaster

print("\n[09] Rolling-Origin-Backtest …")

# freq-Attribut muss explizit gesetzt sein (geht beim Parquet-Laden verloren)
y_bt = y_clean.loc[exog.index.min() : exog.index.max()].copy()
y_bt.index    = pd.DatetimeIndex(y_bt.index, freq="h")
exog_bt       = exog.copy()
exog_bt.index = pd.DatetimeIndex(exog_bt.index, freq="h")

cv = TimeSeriesFold(
    steps=24,
    initial_train_size=len(y_train),
    refit=False,             # selbes Modell über alle Folds (Produktionssimulation)
    fixed_train_size=False,
    verbose=False,
)

metrics_df, predictions_df = backtesting_forecaster(
    forecaster=model.forecaster,
    y=y_bt,
    cv=cv,
    metric="mean_absolute_error",
    exog=exog_bt,
    show_progress=False,
    verbose=False,
)

n_folds  = len(metrics_df)
mean_mae = float(metrics_df.iloc[0, 0])
n_preds  = len(predictions_df)
print(f"    Anzahl Folds:  {n_folds}")
print(f"    Mittlerer MAE: {mean_mae:.1f} MW")
print(f"    Vorhersagen:   {n_preds:,} h")

# Per-Fold-MAE aus Prognosen berechnen
pred_col   = predictions_df.columns[0]
actuals    = y_bt.loc[predictions_df.index]
fold_mae   = predictions_df[pred_col].sub(actuals).abs().groupby(predictions_df.index.date).mean()

# Plot: Zeitreihe + Fehlerverteilung
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(y_bt.loc["2024-11-01":].index, y_bt.loc["2024-11-01":].values,
             color="gray", lw=0.5, label="Tatsächlich (Nov 2024)")
axes[0].set_title("Backtest: Rolling-Origin-Folds (November 2024)")
axes[0].set_ylabel("Last [MW]")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(fold_mae.values, bins=15, color="darkorange", edgecolor="white", alpha=0.8)
axes[1].axvline(fold_mae.median(), color="black", linestyle="--", lw=2,
                label=f"Median MAE = {fold_mae.median():.0f} MW")
axes[1].set_title("MAE-Verteilung über alle Backtest-Folds")
axes[1].set_xlabel("MAE [MW]")
axes[1].set_ylabel("Häufigkeit")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "09_backtest.png", dpi=120)
plt.close()
print("    Plot gespeichert: 09_backtest.png")
logger.info(f"Kapitel 09: Backtest abgeschlossen, mittlerer MAE = {mean_mae:.1f} MW")


# ─────────────────────────────────────────────────────────────────────────────
# KAPITEL 10 – PROVENIENZ
# ─────────────────────────────────────────────────────────────────────────────

print("\n[10] Provenienz-Dokumentation …")

provenance: dict = {
    "schaetzer":              type(model.forecaster.estimator).__name__,
    "anzahl_baeume":          int(model.forecaster.estimator.n_estimators),
    "lernrate":               float(model.forecaster.estimator.learning_rate),
    "lags":                   int(model.forecaster.lags.size),
    "seed":                   int(model.random_state),
    "trainings_index_von":    str(model.forecaster.training_range_[0]),
    "trainings_index_bis":    str(model.forecaster.training_range_[1]),
    "land_kalender":          COUNTRY,
    "anzahl_perioden":        len(periods),
    "exog_spalten_anzahl":    int(exog.shape[1]),
    "mae_eval_24h_mw":        round(mae,      1),
    "rmse_eval_24h_mw":       round(rmse,     1),
    "mape_eval_24h":          round(mape,     4),
    "backtest_mittlerer_mae": round(mean_mae, 1),
}

# Optionale Felder (abhängig von Modellklasse)
for attr, key in [
    ("name",                        "modell_name"),
    ("preprocessor.country_code",   "preprocessor_country"),
]:
    try:
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part)
        provenance[key] = str(obj)
    except AttributeError:
        pass

try:
    provenance["exog_spalten"] = list(model.forecaster.exog_names_in_ or [])
except AttributeError:
    pass

print(json.dumps(provenance, indent=2, ensure_ascii=False))

# Audit-Log-Zusammenfassung
log_files = sorted(log_dir.glob("*.log")) if log_dir.exists() else []
if log_files:
    latest = log_files[-1]
    lines  = latest.read_text().splitlines()
    print(f"\n    Log-Datei: {latest.name}, {len(lines)} Zeilen")
    print("    Erste fünf Einträge:")
    for line in lines[:5]:
        print(f"      {line}")

out = CACHE / "provenance.json"
out.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
print(f"\n    Provenance gespeichert: {out}")
logger.info("Kapitel 10: Provenienz-Dokumentation abgeschlossen. Pipeline-Lauf beendet.")


# ─────────────────────────────────────────────────────────────────────────────
# FERTIG
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Pipeline erfolgreich abgeschlossen!")
print("=" * 60)
print(f"Plots:      02_lastdaten.png … 09_backtest.png")
print(f"Cache:      {CACHE}")
print(f"Provenance: {out}")
