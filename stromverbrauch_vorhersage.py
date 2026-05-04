"""
Stromverbrauchsvorhersage Deutschland
Datenquelle: ENTSO-E Transparency Platform
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from entsoe import EntsoePandasClient

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────

API_KEY = "948c9100-84da-4aac-b891-f3f8f4a4de46"

LAND = "DE"
ZEITZONE = "Europe/Berlin"

START = pd.Timestamp("2023-01-01", tz=ZEITZONE)
# END = heute, damit das Modell auf aktuellen Daten trainiert wird
END   = pd.Timestamp.now(tz=ZEITZONE).floor("h")


# ─────────────────────────────────────────────
# 1. DATEN LADEN
# ─────────────────────────────────────────────

def lade_lastdaten(api_key: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Lädt stündliche Lastdaten (actual load) für Deutschland von ENTSO-E."""
    client = EntsoePandasClient(api_key=api_key)
    last = client.query_load(LAND, start=start, end=end)
    last = last.resample("h").mean()
    if isinstance(last, pd.DataFrame):
        last = last.sum(axis=1)
    last.name = "last_mw"
    return last


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Lags die für eine 24h-Vorschau alle bekannt sind (≥ 24h in der Vergangenheit)
LAGS = [24, 48, 168]

def erstelle_features(series: pd.Series) -> pd.DataFrame:
    """Erstellt Zeitreihen-Features aus dem Timestamp-Index."""
    df = series.to_frame()
    idx = df.index.tz_convert(ZEITZONE)

    df["stunde"]         = idx.hour
    df["wochentag"]      = idx.dayofweek
    df["monat"]          = idx.month
    df["tag_im_jahr"]    = idx.dayofyear
    df["ist_wochenende"] = (idx.dayofweek >= 5).astype(int)

    # Zyklische Kodierung (verhindert Sprung 23→0 bzw. Dez→Jan)
    df["stunde_sin"] = np.sin(2 * np.pi * df["stunde"] / 24)
    df["stunde_cos"] = np.cos(2 * np.pi * df["stunde"] / 24)
    df["monat_sin"]  = np.sin(2 * np.pi * df["monat"] / 12)
    df["monat_cos"]  = np.cos(2 * np.pi * df["monat"] / 12)

    for lag in LAGS:
        df[f"lag_{lag}h"] = df["last_mw"].shift(lag)

    df["rolling_24h"]  = df["last_mw"].shift(24).rolling(24).mean()
    df["rolling_168h"] = df["last_mw"].shift(168).rolling(168).mean()

    df = df.dropna()
    return df


def baue_morgen_features(last: pd.Series) -> pd.DataFrame:
    """
    Baut den Feature-Vektor für alle 24 Stunden von morgen.
    Alle verwendeten Lags sind ≥ 24h, also vollständig aus historischen Daten bekannt.
    """
    morgen_start = (pd.Timestamp.now(tz=ZEITZONE) + pd.Timedelta(days=1)).normalize()
    morgen_stunden = pd.date_range(morgen_start, periods=24, freq="h", tz=ZEITZONE)

    rows = []
    for ts in morgen_stunden:
        row = {
            "stunde":         ts.hour,
            "wochentag":      ts.dayofweek,
            "monat":          ts.month,
            "tag_im_jahr":    ts.day_of_year,
            "ist_wochenende": int(ts.dayofweek >= 5),
            "stunde_sin":     np.sin(2 * np.pi * ts.hour / 24),
            "stunde_cos":     np.cos(2 * np.pi * ts.hour / 24),
            "monat_sin":      np.sin(2 * np.pi * ts.month / 12),
            "monat_cos":      np.cos(2 * np.pi * ts.month / 12),
        }
        for lag in LAGS:
            vergangenheit = ts - pd.Timedelta(hours=lag)
            # Nächsten bekannten Wert suchen
            idx_utc = vergangenheit.tz_convert("UTC")
            try:
                row[f"lag_{lag}h"] = last.asof(idx_utc)
            except Exception:
                row[f"lag_{lag}h"] = np.nan

        # Rollende Mittelwerte: letzte 24h bzw. 168h, verschoben um 24h/168h
        t24  = ts - pd.Timedelta(hours=24)
        t168 = ts - pd.Timedelta(hours=168)
        row["rolling_24h"]  = last[last.index <= t24.tz_convert("UTC")].iloc[-24:].mean()
        row["rolling_168h"] = last[last.index <= t168.tz_convert("UTC")].iloc[-168:].mean()

        rows.append(row)

    return pd.DataFrame(rows, index=morgen_stunden)


# ─────────────────────────────────────────────
# 3. MODELL TRAINIEREN & EVALUIEREN
# ─────────────────────────────────────────────

def trainiere_modell(df: pd.DataFrame):
    """Trainiert LightGBM auf allen verfügbaren Daten, evaluiert auf letzten 30 Tagen."""
    import lightgbm as lgb

    feature_cols = [c for c in df.columns if c != "last_mw"]
    X, y = df[feature_cols], df["last_mw"]

    split = len(df) - 24 * 30
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    modell = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                num_leaves=63, random_state=42, n_jobs=-1)
    print(f"Training:  {X_train.index[0].date()} – {X_train.index[-1].date()} ({len(X_train)} Stunden)")
    print(f"Test:      {X_test.index[0].date()}  – {X_test.index[-1].date()}  ({len(X_test)} Stunden)")
    modell.fit(X_train, y_train)

    y_pred = modell.predict(X_test)
    mae  = np.mean(np.abs(y_test.values - y_pred))
    rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    print(f"\nErgebnisse (Test-Set, letzte 30 Tage):")
    print(f"  MAE:  {mae:,.0f} MW  |  RMSE: {rmse:,.0f} MW  |  MAPE: {mape:.2f}%")

    # Für die echte Vorhersage: auf ALLEN Daten trainieren
    modell.fit(X, y)
    return modell, feature_cols, X_test, y_test, y_pred


# ─────────────────────────────────────────────
# 4. PLOTS
# ─────────────────────────────────────────────

def plotte_ergebnis(y_test: pd.Series, y_pred: np.ndarray,
                    morgen_idx: pd.DatetimeIndex, morgen_pred: np.ndarray):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Oben: Rückblick (letzte 7 Tage Test-Set)
    n = 7 * 24
    ax = axes[0]
    ax.plot(y_test.index[-n:], y_test.values[-n:], label="Tatsächlich", color="steelblue")
    ax.plot(y_test.index[-n:], y_pred[-n:], label="Modell (Test)", color="tomato", linestyle="--")
    ax.set_title("Rückblick: Modell vs. Realität (letzte 7 Tage des Test-Sets)")
    ax.set_ylabel("Last [MW]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Unten: Vorhersage morgen
    ax = axes[1]
    ax.plot(morgen_idx, morgen_pred, color="darkorange", linewidth=2, marker="o", markersize=3)
    ax.set_title(f"Vorhersage: {morgen_idx[0].date()} (morgen, 24 Stunden)")
    ax.set_ylabel("Last [MW]")
    ax.set_xlabel("Uhrzeit")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M", tz=ZEITZONE))

    plt.tight_layout()
    plt.savefig("vorhersage_ergebnis.png", dpi=150)
    plt.show()
    print("Plot gespeichert: vorhersage_ergebnis.png")


# ─────────────────────────────────────────────
# HAUPTPROGRAMM
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Stromverbrauchsvorhersage Deutschland ===\n")

    print("Lade Lastdaten von ENTSO-E (bis heute)...")
    last = lade_lastdaten(API_KEY, START, END)
    print(f"Geladen: {len(last)} Stunden ({last.index[0].date()} – {last.index[-1].date()})")
    print(f"Verbrauch: min={last.min():.0f} MW, max={last.max():.0f} MW, mean={last.mean():.0f} MW\n")

    print("Erstelle Features...")
    df = erstelle_features(last)
    print(f"Feature-Matrix: {df.shape[0]} Zeilen × {df.shape[1]} Spalten\n")

    print("Trainiere Modell...")
    modell, feature_cols, X_test, y_test, y_pred = trainiere_modell(df)

    print("\nErstelle Vorhersage für morgen...")
    X_morgen = baue_morgen_features(last)
    morgen_pred = modell.predict(X_morgen[feature_cols])

    morgen_idx = X_morgen.index
    print(f"\nVorhersage für {morgen_idx[0].date()}:")
    for ts, mw in zip(morgen_idx, morgen_pred):
        print(f"  {ts.strftime('%H:%M')}  →  {mw:,.0f} MW")

    plotte_ergebnis(y_test, y_pred, morgen_idx, morgen_pred)
