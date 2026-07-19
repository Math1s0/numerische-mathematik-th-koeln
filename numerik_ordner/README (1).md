# team_4 — Reproduzierbarkeits-Paket

Dieses Paket enthält alles, um die Energielast-Vorhersage für **[ZIELTAG einsetzen, z.B. 2026-07-03]**
exakt nachzuvollziehen — mit gepinnten Abhängigkeiten und eingefrorenen Daten (kein Live-API-Zugriff nötig).

## Inhalt

```
.
├── pyproject.toml           # Abhängigkeiten (Namen + Constraints)
├── uv.lock                  # Exakt gepinnte Versionen (inkl. transitiver Deps)
├── .python-version           # Fixierte Python-Version
├── 14_team_4_submission.py   # Prognose-Script (Code-Zellen des Notebooks, ohne Plots)
├── interim/
│   ├── energy_load.csv           # ENTSO-E Actual/Forecasted Load
│   ├── renewable_forecast.csv    # Wind/Solar Day-Ahead Forecast
│   ├── day_ahead_price.csv       # DE/LU Strompreis
│   ├── weather_frozen.csv        # Eingefrorene Wetterdaten (Open-Meteo, Snapshot)
│   └── covid_incidence.csv       # COVID-7-Tage-Inzidenz (statisch)
└── submissions/
    └── a_team/
        └── [DATUM]_referenz.csv   # Referenz-Ergebnis zur Verifikation
```

## Voraussetzungen

- [uv](https://docs.astral.sh/uv/) installiert (`pip install uv` oder via offiziellem Installer)
- Keine Internetverbindung zur ENTSO-E- oder Open-Meteo-API nötig — alle Daten liegen bereits als CSV vor

## Installation

```powershell
# 1. In das entpackte Verzeichnis wechseln
cd repro_[DATE]

# 2. Umgebung exakt gemäß uv.lock installieren
uv sync
```

`uv sync` installiert **exakt** die im Lockfile fixierten Paketversionen
(inkl. `spotforecast2-safe`, `lightgbm`, `pandas`, `numpy`, `shap`, etc.) —
unabhängig davon, was aktuell auf PyPI verfügbar ist.

## Ausführung

```powershell
uv run python a_team_script.py
```

Das Script läuft von oben nach unten vollständig automatisch durch und
schreibt am Ende die Submission-CSV.

**Wichtig:** Die Abschnitte, die normalerweise Daten von der ENTSO-E-/Open-Meteo-API
herunterladen, lesen in diesem Paket stattdessen aus dem mitgelieferten
`interim/`-Ordner. Es findet **kein** Live-Download statt — die Zeitkonstanten
(`NOW_UTC`, `TODAY_UTC`, `TOMORROW_UTC`) sind im Script fest auf den
Zielzeitpunkt gesetzt, nicht auf die aktuelle Systemzeit.

Das Script enthält bewusst **keine** Diagramm-/Plot-Ausgaben (kein Matplotlib-,
Plotly- oder SHAP-Plot) — es ist auf den reinen Vorhersage-Pfad reduziert und
läuft daher auch ohne grafische Oberfläche (z. B. auf einem Server) vollständig
non-interaktiv durch.

## Verifikation

Nach vollständigem Durchlauf sollte die neu erzeugte Submission-Datei
**exakt** mit der im Paket mitgelieferten Referenzdatei (`submissions/a_team/[DATUM]_referenz.csv`)
übereinstimmen. Prüfe dies z.B. mit:


## Kontakt / Herkunft

- Erstellt von: [a_team]
- Datum der Original-Vorhersage: [DATE]
- Modell: LightGBM, rekursive Ein-Schritt-Vorhersage, SpotOptim-Hyperparameter-Tuning
- Git-Commit / Version der Analyse-Software: [COMMIT-HASH einsetzen, falls vorhanden]
