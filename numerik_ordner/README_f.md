# team_4 — Energielast-Vorhersage

Script zur Energielast-Vorhersage für Deutschland (LightGBM, rekursive Ein-Schritt-Vorhersage
mit SpotOptim-Hyperparameter-Tuning).

## Voraussetzungen

- [uv](https://docs.astral.sh/uv/) installiert (`pip install uv` oder via offiziellem Installer)
- Internetzugang zur ENTSO-E-API (und ggf. Open-Meteo, falls Wetter-Features aktiv sind)

## Installation

```powershell
uv sync
```

Installiert alle benötigten Pakete (`spotforecast2-safe`, `lightgbm`, `pandas`, `numpy`, etc.)
gemäß `pyproject.toml` / `uv.lock`.

## Ausführung

```powershell
uv run python a_team_script.py
```

Das Script läuft von oben nach unten vollständig automatisch durch:

1. Lädt aktuelle ENTSO-E-Daten (Actual/Forecasted Load, Wind/Solar-Prognose, Day-Ahead-Preis)
2. Bereitet die Daten auf und führt Qualitätschecks durch
3. Baut die Features (Lags, saisonale Kodierungen, Kalender, Wetter, exogene Provider)
4. Tuned die Hyperparameter (SpotOptim) und trainiert das LightGBM-Modell
5. Erstellt die rekursive 24h-Vorhersage für den Zieltag
6. Validiert das Ergebnis (Schema-Check, Plausibilitäts-Shape-Check)
7. Schreibt die Submission-CSV nach `submissions/a_team/`


## Zeitkonstanten anpassen

Die Vorhersage bezieht sich standardmäßig auf "jetzt" (`NOW_UTC = pd.Timestamp.now(tz="UTC")`).
Soll stattdessen ein fester Stichtag verwendet werden (z. B. um für einen bestimmten Tag
vorherzusagen), müssen `NOW_UTC`, `TODAY_UTC`, `YESTERDAY_UTC` und `TOMORROW_UTC` im Script
entsprechend fest gesetzt werden.


## Kontakt

- Team: a_team
- Modell: LightGBM, rekursive Ein-Schritt-Vorhersage, SpotOptim-Hyperparameter-Tuning
