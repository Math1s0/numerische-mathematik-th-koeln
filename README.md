# Sicherheitskritische Zeitreihenprognose

Projekt im Rahmen des Kurses **Numerische Mathematik** an der TH Köln.

Vollständige Pipeline zur Stromlastprognose für Deutschland – KI-VO-konform und reproduzierbar.

## Was die Pipeline macht

Lädt stündliche Stromlastdaten von ENTSO-E (2024), bereinigt sie, trainiert ein LightGBM-Modell und erstellt eine 24-Stunden-Vorhersage – komplett mit Audit-Logging, Backtest und Provenienz-Dokumentation.

## Voraussetzungen

- Python 3.x
- Pakete installieren:
```powershell
pip install spotforecast2-safe lightgbm pandas numpy matplotlib
```

## Starten

```powershell
python pipeline.py
```

## Ausgabe

| Datei | Inhalt |
|-------|--------|
| `02_lastdaten.png` | Jahresübersicht Stromlast 2024 |
| `03_datenluecken.png` | Datenlücken vor/nach Bereinigung |
| `04_split.png` | Train/Eval/Reserve-Split |
| `05_rbf.png` | Zyklische Kalender-Features |
| `06_feature_importance.png` | Wichtigste Modell-Features |
| `07_vorhersage.png` | 24-h-Vorhersage vs. Realität |
| `08_metriken.png` | MAE / RMSE / MAPE |
| `09_backtest.png` | Rolling-Origin-Backtest |
| `_cache/provenance.json` | Modell-Dokumentation |

## Zusammenarbeit

```powershell
git pull          # vor jeder Session
git add .
git commit -m "Beschreibung"
git push
```
