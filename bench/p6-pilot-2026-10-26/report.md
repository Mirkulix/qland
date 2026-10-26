# Phase-6-Pilot 2026-10-26: kompakte MCP-Antworten

**Frage:** Senkt die MCP-Antwort ohne doppeltes `structuredContent` den QAL/Rust-Tokenfaktor
unter 1,2? **Antwort: Nein, in diesem Pilot bleibt er bei 1,30.**

## Aufbau

- 23 Aufgaben, beide Arme (QAL, Rust), eine Runde je Aufgabe: **46 Dispatches**.
- Modell: **Claude Sonnet** (headless `claude -p`), gleiche Prompt-Regeln und Testfälle wie im
  bestehenden Benchmark.
- QAL-Server mit der kompakten Antwort (JSON nur in `content`; `structuredContent` aus).
- Ergebnis: 44 von 46 Läufen gelöst; **34 valide** (22 Rust, 11 QAL, 1 QAL valide, aber nicht
  bestanden, 1 Rust invalide).
- **11 valide, bestandene Paare** (gleiche Aufgabe in beiden Armen) sind die Vergleichsbasis.

## Ergebnis

| Messung | QAL | Rust | Faktor |
|---|---|---|---|
| verarbeitete Tokens, Summe der 11 Paare | 812 538 | 624 701 | **1,301** |
| Median je Lauf | 73 848 | 56 913 | 1,298 |

Damit liegt der Pilot **über dem Ziel 1,2** und praktisch auf dem bisherigen Living-Wert
(1,32). Die Halbierung der Tool-Antworten hat den Abstand nicht geschlossen.

## Warum nur 11 Paare?

Die Audit-Regel verlangt, dass jede Bash-Zeile mit `cd <workdir> &&` beginnt. Bei 10 QAL-Läufen
fehlte das Prefix (der Agent arbeitete bereits im richtigen Verzeichnis), obwohl die Lösungen
bestanden; diese Läufe sind **invalid**, nicht falsch. Ein QAL-Lauf bestand die Tests nicht, ein
Rust-Lauf war invalid. Die Quote ist ein Artefakt des Headless-Aufbaus, nicht der Sprache; die
11 Paare sind jedoch sauber und in beide Richtungen gepaart.

## Grenzen

- **Ein Lauf je Aufgabe**, nicht die vorgesehenen zwei; 11 statt 40 Paare.
- **Top-level Claude Code**, nicht die Subagenten des ursprünglichen Harness; gleiches Modell,
  aber anderer Prompt-/Tool-Kontext.
- Selbstprüfung auf der Entwicklungsmaschine; keine unabhängige Replikation.
- Die ungültigen QAL-Läufe wurden nicht wiederholt; ihre Tokens sind nicht in der Auswertung.

## Dateien

- `schedule.json` – die 46 Dispatches.
- `results.jsonl` – alle Datensätze inklusive `valid`/`passed` und Dispatch-Tokens.
- `driver.log` – Prepare-/Claude-/Finish-Protokoll.

## Konsequenz

Der nächste Phase-6-Schritt ist nicht eine weitere Tool-Antwort-Kürzung. Es ist die
**Runden-/Prompt-Seite**: weniger Runden und ein kleinerer Systemprompt, gemessen mit dem
vollständigen 80-Lauf-Aufbau (zwei Runden je Aufgabe, valide QAL-Läufe). Bis dahin bleibt
Phase 6 offen.
