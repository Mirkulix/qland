# Phase-6-Pilot 2026-10-26: 92 Dispatches, kompakte MCP-Antworten

**Frage:** Senkt die MCP-Antwort ohne doppeltes `structuredContent` den QAL/Rust-Tokenfaktor
unter 1,2? **Antwort: Nein, nach 92 Dispatches bleibt er bei 1,31.**

## Aufbau

- 23 Aufgaben, beide Arme (QAL, Rust), zwei Runden je Aufgabe: **92 Dispatches** (zwei Wellen).
- Modell: **Claude Sonnet** (headless `claude -p`), gleiche Prompt-Regeln und Testfälle wie im
  bestehenden Benchmark.
- QAL-Server mit der kompakten Antwort (JSON nur in `content`; `structuredContent` aus).
- Ergebnis: **90 von 92 Läufen gelöst**; **70 valide** (43 Rust, 26 QAL, 1 QAL valide, aber nicht
  bestanden, 19 QAL und 3 Rust invalide).
- **18 valide, bestandene Aufgabenpaare** (je zwei QAL- und zwei Rust-Läufe, gemittelt) sind die
  Vergleichsbasis.

## Ergebnis

| Messung | QAL | Rust | Faktor |
|---|---|---|---|
| verarbeitete Tokens, Summe der 18 Paare | 1 347 588,5 | 1 026 006 | **1,313** |
| Median der Paar-Mittel | 74 973 | 56 977 | 1,316 |

Damit liegt der Pilot **über dem Ziel 1,2** und praktisch auf dem bisherigen Living-Wert
(1,32). Die Halbierung der Tool-Antworten hat den Abstand nicht geschlossen.

## Warum 19 QAL-Läufe invalid sind

Die Audit-Regel verlangt, dass jede Bash-Zeile mit `cd <workdir> &&` beginnt. Bei 19 QAL-Läufen
fehlte das Prefix (der Agent arbeitete bereits im richtigen Verzeichnis), obwohl die Lösungen
bestanden; diese Läufe sind **invalid**, nicht falsch. Ein QAL-Lauf bestand die Tests nicht, drei
Rust-Läufe waren invalid. Die Quote ist ein Artefakt des Headless-Aufbaus, nicht der Sprache; die
18 Paare sind jedoch sauber und in beide Richtungen gepaart.

## Grenzen

- Zwei Läufe je Aufgabe, aber nur **18 von 23 Aufgaben** sauber gepaart (Audit-Invalide).
- **Top-level Claude Code**, nicht die Subagenten des ursprünglichen Harness; gleiches Modell,
  aber anderer Prompt-/Tool-Kontext.
- Selbstprüfung auf der Entwicklungsmaschine; keine unabhängige Replikation.
- Die ungültigen QAL-Läufe wurden nicht wiederholt; ihre Tokens sind nicht in der Auswertung.
- Beide Wellen liefen mit demselben Modell und denselben Prompts; die zweite Welle nutzte die
  klargestellte `cd`-Regel.

## Dateien

- `schedule.json` – die 46 Aufgabenpaare (zweite Welle mit `run=2`).
- `results.jsonl` – alle 92 Datensätze inklusive `valid`/`passed` und Dispatch-Tokens.
- `driver.log`, `driver2.log` – Prepare-/Claude-/Finish-Protokoll beider Wellen.

## Konsequenz

Der nächste Phase-6-Schritt ist nicht eine weitere Tool-Antwort-Kürzung. Es ist die
**Runden-/Prompt-Seite**: weniger Runden und ein kleinerer Systemprompt, gemessen mit dem
vollständigen 80-Lauf-Aufbau (zwei Runden je Aufgabe, valide QAL-Läufe). Bis dahin bleibt
Phase 6 offen.
