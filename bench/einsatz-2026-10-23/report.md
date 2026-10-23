# Einsatztest 2026-10-23: Spesenfreigabe, QAL gegen Rust

Erster Einsatz nach Phase 6 (Aufgaben 1–6 auf `Mac`, `1c0a88a4`): eine realistische Regelaufgabe, die nicht aus dem Benchmark stammt, einmal von einem Agenten in QAL über die MCP-Werkzeuge gebaut, einmal vom gleichen Modell (Claude Code Subagent, `sonnet`) in Rust; gleiche Aufgabenstellung, gleiche Regeln, zwölf versteckte Tests, die vorher aus einem Python-Orakel erzeugt wurden (`bench/tasks/21-expense-approval/`). Protokoll wie der Rerun vom 22.10. ab Lauf 37 (`TASK.md` im Arbeitsverzeichnis), Kontrollsitzung außerhalb jedes Worktrees, je Arm ein Lauf.

## Aufgabe

Spesenanträge zeilenweise prüfen (`employee;date;category;amount`): Formatprüfung mit echten Kalenderdaten (Schaltjahre), Duplikate, Kategorielimits, Monatsobergrenze je Mitarbeiter nur über genehmigte Beträge, Zusammenfassung je Mitarbeiter in Byte-Reihenfolge, Exit-Code 1 bei irgendeiner Ablehnung. Versteckte Tests decken leere Eingabe, CRLF, fehlenden Zeilenumbruch, Betragsgrenzen (0, 10 Stellen, führende Nullen, `+5`), exakte Obergrenze, nicht gezählte Ablehnungen, Duplikat nach Ablehnung und Sortierung ab.

## Ergebnis

| | QAL | Rust |
|---|---|---|
| versteckte Tests | **12 / 12**, erster Versuch | **12 / 12**, erster Versuch |
| Runden (API-Aufrufe) | 4 | 3 |
| Werkzeugaufrufe | 3 (Read, define, run mit 4 Fällen) | 2 (Read, Bash: schreiben + bauen + beide Beispiele) |
| Dauer | 102 s | 53 s |
| verarbeitete Tokens | 217 956 | 135 718 (Faktor **1,61**) |
| Tokens je Runde | 54 489 | 45 239 (Faktor 1,20) |
| Ausgabe-Tokens | 4 731 | 2 980 |
| Prompt (`TASK.md`) | 27 074 Zeichen | 5 538 Zeichen |
| Programm | 67 Zeilen, 23 Funktionen, 0 Tests, 0 Verträge | ein `main.rs`, keine Tests |
| Sprachfehler | keine | keine |

Zahlen aus `results.jsonl` dieses Verzeichnisses (Audit `bench/run.py`) und `bench/analyze.py` über die beiden Transkripte.

## Beobachtungen

1. **Phase 6 wirkt sofort.** Der QAL-Agent führte seine vier Testfälle in *einem* `qal_run`-Aufruf mit `cases` aus. Im Benchmark vom 22.10. kostete jeder Fall eine Runde (2,3 je Lauf); hier wäre das ohne `cases` eine Runde mehr gewesen, also rund 60 000 verarbeitete Tokens.
2. **Der Rest des Abstands ist der Kontext je Runde (1,20×)**, und der kommt fast ganz aus dem Prompt: die Sprachreferenz mit allen Werkzeugbeschreibungen macht `TASK.md` fünfmal so lang wie im Rust-Arm (27 KB gegen 5,5 KB). Jede Runde liest das neu. Die Antwort auf `define` (3,1 KB: Signaturen, Hashes) ist dagegen klein.
3. **Die Garantien wurden nicht genutzt.** Weder Verträge (`req`/`ens`) noch `test`-Definitionen kamen vor, obwohl die Referenz sie erklärt; der Rust-Agent schrieb ebenfalls keine Tests. Beide Agenten verließen sich auf die zwei Beispiele und, im QAL-Fall, auf zwei eigene Randfälle. Typ- und Effektprüfung griffen still (kein Fehler beim ersten `define`); die Sandbox ließ nur `io` zu. Was QAL hier mehr liefert, ist ein geprüfter, gehashter Artefakt im Store, nicht ein gefundener Fehler.
4. **`isError: true` bei erwartetem Exit-Code 1.** Zwei der vier Fälle enden laut Aufgabe mit Exit-Code 1; der Aufruf meldet deshalb einen Fehler, obwohl alles stimmt. Der Agent ließ sich nicht beirren, das Audit zählte aber einen „fehlgeschlagenen Werkzeugaufruf“. Das Verhalten ist planmäßig (Task 5), aber falsch für Programme, deren Exit-Code Teil der Spezifikation ist: `isError` sollte sich nach Laufzeitfehlern richten, nicht nach dem Exit-Code.
5. **Messartefakt:** Die „neuen“ Tokens der ersten Runde (39 611 gegen 14 655) spiegeln nur, dass der zuerst gestartete Agent den Cache des Systemprompts anlegt und der zweite ihn liest. `processed` ist davon unabhängig; `new` darf zwischen parallel gestarteten Armen nicht verglichen werden.
6. **Dauer:** QAL brauchte doppelt so lange (102 s gegen 53 s), überwiegend Modellzeit für die längere Ausgabe und die zusätzliche Runde; der Server selbst antwortete in Sekundenbruchteilen.

## Was das für die Frage „hat sich der Aufwand gelohnt“ sagt

Ein Lauf je Arm ist kein Beweis, aber ein belastbares Signal, weil beide Agenten die Aufgabe im ersten Anlauf vollständig lösten und die Zahlen zur Benchmark-Analyse passen:

- **Gelungen:** Ein Agent baut in einer fremden Sprache eine 67-zeilige Regelmaschine mit Datumslogik, Maps und Zustandsübergabe in einem Zug korrekt, allein aus Referenz und Werkzeugen, ohne eine einzige Diagnose. Das ist das Ergebnis der fünf Tage: Sprache, Checker, Store, Werkzeuge und Referenz tragen eine reale Aufgabe.
- **Teuer, aber erklärbar:** 1,61× verarbeitete Tokens (Benchmark-Mittel vorher 2,06×). Die Lücke zu Rust ist jetzt zu etwa zwei Dritteln der Prompt (Referenz und Werkzeugbeschreibungen) und zu einem Drittel die eine Runde mehr. Beides ist messbar und adressierbar: eine kürzere Werkzeugbeschreibung im Prompt (die Referenz nur in `qal_define`, nicht in jeder Runde) und ein `qal_define`, das die Beispiele gleich mit ausführt.
- **Nicht gezeigt:** Dass die Garantien im Alltag einen Fehler verhindern. Dafür müsste die Aufgabe Verträge verlangen oder der Agent sie von sich aus schreiben. Das ist der Punkt, an dem Phase 6b (Gedächtnis) und ein Auftrag mit ausdrücklicher Vertragspflicht ansetzen müssen.

**Urteil:** Der Aufwand hat ein funktionierendes, benutzbares System ergeben, das mit einem Agenten reale Aufgaben korrekt löst; das ist belegt. Ob er sich *gelohnt* hat, hängt daran, ob jemand die Garantien braucht, denn nur die rechtfertigen den Tokenaufschlag. Heute zahlt man 1,6× für ein Ergebnis, das in Rust gleich gut war. Der nächste Schritt ist deshalb nicht mehr Sprache, sondern ein Auftrag, bei dem Verträge und Tests verlangt werden, und ein Prompt, der die Referenz nicht in jede Runde trägt.

## Dateien

- `bench/tasks/21-expense-approval/task.md`, `tests.json` (Aufgabe, versteckte Tests)
- `bench/einsatz-2026-10-23/schedule.json`, `results.jsonl`, `workdirs.json` (Zustand des Laufs; Transkripte unter `~/.claude/projects/…/subagents/agent-aab-21-expense-approval-*.jsonl`)
