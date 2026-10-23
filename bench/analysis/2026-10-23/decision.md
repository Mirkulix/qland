# Token-Analyse 2026-10-23 — Befund und Entscheidung für Phase 6

Zahlen aus `report.md` / `summary.json` (erzeugt von `bench/analyze.py` über die 164 Läufe der Benchmarks vom 15.10. und 22.10., 82 je Arm; Haiku-Urteile aus `../llm-cache.jsonl`). Dieses Dokument wird von Hand geschrieben und von `--report` nicht überschrieben.

## Befund

| je Lauf | QAL | Rust | Faktor |
|---|---|---|---|
| verarbeitete Tokens | 299 376 | 145 224 | **2,06** |
| Runden (API-Aufrufe) | 5,00 | 2,95 | 1,69 |
| verarbeitete Tokens je Runde | 59 875 | 49 208 | 1,22 |

Der Faktor 2,06 zerfällt in **1,69 mehr Runden** und **1,22 mehr Kontext je Runde**. Jede Runde liest den ganzen bisherigen Kontext neu; eine Runde mehr kostet rund 60 000 Tokens.

**Woher die zusätzlichen 2,05 Runden je QAL-Lauf kommen** (Runden je Lauf, QAL gegen Rust):

| Rundenursache | QAL | Rust | Differenz | Anteil an QAL-Tokens |
|---|---|---|---|---|
| `ok.*` nützliche Arbeit (define, run, Hilfsbefehle) | 2,63 | 1,43 | **+1,20** | 55,6 % |
| `env.*` Umgebung (Guard-Absagen 0,54, TASK.md lesen 0,24, Werkzeugfehler 0,05) | 0,83 | 0,50 | +0,33 | 14,9 % |
| `lang.*` Sprachfehler (Name 0,24, Typ 0,13, Laufzeit 0,10, Syntax 0,04, Argumente 0,01) | 0,52 | 0,00 | +0,52 | 11,8 % |
| `prompt` erste Runde | 1,00 | 1,00 | 0 | 17,5 % (Referenz allein 2,2 %) |

**Der größte Posten ist nützliche Arbeit in zu vielen Runden.** Rust-Agenten ketten im Mittel 9,4 Shell-Operatoren je Bash-Aufruf (98 von 176 Aufrufen mit drei oder mehr: bauen, mehrere Testeingaben, Vergleich in *einer* Runde). QAL-Agenten rufen `qal_run` im Mittel 2,29-mal je Lauf auf, jede Beispieleingabe in einer eigenen Runde, und definieren 2,2-mal (178 erfolgreiche `define` auf 82 Läufe).

**Sprachfehler** (46 Runden): E0201 „unknown name“ 33-mal, davon 27 lokale Namen außerhalb ihres Gültigkeitsbereichs (meist `let` in einer Sequenz, dessen Bindung nach dem `;` nicht mehr gilt), 3-mal `String` statt `Str`, 2-mal Built-ins als Werte (`trim`, `to_lower` ohne Closure); E0203 Typfehler 18; E0502 Int-Überlauf 7 (Agenten rechnen ohne Rücksicht auf 64-Bit-Fallen); E0100 Syntax 3. E0510 „no entry point main“ 25-mal ist in 23 Fällen die Folge eines gescheiterten `define` im selben verketteten Bash-Aufruf, keine eigene Runde.

**Umgebung:** Die Guard-Absagen (44 Runden QAL, 17 Rust) sind ein Artefakt der Worktree-Sitzung und verschwinden mit der Messvorschrift (Kontrollprozess außerhalb des Worktrees). Das Lesen von `TASK.md` ist in beiden Armen gleich und bleibt.

**Kontext je Runde** (1,22×): Der erste Aufruf kostet in QAL 52 349 gegenüber 47 891 Tokens (Referenz ≈ 3 000 Tokens, längere Regeln, Werkzeugbeschreibungen); dazu kommen die JSON-Ergebnisse der QAL-Werkzeuge, die größer sind als Shell-Ausgaben.

**Ausreißer:** `08-expr-eval` 4,73× (11,7 Runden, viele `qal_run`-Einzelläufe); `16-anagrams` 1,15× (2,5 Runden in beiden Armen: wo QAL in so wenigen Runden auskommt wie Rust, ist der Unterschied nur der Kontext je Runde).

## Entscheidung (Gate: ein Bau-Task braucht ≥ 5 % der verarbeiteten QAL-Tokens)

1. **Task 4, Harness:** ja. Guard-Absagen 7,9 % und außerhalb der Sprache; Audit zählt Absagen und Wiederholungen künftig mit.
2. **Task 5, „Antwort statt Dialog“ wird „weniger Runden“:** ja, erste Priorität. Konkret: (a) `qal_run` nimmt eine Liste von Fällen (`cases: [{args, stdin}]`) und liefert alle Ergebnisse in einer Antwort; (b) `qal_define` führt die Tests der definierten Namen gleich aus und meldet sie; (c) Diagnosen mit Korrekturvorschlag für die drei häufigsten Fehler (lokaler Name nach `;`, `String`, Built-in als Wert). Ziel: `ok.*`-Runden von 2,63 auf etwa 1,6 je Lauf. Betroffener Posten: 55,6 %.
3. **Task 6, Referenz:** ja, aber umgeschnitten. Nicht kürzen (die Referenz kostet nur 2,2 %), sondern die drei Fehlerklassen mit 8,2 % der Tokens adressieren: `let` in Sequenzen, `Str`, Built-ins als Closures, 64-Bit-Überlauf. **Die zweite Seite (`qal_reference`) entfällt** (Anteil der Referenz 2,2 % < 5 %).
4. **Task 7, Rezepte: entfällt.** Kein Posten zeigt fehlende Muster; die Fehler sind Sprachverständnis, und ein Rezept kostet eine Suchrunde, bevor es spart. Nach dem Rerun neu bewerten.
5. **Task 8, Rerun:** nach 4, 5, 6.

**Erwartung, ehrlich gerechnet:** Guard weg (−0,5 Runden), Fälle gebündelt (−1,0), weniger Sprachfehler (−0,25) ergeben etwa 3,3 Runden je Lauf. Bei unverändertem Kontext je Runde (1,22×) wäre der Faktor ≈ (3,3 / 2,95) × 1,22 ≈ **1,36**: im Zwischenbereich der Spec, nicht unter 1,2. Unter 1,2 geht es nur, wenn auch der Kontext je Runde fällt: kompaktere JSON-Antworten der Werkzeuge (Task 5 nimmt das als Nebenziel) oder weniger Runden als Rust. Der Sprachanteil (11,8 %) liegt weit unter der Abbruchgrenze; Phase 8 wird nicht vorgezogen.
