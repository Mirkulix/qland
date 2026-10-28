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
2. **Der Rest des Abstands ist der Kontext je Runde (1,20×)**, und der kommt fast ganz aus dem Prompt: die Sprachreferenz mit allen Werkzeugbeschreibungen macht `TASK.md` fünfmal so lang wie im Rust-Arm (27 KB gegen 5,5 KB). Der Prompt ist fester Teil des Kontexts jeder Runde, also zählt er so oft wie es Runden gibt; der Hebel ist eine kürzere Referenz und kürzere Werkzeugbeschreibungen, nicht ihr Ort. Die Antwort auf `define` (3,1 KB: Signaturen, Hashes) ist dagegen klein.
3. **Die Garantien wurden nicht genutzt.** Weder Verträge (`req`/`ens`) noch `test`-Definitionen kamen vor, obwohl die Referenz sie erklärt; der Rust-Agent schrieb ebenfalls keine Tests. Beide Agenten verließen sich auf die zwei Beispiele und, im QAL-Fall, auf zwei eigene Randfälle. Typ- und Effektprüfung griffen still (kein Fehler beim ersten `define`); die Sandbox ließ nur `io` zu. Was QAL hier mehr liefert, ist ein geprüfter, gehashter Artefakt im Store, nicht ein gefundener Fehler.
4. **`isError: true` bei erwartetem Exit-Code 1.** Zwei der vier Fälle enden laut Aufgabe mit Exit-Code 1; der Aufruf meldet deshalb einen Fehler, obwohl alles stimmt. Der Agent ließ sich nicht beirren, das Audit zählte aber einen „fehlgeschlagenen Werkzeugaufruf“. Das Verhalten ist planmäßig (Task 5), aber falsch für Programme, deren Exit-Code Teil der Spezifikation ist: `isError` sollte sich nach Laufzeitfehlern richten, nicht nach dem Exit-Code.
5. **Messartefakt:** Die „neuen“ Tokens der ersten Runde (39 611 gegen 14 655) spiegeln nur, dass der zuerst gestartete Agent den Cache des Systemprompts anlegt und der zweite ihn liest. `processed` ist davon unabhängig; `new` darf zwischen parallel gestarteten Armen nicht verglichen werden.
6. **Dauer:** QAL brauchte doppelt so lange (102 s gegen 53 s), überwiegend Modellzeit für die längere Ausgabe und die zusätzliche Runde; der Server selbst antwortete in Sekundenbruchteilen.

## Was das für die Frage „hat sich der Aufwand gelohnt“ sagt

Ein Lauf je Arm ist kein Beweis, aber ein belastbares Signal, weil beide Agenten die Aufgabe im ersten Anlauf vollständig lösten und die Zahlen zur Benchmark-Analyse passen:

- **Gelungen:** Ein Agent baut in einer fremden Sprache eine 67-zeilige Regelmaschine mit Datumslogik, Maps und Zustandsübergabe in einem Zug korrekt, allein aus Referenz und Werkzeugen, ohne eine einzige Diagnose. Das ist das Ergebnis der fünf Tage: Sprache, Checker, Store, Werkzeuge und Referenz tragen eine reale Aufgabe.
- **Teuer, aber erklärbar:** 1,61× verarbeitete Tokens (Benchmark-Mittel vorher 2,06×). Die Lücke zu Rust ist jetzt zu etwa zwei Dritteln der Prompt (Referenz und Werkzeugbeschreibungen) und zu einem Drittel die eine Runde mehr. Beides ist messbar und adressierbar: eine kürzere Werkzeugbeschreibung im Prompt (die Referenz nur in `qal_define`, nicht in jeder Runde) und ein `qal_define`, das die Beispiele gleich mit ausführt.
- **Nicht gezeigt:** Dass die Garantien im Alltag einen Fehler verhindern. Dafür müsste die Aufgabe Verträge verlangen oder der Agent sie von sich aus schreiben. Das ist der Punkt, an dem Phase 6b (Gedächtnis) und ein Auftrag mit ausdrücklicher Vertragspflicht ansetzen müssen.

**Urteil nach dem ersten Lauf:** Der Aufwand hat ein funktionierendes, benutzbares System ergeben, das mit einem Agenten reale Aufgaben korrekt löst; das ist belegt. Ob er sich *gelohnt* hat, hängt daran, ob jemand die Garantien braucht, denn nur die rechtfertigen den Tokenaufschlag. Heute zahlt man 1,6× für ein Ergebnis, das in Rust gleich gut war. Deshalb folgte der zweite Lauf: dieselbe Aufgabe mit Vertrags- und Testpflicht in beiden Armen.

## Zweiter Lauf: dieselbe Aufgabe mit Vertrags- und Testpflicht

Aufgabe `22-expense-approval-contracts`: gleiche Regeln und dieselben zwölf versteckten Tests, ergänzt um Anforderungen an den Code selbst: Verträge auf jeder Regelfunktion (QAL `req`/`ens` mit `qal_prove`; Rust `assert!`/`debug_assert!` an den Funktionsgrenzen) und mindestens fünf eigene Tests zu Regelreihenfolge, ungültigem Datum, exakter Obergrenze, Wirkung einer ungültigen Zeile auf die Zusammenfassung und leerer Eingabe (QAL `test`-Definitionen mit `qal_test`; Rust `#[test]` mit `cargo test`).

| | QAL | Rust |
|---|---|---|
| versteckte Tests | **12 / 12**, erster Versuch | **12 / 12**, erster Versuch |
| Runden | 4 | 4 |
| Werkzeugaufrufe | 3 (Read, define; define + prove + test + run mit 3 Fällen) | 3 (Read, Write, Bash: `cargo test` + beide Beispiele) |
| Dauer | 90 s | 48 s |
| verarbeitete Tokens | 243 514 | 191 603 (Faktor **1,27**) |
| Ausgabe-Tokens | 10 274 | 5 224 |
| Programm | 94 Zeilen, 24 Funktionen | 269 Zeilen (mit Tests) |
| eigene Tests | 10 `test`-Definitionen, alle grün; dazu 2 Eigenschaftstests mit je 100 Läufen | 7 `#[test]`, alle grün |
| Verträge | 6 `req`, 8 `ens`; **14 Pflichten bewiesen, 1 unbekannt, 0 widerlegt** | 7 `debug_assert!` (laufen nur im Debug-Build; die versteckten Tests liefen gegen den Release-Build, also ohne sie) |

Beobachtungen:

1. **Der Aufschlag sinkt, wenn Verifikation verlangt wird: 1,27× statt 1,61×.** Rust zahlt für Tests eine eigene Runde (`cargo test`) und doppelt so viel Quelltext; QAL trägt Verträge und Tests im selben `define`, und `qal_define` führt die Tests sofort mit aus (Phase 6, Aufgabe 5): die zehn Tests waren in der Antwort auf das erste `define` schon grün, ohne weitere Runde.
2. **Der Beweiser hat gearbeitet, und der Agent hat mit ihm gearbeitet.** Zwei Pflichten blieben zunächst unbekannt. Eine (`req y >= 0` der Datumsprüfung) lag daran, dass die aufgerufene Hilfsfunktion keinen `ens` hatte; der Agent las den Hinweis, gab `num_part` die Zusicherung `ens r >= 0`, und die Pflicht war bewiesen. Das ist der vorgesehene Mechanismus: Verträge setzen sich aus den Verträgen der Aufgerufenen zusammen. Die zweite (`len(r) == map_size(st.emps)` der Zusammenfassung) blieb unbekannt, weil der Beweiser `map_keys`/`map_size` nur näherungsweise modelliert; der Agent akzeptierte das begründet. Diese Lücke ist ein Befund für `qal-prove`.
3. **Immer noch kein verhinderter Fehler.** Beide Lösungen waren auch ohne Verträge richtig; die unbekannten Pflichten waren Grenzen des Beweisers, keine Fehler im Programm. Was QAL mehr liefert, ist jetzt greifbar: 14 maschinell bewiesene Zusicherungen, 10 Tests und 2 Eigenschaftstests liegen gehasht im Store neben dem Code und laufen bei jeder Neudefinition mit. Die Rust-Garantien sind 7 Tests plus Assertions, die im ausgelieferten Binary nicht ausgeführt werden.
4. **Zweites Mal `isError: true` bei erwartetem Exit-Code 1** (siehe oben, Punkt 4). Das gehört korrigiert, bevor ein Agent lernt, Fehlermeldungen zu ignorieren.
5. **Dauer bleibt das doppelte**, weil QAL die größere Ausgabe schreibt (10 274 gegen 5 224 Tokens): Verträge und Tests sind Text, den das Modell erzeugen muss.

## Dritter Lauf: Änderung an bestehendem Code, schwächeres Modell

Aufgabe `23-expense-approval-change`: ein Änderungsauftrag auf dem Stand aus Lauf 22 (QAL-Store mit Verträgen und Tests; Rust-Datei mit Tests), drei Änderungen: Wochenendregel für `travel` (Wochentagsberechnung, zwischen `duplicate` und `over_limit`), Essenslimit 6000 → 8000, fünfte Spalte „Monate mit Genehmigung“ in der Zusammenfassung. 15 versteckte Tests, darunter acht Wochentagsdaten über Schaltjahre und Jahrhundertgrenzen. Beide Agenten liefen diesmal auf **Haiku**, damit Fehler überhaupt wahrscheinlich werden.

| | QAL | Rust |
|---|---|---|
| versteckte Tests | **15 / 15**, erster Versuch | **15 / 15**, erster Versuch |
| Runden | 16 | 7 |
| Dauer | 200 s | 120 s |
| verarbeitete Tokens | 1 546 106 | 446 091 (Faktor **3,47**) |
| Ausgabe-Tokens | 36 315 | 21 593 |
| Art der Änderung | 40 Zeilen `define`: 8 Funktionen ersetzt oder neu, 7 Tests angepasst, 6 neu; Rest unverändert, `main` nur neu verknüpft | Datei komplett neu geschrieben (356 Zeilen, 16 Tests) |
| Verifikation | 15 Tests grün in der Antwort auf das `define`; 15 Pflichten bewiesen, 1 unbekannt (wie in Lauf 22) | 16 Tests grün |
| Fehler unterwegs | 4 Werkzeugfehler, alle im eigenen Python-Prüfskript des Agenten (KeyError, `sed`-Syntax, fehlendes `allow`); eine Sprachdiagnose E0508 (Effekt `io` nicht gewährt), korrekt | keine |

Beobachtungen:

1. **Auch die Verträge und Tests aus Lauf 22 haben nichts gefangen, weil wieder nichts falsch war.** Haiku änderte in beiden Sprachen richtig, einschließlich der Wochentagsberechnung. Das Experiment „ein Vertrag bricht“ ist damit zum dritten Mal nicht eingetreten; die Garantien waren jedes Mal grün, und sie *konnten* es nur sein, weil der Code stimmte. Ein Fehler, den sie hätten fangen können, ist in drei Läufen nicht entstanden.
2. **Was QAL beim Ändern anders macht, ist sichtbar.** Der Store meldete die Änderung präzise: 8 Funktionen `replaced`/`added`, 2 Tests `unchanged`, `main` `relinked`. Die Änderung war ein 40-Zeilen-Delta gegen die geprüfte Basis, und alle 15 Tests liefen sofort mit; der Rust-Agent schrieb die Datei neu und prüfte dann. Für die Frage „was hat sich geändert, und gilt noch, was vorher galt“ ist QAL hier klar im Vorteil.
3. **Haiku ist in QAL dreimal so teuer wie in Rust, aber nicht wegen der Sprache.** Von den 16 Runden gingen drei auf das Lesen des Programms (`qal_query`, zweimal `qal_view`, zusammen 35 KB Antwort) und neun auf ein selbst geschriebenes Python-Differenzprüfskript mit 240 Zufallsfällen gegen ein eigenes Referenz-Orakel samt dessen Fehlern. Die Werkzeuge der Sprache brauchten vier Runden. Der Tokenabstand ist ein Modell- und Arbeitsstil-Effekt, nicht ein Sprachaufschlag; er zeigt aber, dass ein schwaches Modell mit einer fremden Sprache *mehr* herumprobiert, und jede Runde trägt den 29-KB-Prompt.
4. **`qal_view` des ganzen Programms kostet 21 KB Kontext.** Ein Agent, der ändern soll, braucht eine kompakte Sicht (Signaturen, Verträge, Testnamen) und den Quelltext nur der Funktionen, die er anfasst; `qal_query` liefert die Signaturen, aber kein Werkzeug sagt „welche Funktionen hängen von `cat_limit` ab“. Das ist ein Befund für die Werkzeuge.
5. **Protokollverstoß:** Der QAL-Agent meldete sein Ergebnis über `SendMessage` statt als Abschlussantwort; der Harness wertet den Lauf deshalb als ungültig. Für diesen Einsatztest zählt die Lösung (15/15), der Verstoß bleibt notiert.

## Urteil

Alle drei Läufe zusammen beantworten die Frage so:

- **Das System funktioniert.** Sechs Agentenläufe mit zwei Modellen, sechs richtige Lösungen im ersten Anlauf, davon drei in einer Sprache, die es vor fünf Tagen in dieser Form nicht gab; eine einzige Sprachdiagnose (eine korrekte Effektverweigerung), kein Umweg über den Checker. Sprache, Store, Checker, Beweiser, Sandbox und Werkzeuge tragen reale Aufgaben, auch Änderungen an bestehendem Code, auch mit einem schwachen Modell.
- **Der Preis ist bekannt und hängt vom Auftrag ab.** Ohne Verifikationspflicht 1,61×, mit ihr 1,27×, bei einer Änderung durch Haiku 3,47× verarbeitete Tokens gegenüber Rust. Verlangte Garantien machen QAL relativ günstiger, weil Verträge, Tests und Ausführung in einem Werkzeugaufruf zusammenfallen; ein schwaches Modell macht es teurer, weil es in einer fremden Sprache mehr herumprobiert und jede Runde den 27-KB-Prompt trägt. Der Hebel bleibt derselbe: kürzere Referenz, kompaktere Sicht auf den Store.
- **Der Beleg, dass die Garantien einen Fehler fangen, steht weiter aus.** In drei Aufgaben, zwei Modellen und sechs Läufen ist kein Fehler entstanden, den ein Vertrag oder Test hätte fangen können. Das sagt zweierlei: die Aufgaben waren für heutige Modelle zu klein, und der Wert der Garantien liegt nicht beim Erstschreiben kleiner Programme, sondern beim Ändern großer, über Zeit, durch viele Hände. Dort ist er in diesem Test nur in einer Vorform sichtbar geworden: der Store meldete bei der Änderung exakt, was sich geändert hat und dass alles Vorherige noch gilt.

**Hat sich der Aufwand der fünf Tage gelohnt?** Ja, als Fundament: es gibt ein benutzbares, gemessenes, in sich geschlossenes System, dessen Mehrkosten gegenüber dem Stand der Technik bekannt sind und bei verlangter Verifikation auf rund ein Viertel schrumpfen. Noch nicht als Nachweis des Nutzens: der eigentliche Vorteil, maschinell geprüfte Garantien statt Vertrauen, ist gebaut und funktioniert, hat aber in sechs Läufen keinen Fehler verhindert, weil keiner gemacht wurde. Der nächste Test darf deshalb keine weitere kleine Aufgabe sein, sondern ein Programm von einigen hundert Funktionen, das über mehrere Änderungsaufträge hinweg lebt; erst dort entscheidet sich, ob Verträge und Gedächtnis (Phase 6b) tragen.

## Dateien

- `bench/tasks/21-expense-approval/`, `22-expense-approval-contracts/`, `23-expense-approval-change/` (`task.md`, `tests.json`: Aufgaben, versteckte Tests)
- `bench/einsatz-2026-10-23/schedule.json` (mit Modell je Aufgabe), `results.jsonl`, `workdirs.json` (Zustand der sechs Läufe; Transkripte unter `~/.claude/projects/…/subagents/agent-aab-2[123]-expense-approval-*.jsonl`)
