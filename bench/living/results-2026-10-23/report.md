# Lebendes Programm, Lauf vom 2026-10-23: Ergebnis

Protokoll: `bench/living/README.md`. Ein Bestellsystem (Produkte, Kunden, Bestellungen, Bericht) wird in Stufe 0 mit Verträgen und Tests gebaut und dann durch zwölf Änderungsaufträge geführt (Mengenrabatt, Kundenstufen, Storno, Steuer, Kredit, Nachbestellung, Rückstandsaufträge, Preisänderung, Top-Produkte, Rückgaben, Aktionen, Mehrzeilenbestellungen). Vier Linien: QAL und Rust, je mit Sonnet und Haiku; jede Stufe ein frischer Agent, der nur den Auftragstext und den Code der Vorstufe seiner Linie kennt. 16 bis 50 versteckte Tests je Stufe aus einem Python-Orakel; eine Wiederholung je Stufe erlaubt. Zahlen aus `summary.json` (Transkripte, `bench/analyze.py`), Rohdaten in `results.jsonl`.

## Ergebnis in einer Zeile

**52 Läufe, 52 Stufen im ersten Versuch bestanden, kein einziger Fehler hat die versteckten Tests erreicht, in keiner Linie.** Kein Wiederholungsversuch war nötig.

## Zahlen

| Linie | Stufen | Runden | verarbeitete Tokens | Ausgabe-Tokens | Dauer | Programm am Ende |
|---|---|---|---|---|---|---|
| QAL / Sonnet | 13 / 13 | 114 | 8,33 Mio | 110 k | 25 min | 62 Fn, 6 Typen, 82 Tests, 19 Vertragsklauseln, **59 Pflichten bewiesen, 0 unbekannt** |
| Rust / Sonnet | 13 / 13 | 83 | 4,48 Mio | 43 k | 25 min | 678 Zeilen, 57 Fn, 47 Tests, 117 Assertions |
| QAL / Haiku | 13 / 13 | 161 | 14,62 Mio | 367 k | 35 min | 68 Fn, 9 Typen, 133 Tests, 13 Vertragsklauseln, 19 bewiesen, 1 unbekannt |
| Rust / Haiku | 13 / 13 | 96 | 6,71 Mio | 231 k | 24 min | 1 775 Zeilen, 144 Fn, 127 Tests, 249 Assertions |

Faktor QAL gegen Rust in verarbeiteten Tokens: **1,86×** (Sonnet), **2,18×** (Haiku). Je Änderung: QAL/Sonnet 641 k, Rust/Sonnet 345 k, QAL/Haiku 1,12 Mio, Rust/Haiku 516 k. Die Dauer ist bei Sonnet gleich (QAL schreibt mehr Text, Rust kompiliert); bei Haiku braucht QAL die Hälfte länger.

## Was die eigene Verifikation unterwegs gefangen hat

Gezählt sind Werkzeugantworten, in denen die Verifikation des Programms selbst einen Fehler meldete, bevor die versteckten Tests liefen (Details je Stufe in `summary.txt`):

| Linie | rote Tests beim `define` / `cargo test` | davon neue Tests des Laufs | davon vorher bestehende Tests | verweigerte Typänderung (E0401) | Eigenschaftstest rot |
|---|---|---|---|---|---|
| QAL / Sonnet | 7 Antworten | 8 | 3 | 4 | 0 |
| QAL / Haiku | 1 Antwort | 0 | 5 | 11 | 1 |
| Rust / Sonnet | 10 Antworten | 3 | 1 (plus ungenannte) | – | – |
| Rust / Haiku | 5 Antworten | 6 | 0 | – | – |

Was dahinter steckt, nach Lektüre der Transkripte:

1. **Beide Arme fangen die eigenen Fehler, bevor sie ausgeliefert werden.** Die roten *neuen* Tests sind Fälle, in denen die erste Implementierung einer Regel falsch war und der eigene Test es zeigte: in QAL der Überlauf von `price × qty × 95` (Stufe 1), vier falsche `top`-Tests (Stufe 9), die Steueraufteilung bei Rückgaben (Stufe 10), die 50-%-Kappung zweimal (Stufe 11); in Rust der Überlauf bei großen Beträgen (Stufe 2), die Kreditfreigabe (Stufe 5), die Rückgabemenge (Stufe 10). Die roten *bestehenden* Tests sind überwiegend Erwartungen, die der Auftrag absichtlich änderte (Stufe 7: `REJECT stock` wird `BACKORDER`, fünf Tests auf einmal), also kein Fehler, sondern der Nachweis, dass die alten Tests noch gelten und mitlaufen.
2. **Der Unterschied liegt im Zeitpunkt, nicht im Ergebnis.** In QAL kamen die roten Tests in der Antwort auf das `define` selbst, ohne eigenen Testaufruf; in Rust erst nach `cargo test`, das jeder Agent brav aufrief. Am Ende stand in beiden Armen null durchgerutschte Fehler. Die Garantien haben hier keinen Fehler verhindert, den Rust durchgelassen hätte, weil auch die Rust-Agenten mit Tests arbeiteten und keinen durchließen.
3. **Die verweigerte Typänderung (E0401) ist der teuerste Posten von QAL.** Wer einen Typ um ein Feld erweitert, muss die abhängigen Definitionen im selben `define` mitsenden; sonst verweigert der Store. Haiku lief in 11 von 12 Änderungen in diese Verweigerung und brauchte je eine Runde (rund 100 k Tokens), um die Abhängigen nachzureichen; Sonnet viermal. Das ist korrektes Verhalten des Stores (alles oder nichts), aber eine Werkzeuglücke: die Diagnose nennt die Abhängigen, der Agent könnte sie mit einem Schalter automatisch neu verknüpfen lassen, wenn ihre Texte unverändert noch typisieren.
4. **Eine unbekannte Pflicht wurde zwölf Stufen lang vererbt.** In der Haiku-Linie blieb `amount`s `ens` von Stufe 0 bis 12 „unknown“; jeder Agent schrieb „nicht von mir geändert“ und ließ es stehen. In der Sonnet-Linie wurden die unbekannten Pflichten in Stufe 7 geschlossen, danach war alles bewiesen. Das ist genau der Fall für das Erfahrungsgedächtnis (Phase 6b): der Hinweis „so wurde das früher bewiesen“ fehlt dem Agenten.
5. **Die Outline wurde durchgehend genutzt.** Jeder QAL-Agent las zuerst `qal_query {"outline": true}` und sah sich dann nur die Definitionen an, die er änderte; nach der Kürzung (ohne `std.`-Namen und Hashes) kostete das 7 bis 14 KB statt 84 KB.
6. **Haiku ist in QAL doppelt so teuer wie in Rust, Sonnet nicht ganz.** Haikus QAL-Läufe brauchten 10 bis 19 Runden: die Typverweigerung, selbstgeschriebene Prüfskripte, wiederholte `qal_test`-Aufrufe. Sein Rust wuchs auf 1 775 Zeilen und 127 Tests, dreimal so viel wie Sonnets, aber richtig.
7. **Protokollverstöße:** fünf Läufe schrieben eine Datei nach `/tmp` oder benutzten ein fremdes Werkzeug (zweimal Rust/Sonnet, zweimal QAL/Haiku, einmal Rust/Sonnet `bash_placeholder`). Für die Lösungen ohne Belang, als Harness-Befund notiert.

## Was das für QAL bedeutet

- **Die Sprache trägt ein Programm über zwölf Änderungen durch viele fremde Hände.** 26 QAL-Agenten, die das Programm nie gesehen hatten, änderten es 26-mal richtig, mit am Ende 59 bewiesenen Pflichten in der Sonnet-Linie. Keine Sprachdiagnose außer der beabsichtigten Typverweigerung; kein Umweg über den Checker. Das war vor fünf Tagen nicht möglich und ist jetzt gemessen.
- **Der Nachweis „Verträge fangen, was Rust durchlässt“ ist auch hier nicht erbracht**, und zwar aus einem Grund, den das Experiment klar zeigt: beide Arme lieferten null Fehler aus. Der Baseline-Arm (Rust, `cargo test`, gewissenhafte Agenten) ist zu gut, als dass ein Vorteil in der Auslieferungsqualität sichtbar würde. Um ihn zu sehen, müsste man den Druck erhöhen: vagere Aufträge, größere Programme, Änderungen, die alte Annahmen brechen, oder Agenten ohne Testpflicht.
- **Der messbare Vorteil liegt in der Form der Verifikation:** sie ist automatisch (Tests laufen beim Definieren), vollständig (jeder Lauf meldet, was sich geändert hat und ob alles Alte noch gilt), und sie wächst mit (59 Pflichten am Ende, keine unbekannt). Das kostet 1,9- bis 2,2-fache Tokens, davon ein gut benennbarer Teil (E0401-Runden, 27-KB-Prompt) behebbar.

## Nächste Schritte, die aus diesem Lauf folgen

1. `qal_define` mit `relink: true`: abhängige Definitionen, deren Text unverändert noch typisiert, automatisch neu verknüpfen statt verweigern. Spart bei Haiku rund ein Sechstel der Tokens.
2. Phase 6b (Gedächtnis), damit „unknown“-Pflichten nicht über zwölf Stufen vererbt werden: der Hinweis in der Diagnose nennt die frühere Lösung.
3. Ein Lauf mit erhöhtem Druck: dieselbe Änderungsfolge mit Aufträgen ohne Testpflicht und ohne „Full specification“, nur der Änderungstext. Erst dort kann sich zeigen, ob die Garantien einen Unterschied in der Auslieferung machen.

## Dateien

`schedule.json`, `results.jsonl`, `workdirs.json` (Zustand), `summary.json` / `summary.txt` (Auswertung), Aufgaben unter `bench/living/tasks/`, Archive der Arbeitsverzeichnisse unter `~/.cache/qal-bench-archive/3*-living-*` und `4*-living-*`, Transkripte unter `~/.claude/projects/…/subagents/agent-alv-s*-*.jsonl`.
