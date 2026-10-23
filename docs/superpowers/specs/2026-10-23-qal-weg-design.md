# QAL — Der Weg zu einer Sprache für KI-Systeme: Design

**Datum:** 2026-10-23
**Status:** Entwurf zur Prüfung
**Scope:** Zielbild, Begründung, Phasen 6–9 mit Abnahmekriterien, Abbruchkriterien. Keine Implementierungsdetails; die kommen je Phase in eigenen Spezifikationen.

---

## 1. Ausgangslage

QAL ist seit dem 15.10.2026 in fünf Phasen entstanden: inhaltsadressierter Programmgraph statt Dateien, Operationen (MCP-Werkzeuge) statt Textdiffs, Typ- und Effektprüfung als Bedingung fürs Speichern, Verträge mit Z3-Beweisen, WebAssembly-Sandbox mit Effekten als Berechtigungen, Mehr-Agenten-Protokoll mit Rollen und Ed25519-Signaturen, selbst-hostender Compiler mit byte-identischem Fixpunkt. Seit dem 22.10. Effekt-Polymorphie, seit dem 23.10. Beweise über Str und List.

Die ursprüngliche These lautete: *Eine Sprache, die für Agenten gebaut ist, macht Agenten messbar effektiver.* Zwei Messungen mit je 80 Läufen (`bench/report.md`, `bench/rerun-2026-10-22/report.md`) ergaben: gleiche Korrektheit wie Rust (40 von 40), aber etwa das 1,9-Fache an Tokens; ein Vierer-Team kostet das Sechsfache eines Einzelagenten. Die These ist weder bestätigt noch widerlegt. Belegt ist: Agenten können in QAL vollständige Systeme bauen (Standardbibliothek, Todo-Backend, Escrow-Pilot, Teile des Compilers), und die Garantien halten (Effekt-Isolation, Beweise, Fixpunkt).

Der Schmerz, den QAL beseitigen soll, ist benannt (Auftraggeber, 23.10.): Wenn ein Agent heute Software baut, geht der Großteil der Tokens und Fehler nicht in das Problem, sondern in den Kampf mit der Umgebung: Dateien, Build-Systeme, unklare Fehlermeldungen, unbekannte Bibliotheken, Code ohne Korrektheitsnachweis.

## 2. Zielbild

**In zwei Jahren:** KI-Systeme (LLMs und andere Modellklassen; der Auftraggeber nennt sie zusammen „LLM und NCM“) bauen Software, indem sie Definitionen in einen geprüften Graphen schreiben und von jedem Schreibvorgang eine vollständige Antwort bekommen: Typen, Effekte, Testergebnisse, Beweis oder Gegenbeispiel. Software *für* KI-Systeme (Werkzeuge, Orchestrierung, Auswertung von Modellantworten) ist in derselben Sprache geschrieben, läuft deterministisch und wiederholbar in der Sandbox, und jeder Modellaufruf ist ein Effekt mit Budget. Ein Modell, das die Sprache kennt, braucht für eine korrekte Lösung weniger Tokens als in Python oder Rust, nicht mehr.

**Drei Eigenschaften machen den Unterschied, und nur diese drei werden weiterverfolgt:**

1. **Geschlossener Kreis.** Schreiben, prüfen, testen, beweisen und ausführen sind *ein* Aufruf mit *einer* strukturierten Antwort. Kein Shell, keine Dateien, kein Raten.
2. **Nur Geprüftes existiert.** Was gespeichert ist, ist typ- und effektkorrekt, seine Verträge sind bewiesen oder als unbewiesen markiert, seine Tests laufen. Ein Agent muss nichts „glauben“.
3. **Determinismus mit Budget.** Jedes Programm, auch eines, das Modelle aufruft, ist mit aufgezeichneten Antworten exakt wiederholbar und hat eine Obergrenze an Rechenzeit, Speicher und Modellkosten.

## 3. Ziele, Nicht-Ziele, Anti-Ziele

**Ziele**
- Gleiche Korrektheit wie Rust bei höchstens 1,2-fachem Tokenverbrauch im bestehenden Benchmark (Phase 6), später darunter (Phase 8).
- Ein echter Dienst in produktivem Gebrauch, von Agenten gebaut, dessen Garantien mindestens einen Fehler nachweislich verhindert haben (Phase 7).
- Modellaufrufe als Effekt mit Aufzeichnung, Wiederholung und Budget; eine Agenten-Orchestrierung in QAL, die deterministisch getestet wird (Phase 7).
- Ein offenes Modell, das QAL kann, trainiert auf einem selbst erzeugten, maschinell verifizierten Korpus; Vergleich im Benchmark (Phase 8).
- Installation in einem Schritt, Anbindung an jede Agenten-Umgebung über MCP, veröffentlichter Bericht (Phase 9).

**Nicht-Ziele**
- Menschliche Lesbarkeit als Optimierungsziel. Menschen lesen Diagnosen und Verträge, nicht Rümpfe.
- Grafische Oberflächen, Mobile, Browser-Code. QAL bleibt Logik-, Daten- und Orchestrierungsschicht; Oberflächen kommen von außen oder als Server-Rendering.
- Numerik, maschinelles Lernen in QAL selbst. Modelle werden aufgerufen, nicht implementiert.
- Kompatibilität mit bestehenden Paketökosystemen (npm, crates.io, PyPI). Fremdcode kommt nur über Host-Effekte.

**Anti-Ziele** (so wäre es ein Misserfolg, auch wenn es technisch läuft)
- Eine Sprache, die nur mit einem eigens trainierten Modell funktioniert. Jeder Agent muss sie über die Referenz und die Diagnosen lernen können.
- Garantien, die auf dem Papier stehen, aber nicht gemessen werden. Jede Behauptung in der Roadmap nennt den Test, der sie belegt (Roadmap §6).
- Mehr Sprache ohne Einsatz. Nach Phase 6 wird kein Sprachmittel mehr gebaut, das nicht ein konkreter Einsatz (Phase 7) verlangt.

## 4. Was QAL heute davon schon ist

| Eigenschaft | Stand | Lücke |
|---|---|---|
| Geschlossener Kreis | `qal_define` prüft, beweist, testet; Diagnosen mit Pfad und Hinweis | Mehrere Aufrufe pro Änderung üblich (define, test, run); Antworten teils abgeschnitten; Referenz nicht nach Fehlerhäufigkeit geformt |
| Nur Geprüftes existiert | Typen, Effekte, Verträge (Int, Bool, Str, List), Tests, Fixpunkt | Float ungedeckt; Beweise nutzen nur Verträge, nie Rümpfe; Str-Prover neu und dünn getestet |
| Determinismus mit Budget | Fuel und Speicher in der Sandbox; Effekte als Berechtigungen | **Kein Modell-Effekt.** Programme können heute keine Modelle aufrufen; nichts wird aufgezeichnet oder wiederholt |
| Agenten als Bauende | Benchmark 40/40, Team-Protokoll, std von Agenten geschrieben | 1,9-fache Tokens; Team 6-fach teurer; kein produktiver Einsatz |
| Lernende Modelle | — | Kein Korpus, kein Training, kein Vergleich |

Antwort auf die Frage „Hast du das mit QAL angefangen?“: Ja. Die Eigenschaften 1 und 2 sind angefangen und zum Teil belegt. Eigenschaft 3 fehlt in dem Punkt, der KI-Systeme von anderen Programmen unterscheidet: dem Modellaufruf. Der Beleg, dass das Ganze Agenten *schneller* macht, fehlt ganz.

## 5. Der Weg: vier Phasen

Die Reihenfolge ist zwingend. Jede Phase hat ein Abnahmekriterium, das vor der nächsten erfüllt sein muss, und ein Abbruchkriterium, das die Richtung ändert.

### Phase 6 — Der Kreis wird billig (4–6 Wochen)

**Zweck:** Herausfinden und beseitigen, wohin die Tokens gehen. Ohne diesen Schritt ist jede weitere Sprachentwicklung Spekulation.

1. **Token-Analyse.** Aus den 160 Benchmark-Läufen je Lauf auswerten: Anteile für Referenz lesen, Syntaxfehler, Typfehler, Effektfehler, Testfehler, Wiederholungen, Umgebung (Werkzeugabsagen, Pfade). Ergebnis: eine Tabelle, die die drei größten Posten benennt. Werkzeug: `bench/run.py audit`, erweitert.
2. **Antwort statt Dialog.** `qal_define` liefert in *einer* Antwort alles, was der Agent für den nächsten Schritt braucht: Diagnosen mit konkretem Korrekturvorschlag (nicht nur Hinweis), Testergebnisse der betroffenen Definitionen, Beweisstatus mit Gegenbeispiel, und bei Erfolg die Signaturen, die sich geändert haben. Abschneiden nur noch mit Zusammenfassung dessen, was fehlt.
3. **Referenz nach Fehlerhäufigkeit.** Die Sprachreferenz in `qal_define` wird aus der Token-Analyse neu geschnitten: die häufigsten Fehler bekommen das Beispiel, seltene Konstrukte verlassen die Erstseite in eine zweite Seite, die auf Anfrage kommt.
4. **Rezepte.** Für die Aufgabentypen des Benchmarks (Parsen, Zustandsautomaten, Dateien, Tabellen) je ein geprüftes Beispiel im Store, das `qal_query` findet. Agenten kopieren Muster; gute Muster sparen Runden.
5. **Benchmark wiederholen**, gleiche 20 Aufgaben, gleiche Modelle, gleiche Auswertung.

**Messvorschrift** (Festlegung des Verfassers): Das Maß ist *verarbeitete Tokens einschließlich Cache-Lesen* je Lauf, wie in `bench/rerun-2026-10-22/report.md` als Hauptzahl (1,92); Dispatch-Tokens werden mitberichtet. Beide Arme werden im selben Durchgang mit demselben Modell neu gemessen; es gibt keine feste Rust-Baseline, weil der Modell-Alias wechselt. Der QAL-Arm darf den Transport wechseln (native MCP-Werkzeuge statt `qalc.py` über die Shell) und läuft außerhalb einer Worktree-Sitzung, damit kein Git-Guard Befehle abweist; „gleiche Auswertung“ heißt gleiche Aufgaben, gleiche versteckten Tests, gleiche Kennzahlen. Die Token-Analyse (Nr. 1) trennt den Anteil der Umgebung (Transport, Absagen, Pfade) vom Anteil der Sprache (Syntax-, Typ-, Effekt-, Testfehler, Referenz), damit das Abbruchkriterium auf den Sprachanteil bezogen werden kann.

**Abnahme:** Tokens QAL ≤ 1,2 × Rust bei gleicher Korrektheit (40/40), über 80 Läufe. **Zwischenbereich:** Liegt der Faktor nach den Maßnahmen zwischen 1,2 und 1,5, folgen höchstens zwei weitere Wochen Iteration; danach beginnt Phase 7 ungeachtet des Ergebnisses, und die Korpus-Erzeugung aus Phase 8 (Nr. 1) startet parallel, weil sie den Einsatz aus Phase 7 nicht braucht. **Abbruchkriterium:** Bleibt der *Sprachanteil* über 1,5, obwohl die drei größten Posten beseitigt sind, dann ist die Umgebung nicht das Problem, sondern die Fremdheit der Sprache; dann Phase 8 (Modell lernt QAL) vorziehen und Phase 7 verkleinern.

### Phase 6b — Erfahrungsgedächtnis (2–3 Wochen, nach der Token-Analyse, vor Phase 7)

**Zweck:** QAL hat ein Programmgedächtnis (`db`), ein Codegedächtnis (Store, Historie, Signaturen, Reviews) und bekommt in Phase 7 ein Modellgedächtnis (aufgezeichnete Antworten). Was fehlt, ist das Gedächtnis der Agenten: jeder beginnt bei null, und dass 27 von 33 „unknown name“-Fehlern derselbe Denkfehler sind (Token-Analyse 23.10.), weiß kein Agent des nächsten Tages. Festlegung des Auftraggebers (23.10.): das Gedächtnis darf der Sprache nicht fehlen. Es wird ohne neues Sprachmittel gebaut, im Store und über MCP.

1. **Korrekturpaare sammeln.** Ein gescheitertes `define` gefolgt von einem erfolgreichen `define` desselben Namens in derselben Sitzung ist eine Korrektur. Der Store speichert (Diagnosecode, Pfad, fehlerhafter Ausschnitt, korrigierter Ausschnitt, Hash der Definition) inhaltsadressiert und ohne Zutun des Agenten. Die 164 Benchmark-Transkripte liefern den Anfangsbestand (Werkzeug neben `bench/analyze.py`).
2. **`qal_recall`.** Zu einem Diagnosecode, einem Namen oder einem Typmuster: die häufigsten Korrekturen und die passenden gespeicherten Definitionen (`qal_query` findet heute schon nach Typmuster). Antwort klein genug für eine Runde.
3. **Diagnosen mit Gedächtnis.** Tritt ein Code auf, für den Korrekturen bekannt sind, steht der häufigste Vorschlag direkt im Hinweis der Diagnose. Das ist der Teil, der Runden spart; Phase 6 Nr. 2 (Korrekturvorschläge) nutzt ihn.
4. **Projektübergreifend, später.** Weil Definitionen und Korrekturen Hashes sind, lässt sich Erfahrung zwischen Projekten teilen, ohne Code zu kopieren; vor Phase 9 nicht nötig.

Das Erfahrungsgedächtnis ist zugleich der Rohstoff für Phase 8: ein Korpus aus Fehlern und Korrekturen ist das, womit ein Modell die Sprache lernt.

**Abnahme:** Im Benchmark-Rerun von Phase 6 erhält jede Diagnose eines Codes mit bekannten Korrekturen einen Vorschlag, und die `lang.*`-Runden je Lauf (0,52 am 23.10.) fallen messbar; `qal_recall` beantwortet die drei Fehlerklassen der Analyse (lokaler Name nach `;`, `String`, Built-in als Wert) mit der richtigen Korrektur (Test im MCP-Crate). **Abbruchkriterium:** Bringt das Gedächtnis im Rerun keine messbare Senkung der Sprachrunden, bleibt es als Korpusquelle (Phase 8) und wird nicht weiter ausgebaut.

**Stand 2026-10-23:** 1–3 gebaut (Plan `2026-10-23-qal-phase6b-erfahrungsgedaechtnis.md`: Schema 7, `qal_recall`, Hinweise mit Gedächtnis nach Code und Meldungsform, Korpus aus den Transkripten mit 10 Paaren nach dem Filter gegen Platzhalter und Umschriften, Abnahmetest der drei Fehlerklassen); 4 (projektübergreifend) offen; die Abnahme im Rerun steht aus. Befund aus dem lebenden Programm (`bench/living/results-2026-10-23/report.md`): eine unbewiesene Pflicht wurde zwölf Änderungen lang vererbt, weil kein Agent wusste, wie sie früher geschlossen wurde; genau dafür ist das Gedächtnis da.

### Phase 7 — Software für KI-Systeme, in QAL (8–12 Wochen)

**Zweck:** Das fehlende Stück von Eigenschaft 3 bauen und zugleich den ersten echten Einsatz schaffen.

1. **Effekt `model`.** Ein Host-Effekt wie `net` oder `db`: `fn ask(prompt: Str, schema: Str) -> Result[Json, ModelError] !model`. Der Host (Rust) ruft das konfigurierte Modell, prüft die Antwort gegen das Schema, zählt Kosten gegen ein Budget. Die Sprache kennt kein bestimmtes Modell; LLM oder andere Modellklasse ist Konfiguration des Hosts. **Modellzugang** (Entscheidung des Auftraggebers, 23.10.): Der Host bekommt einen API-Schlüssel über die Umgebungsvariable `ANTHROPIC_API_KEY` (abgelegt in `~/.config/qal/secrets.env`, Rechte 600, niemals im Repo) und ein Kostenlimit; der Host verweigert Aufrufe jenseits des Limits. **Limit für die Tests** (Auftraggeber, 23.10.): 20 €, abgelegt als `QAL_MODEL_BUDGET_EUR=20` neben dem Schlüssel; der Host rechnet konservativ 1 € = 1 $ und zählt die Kosten aus `usage` jeder Antwort gegen die Grenze. Mit Haiku 5.5 entspricht das grob 100 Millionen Eingabe- oder 40 Millionen Ausgabe-Tokens, also einem Vielfachen dessen, was die Auswertung der 161 Transkripte braucht. Ein neues Limit für den Dauerbetrieb wird vor Phase 7 genannt. Training in Phase 8 zunächst lokal; Cloud-Rechenzeit nur nach Rücksprache. **Modellwahl** (Vorgabe des Auftraggebers: das günstigste, das den Zweck erfüllt): Standard ist `claude-haiku-5-5` (0,10 $ Eingabe, 0,50 $ Ausgabe je Million Tokens) für Klassifikation und Auswertung, also den Benchmark-Auswerter; Eskalation auf `claude-sonnet-5-5` (2 $ / 10 $) nur dort, wo Haiku im Vergleich auf einer Stichprobe messbar schlechter urteilt oder Code erzeugen soll (Korpus in Phase 8). Jede Eskalation wird gemessen, nicht angenommen; das Ergebnis steht im Bericht der Phase.
2. **Aufzeichnung und Wiederholung.** Jeder `model`-Aufruf wird mit Eingabe-Hash und Antwort in den Store geschrieben. `qal test` und `qal run --replay` spielen aufgezeichnete Antworten ein: Tests von Orchestrierungen sind deterministisch und kosten nichts. Das ist dieselbe Idee wie der inhaltsadressierte Code, auf Modellantworten übertragen.
3. **Budget als Vertrag.** `req budget(1000)` oder ein Effektparameter mit Kostenobergrenze; der Host bricht ab, bevor das Budget überschritten wird, mit einer Diagnose, die sagt, wo es ausgegeben wurde.
4. **Orchestrierung als Programm.** Plan–Ausführen–Prüfen-Schleifen, Werkzeugauswahl, Zusammenführung mehrerer Modellantworten als QAL-Funktionen mit Verträgen über die Ergebnisse (Schema erfüllt, Länge begrenzt, keine Effekte außer den deklarierten). Standardbibliothek `std.model` (Prompt-Bausteine, Schema-Prüfung, Wiederholen mit Rückfall) von Agenten über MCP geschrieben.
5. **Der erste Einsatz: der Benchmark-Auswerter** (Entscheidung des Auftraggebers, 23.10.). Ein QAL-Programm, das Agenten-Protokolle (die 161 vorhandenen Benchmark-Transkripte und jeden künftigen Lauf) über den `model`-Effekt klassifiziert: wohin gingen die Tokens, welche Fehlerklasse, welche Korrektur. Es liest lokale Dateien (`fs`), braucht kein Netz, und seine Ergebnisse müssen wiederholbar sein, weil sie in Berichte eingehen. Es liefert zugleich die Token-Analyse aus Phase 6 und wird bei jedem Benchmark gebraucht; die Nutzung über vier Wochen ergibt sich aus den Benchmark-Wiederholungen der Phasen 6 und 8. Gebaut von Agenten über MCP, abgenommen vom Auftraggeber. Die handgeschriebene Analyse aus Phase 6 Nr. 1 ist sein Vorläufer und seine Referenz.

**Abnahme:** Der Einsatz läuft vier Wochen; mindestens ein Fehler wurde durch Typ, Effekt, Vertrag oder Wiederholung gefunden, bevor er Schaden anrichtete, und ist dokumentiert. Die Orchestrierung hat deterministische Tests mit aufgezeichneten Antworten. **Abbruchkriterium:** Findet sich kein Einsatz, den jemand vier Wochen benutzt, ist QAL ein Forschungsergebnis; dann direkt Phase 9 in der Variante „veröffentlichen und abschließen“.

### Phase 8 — Das Modell lernt die Sprache (6–10 Wochen)

**Zweck:** Den Tokenverbrauch unter den von Rust bringen und beweisen, dass die Sprache mit den Modellen wachsen kann, ohne an eines gebunden zu sein.

1. **Verifizierter Korpus.** Aufgaben generieren (aus Benchmark-Typen, Standardbibliothek und dem Einsatz aus Phase 7), Lösungen von starken Modellen schreiben lassen, mit `check`, `test`, `prove` filtern; Fehler-und-Korrektur-Spuren aufbewahren. Ziel: 5 000 bis 20 000 verifizierte Programme mit Spuren. Nur QAL kann das so billig, weil die Prüfung maschinell und vollständig ist.
2. **Feinjustierung** eines offenen Coder-Modells (LoRA) auf dem Korpus. Vergleich im Benchmark: untrainiert, trainiert, Rust.
3. **Rückkopplung in die Sprache.** Was das trainierte Modell weiterhin falsch macht, ist ein Sprach- oder Referenzproblem und geht zurück in Phase-6-Maßnahmen.
4. **Unabhängigkeit prüfen.** Ein nicht trainiertes Spitzenmodell muss weiter unter 1,2 bleiben (Anti-Ziel 1).

**Vergleichsdesign** (Festlegung des Verfassers): Das offene Modell fährt beide Arme, Rust und QAL, in einem eigenen Agenten-Harness (das heutige treibt nur Claude-Code-Subagenten; ein zweites Harness mit denselben Aufgaben und Tests ist Teil der Phase). Verglichen werden drei Paare: offenes Modell untrainiert (Rust, QAL), offenes Modell trainiert (Rust, QAL), Spitzenmodell untrainiert (Rust, QAL). Erreicht das untrainierte offene Modell in Rust keine 40/40, gilt die Korrektheit relativ: gleiche Zahl bestandener Aufgaben in beiden Armen. Training zunächst lokal (LoRA auf Apple Silicon); Cloud-Rechenzeit nur, wenn der lokale Lauf die Ersparnis andeutet.

**Abnahme:** Trainiertes offenes Modell ≤ 0,8 × seiner eigenen Rust-Tokens bei gleicher Korrektheit; Spitzenmodell untrainiert ≤ 1,2. **Abbruchkriterium:** Bringt Training weniger als 20 % Ersparnis gegenüber dem untrainierten offenen Modell in QAL, wird der Korpus veröffentlicht und Training nicht weiterverfolgt.

### Phase 9 — Verbreitung (4 Wochen, parallel ab Phase 7 möglich)

**Offen** (Entscheidung des Auftraggebers, 23.10.): Ob das Repository öffentlich wird, unter welcher Lizenz und für welche Plattformen, wird nach Phase 7 entschieden, wenn klar ist, ob es etwas zu zeigen gibt. Bis dahin gilt diese Phase als Platzhalter; nur der Bericht (Nr. 2) wird in jedem Fall geschrieben. Bei Veröffentlichung schlägt der Verfasser Apache-2.0 und macOS zuerst vor.

1. Installation in einem Schritt (Binärpaket, `qal init`, MCP-Konfiguration für gängige Agenten-Umgebungen).
2. Bericht: These, Methode, Zahlen aus allen Benchmarks, Grenzen; auf dem Niveau von `agent-advantage-study.md`, aber mit den Ergebnissen der Phasen 6–8.
3. Zwei bis drei externe Nutzer, die den Einsatz aus Phase 7 oder etwas Eigenes bauen; ihre Rückmeldung ist die letzte Korrektur der Referenz.

**Abnahme:** Ein Fremder installiert QAL, verbindet seinen Agenten und baut in einer Sitzung einen Dienst mit Verträgen, ohne Rückfrage.

## 6. Was dabei nicht weiterentwickelt wird

- Sprachmittel ohne Nachfrage aus Phase 7 (keine Traits, keine Module über Pakete hinaus, keine Nebenläufigkeit im Programm, kein Float-Prover), bis ein Einsatz sie verlangt.
- Das Team-Protokoll über das Vorhandene hinaus, bis ein Einsatz mehr als zwei Agenten gleichzeitig braucht. Der Benchmark zeigt, dass Teams heute Kosten ohne Nutzen sind.
- Das Generation-3-Substrat (`../QAL`, ARM64-JIT) als Laufzeit für QAL. Es bleibt ein eigenes Vorhaben, bis Phase 7 eine Latenzanforderung hat, die wasmtime nicht erfüllt.

## 7. Arbeitsweise

- **Eine Roadmap, eine Hand.** Zwei Sitzungen haben am 22.10. dieselbe Funktion parallel gebaut. Ab jetzt steht jedes Arbeitspaket vor Beginn in `2026-10-15-qal-roadmap.md` mit Besitzer und Zustand; eine Sitzung prüft die Tabelle und `git status`, bevor sie anfängt.
- **Behauptung = Test.** Jede neue Zeile in der Garantie-Matrix (Roadmap §6.5) nennt den Test, der sie belegt; `checkrefs.py`-artige Prüfung in der CI.
- **Messen vor Bauen.** Jede Phase beginnt mit der Messung, die ihre Abnahme später wiederholt.
- **Agenten bauen QAL-Teile in QAL.** Standardbibliothek, Rezepte, `std.model`, Korpus-Lösungen: über MCP, nicht von Hand. Jeder dieser Aufträge ist zugleich ein Datenpunkt für Phase 6.

## 8. Was dem Bauenden überlassen bleibt

- Die konkrete Form des `model`-Effekts (eine Funktion mit Schema oder mehrere spezialisierte), die Aufzeichnungsgranularität und das Budgetmodell. Vorgabe ist nur: Host-Effekt, Aufzeichnung im Store, Wiederholung in Tests, Abbruch vor Überschreitung.
- Welche Modelle der Host anbindet. Die Sprache bleibt modellneutral.
- Die Wahl des offenen Modells und des Trainingsverfahrens in Phase 8.
- Die Werkzeuge der Token-Analyse, solange sie je Lauf die genannten Posten trennen.

## 9. Entscheidungen des Verfassers

Diese Punkte hat der Auftraggeber nicht vorgegeben; sie sind Festlegungen dieses Dokuments und können geändert werden:

- Phase 6 vor Phase 7, obwohl Phase 7 das sichtbare neue Stück ist: ohne billigen Kreis vererbt sich der Tokenfaktor in jeden Einsatz.
- Der erste Einsatz muss etwas sein, das der Auftraggeber selbst benutzt. Fremde Nutzer kommen erst in Phase 9.
- Modellaufrufe als Effekt mit Aufzeichnung statt als eingebaute Sprachkonstrukte. Das hält die Sprache klein und die Garantien einheitlich.
- Training erst nach Einsatz und nur mit messbarer Ersparnis; Abbruchkriterium 20 %.
- Schwellen 1,2 und 0,8: gewählt, nicht abgeleitet. 1,2 heißt „der Unterschied ist Rauschen“, 0,8 heißt „die Sprache ist ein Vorteil“.
