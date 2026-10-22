# Empirische Untersuchung des behaupteten Agentenvorteils von QAL

**Datum:** 2026-10-20  
**Gegenstand:** Empirische Auswertung der Benchmark- und Teamergebnisse aus Phase 1e und Phase 4d  
**Autoren/Messung:** Unabhängige Claude-Code Subagenten-Läufe (Modell: Sonnet), gemessen über automatisierte Test-Harnesses (`bench/run.py`, `bench/report.py`, `bench/team/real/score.py`)

---

## 1. Fragestellung und Hypothese

Zu Beginn des Projekts stand die Hypothese im Raum:
> *„Ein auf AST-Definitionen, Inhaltsadressierung, Effekttrennung und MCP-Tools zugeschnittenes System (QAL) verschafft autonomen KI-Agenten signifikante Vorteile gegenüber traditionellen, dateibasierten Sprachen (wie Rust + Cargo) bezüglich Lösungsrate, Tokenverbrauch und Koordinationsaufwand.“*

Zur Überprüfung dieser Hypothese wurden zwei kontrollierte empirische Studien durchgeführt:
1. **Studie 1 (Phase 1e):** 20 repräsentative Einzelprogrammierungs-Aufgaben im direkten Vergleich QAL vs. Rust.
2. **Studie 2 (Phase 4d):** Parallelisierte Multi-Agenten-Entwicklung (6 Agenten mit Rollenmodell) vs. Solo-Agenten-Entwicklung eines Backendservices.

---

## 2. Methodik & Versuchsaufbau

### 2.1 Repräsentative Aufgaben (Phase 1e)
Der Aufgabenkatalog umfasst 20 isolierte CLI- und Datenverarbeitungs-Aufgaben (`bench/tasks/`):
- `01-word-count`: Wort- und Zeilenzählung.
- `02-roman`: Römische Zahlenkonvertierung.
- `03-csv-sum`: Parsen und Aggregieren von CSV-Dateien.
- `04-binary-search`: Binäre Suche mit Randbedingungen.
- `05-brackets`: Klammerungs-Validierung mittels Stack.
- `06-bank`: Kontoführung mit Saldenprüfung.
- `07-tokenizer`: Lexikalischer Scanner für Arithmetik.
- `08-expr-eval`: Rekursiver Parser und Evaluator für Ausdrücke.
- `09-rle`: Run-Length-Encoding und Dekodierung.
- `10-file-stats`: Statistische Kennzahlen über Dateiinhalte.
- `11-prime-factors`: Primfaktorzerlegung.
- `12-sort-words`: Fallunabhängiges Sortieren von Wörterlisten.
- `13-word-freq`: Frequenzanalyse von Textkorpora.
- `14-bst`: Binärer Suchbaum mit Einfügen und Traversierung.
- `15-caesar`: Cäsar-Verschlüsselung.
- `16-anagrams`: Gruppierung von Anagrammen.
- `17-word-wrap`: Zeilenumbruch-Algorithmus nach Wortgrenzen.
- `18-date-diff`: Datumsarithmetik ohne externe Bibliotheken.
- `19-base-convert`: Zahlensystem-Transformation (Basis 2 bis 36).
- `20-ini-query`: Konfigurationsdateien parsen und Schlüssel abfragen.

### 2.2 Kontrollbedingungen
- **Modell & Budget:** Identisches LLM (Claude 3.5 Sonnet) für beide Arme; maximal 3 Versuche je Aufgabe; maximal 60 Tool-Aufrufe.
- **Randomisierung:** 80 Durchläufe (40 QAL, 40 Rust, jeweils 2 Durchläufe pro Aufgabe) in randomisierter Reihenfolge gemäß `bench/schedule.json` (Seed: `qal-1e`).
- **Versteckte Tests:** Die Abnahme erfolgte ausschließlich über separate, dem Modell zur Entwicklungszeit nicht zugängliche Testfälle (`bench/tasks/*/tests.json`).
- **Erfasste Metriken:** Erfolgreiche Lösungen (alle Hidden Tests grün), Tokenverbrauch (Input, Output, Cache-Reads), Wandzeit, Anzahl Tool-Aufrufe, Anzahl gescheiterter Tool-Aufrufe.

---

## 3. Empirische Ergebnisse Phase 1e (Einzel-Agent: QAL vs. Rust)

### 3.1 Gesamtergebnis

| Metrik | QAL (MCP) | Rust (Dateien + Cargo) | Differenz / Sieger |
|---|---|---|---|
| **Gültige Läufe** | 40 | 40 | Identisch |
| **Erfolgreich gelöst (alle Hidden Tests)** | **39 / 40 (97,5 %)** | **40 / 40 (100,0 %)** | Rust (+2,5 %) |
| **Im ersten Versuch gelöst** | 39 / 40 | 39 / 40 | Unentschieden |
| **Gesamte Token (Dispatch)** | 2 709,8 k | 2 402,9 k | Rust (-11,3 %) |
| **Token pro verifizierter Lösung** | **69,5 k** | **60,1 k** | **Rust gewinnt (-13,5 %)** |
| **Verarbeitete Token (inkl. Cache)** | 9 598,3 k | 5 058,0 k | Rust (-47,3 %) |
| **Tool-Aufrufe gesamt** | 143 | 112 | Rust (-21,7 %) |
| **Fehlgeschlagene Tool-Aufrufe** | 38 | 0 | Rust (0 Fehler) |
| **Wandzeit (Minuten)** | 26,3 min | 14,9 min | Rust (-43,3 %) |
| **Direkte Aufgaben-Siege** | 0 | **12** (8 Unentschieden) | **Rust gewinnt deutlich** |

### 3.2 Detailanalyse der Einzelergebnisse
- **Rust gewann 12 Aufgaben:** `02-roman`, `03-csv-sum`, `06-bank`, `08-expr-eval`, `09-rle`, `11-prime-factors`, `12-sort-words`, `13-word-freq`, `14-bst`, `17-word-wrap`, `18-date-diff`, `20-ini-query`.
- **8 Aufgaben endeten unentschieden:** `01-word-count`, `04-binary-search`, `05-brackets`, `07-tokenizer`, `10-file-stats`, `15-caesar`, `16-anagrams`, `19-base-convert`.
- **QAL gewann 0 Aufgaben.**

### 3.3 Gründe für den Rust-Vorteil bei Einzelaufgaben
1. **Pretraining-Asymmetrie:** Das LLM hat Milliarden Zeilen Rust-Code in seinen Trainingsgewichten gesehen, aber vor dem Prompt null Zeilen QAL. Der QAL-Arm musste die Sprache ad-hoc über die Tool-Beschreibungen (Sprachreferenz im MCP-Prompt, ca. 11 KB) erlernen.
2. **Kompilier- und Diagnosezyklen:** In QAL führten Syntaxfehler oder falsche Built-in-Namen zu 38 fehlgeschlagenen Tool-Aufrufen, während der Agent in Rust mit gewohnter Syntax fehlerfrei im ersten Anlauf kompilierte.

---

## 4. Empirische Ergebnisse Phase 4d (Team- vs. Solo-Agent)

In Phase 4d wurde die These untersucht, ob parallele Agenten-Teams mit Rollenmodell (Architekt, Implementierer, Verifizierer, Reviewer) komplexe Backendservices schneller oder besser bauen als ein Solo-Agent.

### 4.1 Messergebnisse des realen Laufs (`bench/team/real/report.md`)

| Versuchsarm | Agenten | Wandzeit | Token (Summe) | Tool-Aufrufe | Gelöste Szenarien | `qal verify` |
|---|---|---|---|---|---|---|
| **Multi-Agenten-Team** | 6 (Architekt, 3 Impl, Verifizierer, Reviewer) | 765 s | 632 k | 154 | **20 / 20** (15 hidden) | **ok** |
| **Solo-Agent** | 1 | 123 s | 101 k | 13 | **20 / 20** (15 hidden) | — |

### 4.2 Bewertung des Team-Experiments
- **Funktionale Korrektheit:** Beide Ansätze lösten alle 20 Szenarien (100 %) fehlerfrei.
- **Effizienz:** Der Solo-Agent benötigte nur **ca. 1/6 der Token** und **ca. 1/6 der Zeit**.
- **Ursache:** Bei Aufgaben, die in das Context Window eines einzelnen Modells passen, erzeugt die Multi-Agenten-Koordination (Warten auf Freigaben, Review-Schleifen, Change-Set-Aufteilung, Benachrichtigungen) massiven Overhead.

---

## 5. Wissenschaftliche Synthese: Wo QAL tatsächlich einen Vorteil bietet

Die Hypothese eines *pauschalen Token- oder Geschwindigkeitsvorteils* von QAL bei kleinen isolierten Aufgaben ist **empirisch widerlegt**.

QAL bietet jedoch nachweisbare, distinkte Vorteile in spezifischen architektonischen Bereichen:

### 5.1 Reale Vorteile von QAL
1. **Deterministischer 3-Wege-AST-Merge:**
   - Wenn mehrere Agenten parallel disjunkte Definitionen modifizieren, fusioniert QAL die Änderungen auf AST-Ebene konfliktfrei. Textbasierte Zeilenkonflikte (z. B. zwei Funktionen am Ende derselben Datei) treten prinzipbedingt nicht auf.
2. **Formale Korrektheit & Invariantenprüfung (`req`/`ens`):**
   - Verträge werden über Z3 statisch bewiesen (`proven`). Logische Widersprüche (z. B. Division durch Null, Out-of-Bounds) werden vor dem Ausführen gemeldet, anstatt erst in Laufzeit-Crashes aufzutreten.
3. **Integrität & Auditierbarkeit (Ed25519-Signaturkette):**
   - Jede Definition und jeder Merge ist fälschungssicher einem kryptografischen Agenten-Schlüssel zugeordnet (`qal verify`). Für Sicherheitsaudits und Compliance im Agentenbereich ist dies ein wesentlicher Vorteil gegenüber unkontrollierten Dateiedits.
4. **Feingranulare MCP-Schnittstelle statt Dateidumping:**
   - Große Codebasen müssen nicht in den Kontext geladen werden; Agenten fragen gezielt `qal_view` oder `qal_query` für einzelne Signaturen ab.

### 5.2 Grenzen und Einschränkungen
- Bei einfachen, monolithischen Aufgaben oder kleinen Programmen übertrifft die Vertrautheit des Modells mit Standard-Programmiersprachen (Rust/Python) jeden tooling-seitigen Abstraktionsvorteil.
- Der Team-Vorteil von QAL greift erst bei Codebasen, deren Umfang das Token-Limit eines einzelnen Kontextfensters übersteigt.
