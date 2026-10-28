# Grenzen & Nicht-Ziele von QAL (QLANG Agent Language)

> [!IMPORTANT]
> **Ehrliche Architekturgrenzen:** QAL ist kein Allzweck-Ersatz für Rust, Python, Go oder C++. Es ist eine spezialisierte, verifizierbare Kontroll- und Regelschicht für autonome KI-Agenten in sicherheitskritischen Domänen. Wer QAL wie Python schreibt oder als Allzweck-Sprache einsetzt, scheitert an den Sprachgrenzen.

---

## 1. Keine imperativen Schleifen und keine Mutation (No Loops, No Mutation)

- **Das Prinzip:** QAL besitzt weder `while`-, `for`- noch `loop`-Schleifen und keine veränderlichen Variablen (`mut`, Zuweisungen). Jeder Zustand ist unveränderlich (*immutable*).
- **Warum diese Grenze existiert:**
  1. Imperative Schleifen machen formale Beweise (Satz von Rice, Halteproblem) unvollständig. Für Schleifen müsste der SMT-Solver Schleifeninvarianten synthetisieren – ein unentscheidbares Problem.
  2. Reine funktionale Ausdrücke sind referentiell transparent und deterministisch.
- **Konsequenz für Entwickler & Agenten:**
  - Iteration erfolgt ausschließlich über **Rekursion** oder eingebaute Kombinatoren: [`fold`](../../qal/crates/qal-check/src/builtins.rs), [`map`](../../qal/crates/qal-check/src/builtins.rs), [`filter`](../../qal/crates/qal-check/src/builtins.rs).
  - Listen sind persistent. Wiederholtes [`push`](../../qal/crates/qal-check/src/builtins.rs) in einer naiven Rekursion erzeugt quadratische Laufzeit $\mathcal{O}(n^2)$. Für lineare Akkumulation muss der Präludium-Typ [`Accum[T]`](../../qal/crates/qal-core/src/prelude.rs) mit [`acc_list`](../../qal/crates/qal-check/src/builtins.rs) genutzt werden.
  - Quellcode für Parser oder String-Transformationen fällt ca. 20–30 % länger aus als in Python.

---

## 2. Striktes 256-MiB-Gast-Speicherlimit (Sandbox Budget)

- **Das Prinzip:** Jeder QAL-Service läuft in einer isolierten WebAssembly-Instanz ([wasmtime](../../qal/crates/qal-run/src/lib.rs)) mit Pooling-Allocator.
- **Budgetgrenzen:**
  - Standard-Speicher pro Request: **64 MiB**.
  - Maximal konfigurierbarer Speicher: **256 MiB**.
  - Maximale SQL-Textgröße: **64 KiB**.
- **Nicht-Ziele:**
  - QAL ist **nicht** für Big-Data-In-Memory-Analysen, Vektor-Embeddings oder das Laden großer ML-Gewichte geeignet.
  - Datenmengen, die nicht in 256 MiB passen, gehören in die Datenbank (SQLite / persistenter Store) und müssen seitenweise gestreamt werden.

---

## 3. Prover-Grenzbereich & `unknown`-Quote (Prover Incompleteness)

- **Das Prinzip:** Der Kontraktprüfer [`qal_prove`](../../qal/crates/qal-prove/src/lib.rs) übersetzt Vor- und Nachbedingungen ([`req`](../../qal/crates/qal-core/src/ast.rs) / [`ens`](../../qal/crates/qal-core/src/ast.rs)) in Verifikationsbedingungen für den SMT-Solver Z3.
- **Was vollständig bewiesen wird:**
  - Quantorenfreie lineare Ganzzahlarithmetik (**QF_LIA**): Addition, Subtraktion, Multiplikation mit Konstanten, lineare Ungleichungen ($<, \le, >, \ge, ==$).
  - Kontrakt-Implikationen an Aufrufstellen (*callsite preconditions*).
- **Wo Z3 `unknown` liefert:**
  1. **Nichtlineare Arithmetik:** Multiplikation zweier Variablen ($x \cdot y$), Division durch Variablen, Modulo mit variablen Operanden.
  2. **String-Semantik:** Ungleichheit (`!=`) oder Substring-Operationen auf `Str`.
  3. **Effekt-behaftete Funktionen:** Funktionen mit `!db` oder `!io` werden an Aufrufstellen abstrahiert; Z3 führt keine I/O aus.
- **Verhalten bei `unknown`:**
  - Liefert der Prover `unknown`, wird die Definition im Store gespeichert (mit Warnung `E0802`).
  - Der Kontrakt wird nicht gelöscht, sondern **zur Laufzeit dynamisch geprüft**.
  - Refutierte Kontrakte (`refuted` mit konkretem Gegenbeispiel) werden rigoros abgewiesen (`E0801`).

---

## 4. Multi-Agenten-Protokoll-Overhead (Team Token Cost)

- **Das Prinzip:** Das 4-Rollen-Protokoll (`architect`, `implementer`, `verifier`, `reviewer`) erzwingt strikte Aufgabenteilung und kryptographische Ed25519-Freigaben für jeden Merge nach `main`.
- **Die Kosten:**
  - Für eine typische Aufgabe benötigt ein 3- bis 4-Agenten-Team ca. **60k–100k Tokens** (im Vergleich zu ~20k–30k Tokens für einen einzelnen Solo-Agenten).
  - Das Erstellen von Change Sets, Einholen von Reviews und Fast-Forward-Mergen erzeugt Kommunikationsrunden.
- **Richtlinie:**
  - QAL-Teams sind **nicht** für schnelle Ad-hoc-Skripte oder triviale CRUD-Apps gedacht.
  - Der Team-Modus rechtfertigt sich dort, wo ein einziger entkommener Defekt (z. B. unberechtigte Kontobuchung, Datenverlust) um Größenordnungen teurer ist als 100k LLM-Tokens.

---

## 5. Rollen sind Leitplanken, keine OS-Sicherheitsgrenzen (Threat Model)

- **Das Prinzip:**
  ```text
  roles are guard rails for cooperative agents, not a security boundary:
  every agent runs as the same OS user and could read the key files;
  the signed records make every change of the store evident (`qal verify --trust team.json`)
  ```
- **Grenzen der Schutzwirkung:**
  - Rollen verhindern, dass ein halluzinierender oder übergriffiger Agent unberechtigte Definitionsänderungen vornimmt, unvollständigen Code mergt oder Tests überspringt.
  - Sie schützen **nicht** vor einem aktiv bösartigen Angreifer mit vollständigem Host-Shell-Zugriff, der die geheimen Schlüsseldateien aus `~/.config/qal/keys` stiehlt.
  - Die Sicherheitsgarantie von QAL ist **Audit-Evidenz** (*tamper-evident audit chain*): Jeder unerlaubte Zugriff oder manipulierte Commit bricht die kryptographische Kette und wird durch `qal verify --head <hash>` sofort erkannt.

---

## 6. Zusammenfassender Vergleich: Wann QAL Sinn macht

| Anwendungsfall | Empfehlung | Begründung |
|---|---|---|
| **Finanz-Escrow / Payouts / Multi-Sig** | ✅ **Optimal für QAL** | Erhaltungssatz wird formal bewiesen; automatischer DB-Rollback bei HTTP $\ge 400$. |
| **Zugriffskontroll- & Compliance-Regeln** | ✅ **Optimal für QAL** | Trennung von Ersteller und Freigeber ist signiert und revisionssicher. |
| **Allgemeine Webanwendung / Blog / Shop-Frontend** | ❌ **Rust / Go / Python** | Unnötiger Beweisaufwand; keine Schleifen verlangsamen simple UI-Entwicklung. |
| **Machine Learning / Model Training / Data Science** | ❌ **Python / C++** | 256-MiB-Limit und fehlende Matrix-Beschleuniger schließen ML-Training aus. |
