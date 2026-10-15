# QAL — QLANG Agent Language: Design-Spezifikation

**Datum:** 2026-10-15
**Status:** Entwurf zur Prüfung
**Scope dieses Dokuments:** Gesamtvision (kurz) + Phase 1 (verbindlich, implementierbar)

---

## 1. Ziel und These

**These:** Programmiersprachen sind für Menschen gebaut. Eine Sprache, die für KI-Agenten gebaut ist, macht Agenten beim Bauen von Software messbar effektiver.

**Effektivität wird gemessen** als *Tokens und Iterationen pro verifiziert korrekter Lösung*, verglichen mit demselben Agenten in einer bestehenden Sprache. Menschliche Lesbarkeit ist kein Ziel.

**Was an QAL neu ist** (in keiner existierenden Sprache in dieser Kombination):

1. Code ist kein Text, sondern ein **inhaltsadressierter, geprüfter Programmgraph**.
2. Agenten arbeiten über **Operationen** (MCP-Tools), nicht über Dateien und Textdiffs.
3. **Nur Geprüftes existiert:** Eine Definition, die nicht typ- und effektkorrekt ist, kann nicht gespeichert werden.
4. **Effekte sind Berechtigungen:** Der Typ einer Funktion bestimmt, welche Systemfähigkeiten sie zur Laufzeit erhält.
5. **Contracts sind Teil der Signatur** (`req`/`ens`) und werden geprüft (Phase 1: zur Laufzeit, Phase 2: per Beweis).
6. **Fehler sind Daten:** Jede Diagnose ist strukturiertes JSON mit Ort, Erwartung, Befund und Korrekturhinweisen.

**Fundament (bewusst bestehend):** WebAssembly als Ausführungsformat, wasmtime als Laufzeit, später Z3 als Beweiser. Rust ist ausschließlich das Bau-Gerüst des Compilers, bis QAL sich selbst kompiliert (Phase 5).

## 2. Phasen

| Phase | Inhalt | Ergebnis |
|---|---|---|
| **1 — Kern** | Graph, Textform, Typ- & Effektprüfung, Laufzeit-Contracts, WASM-Codegen, Laufzeit mit Berechtigungen, MCP-Server, CLI, Benchmark | Ein Agent baut und startet CLI-Programme ausschließlich über QAL-Operationen |
| 2 — Beweise | Z3-Beweis von `req`/`ens`, generierte Property-Tests, Gegenbeispiele, Generics, Closures | Contracts werden statisch bewiesen |
| 3 — Standardbibliothek | Effekte `net`, `db`; JSON, HTTP-Routing, Daten — **geschrieben von Agenten in QAL** | Backend-Services |
| 4 — Team | Rollen (Architekt, Implementierer, Verifizierer, Reviewer), Sperren pro Definition, Merge, Signaturen; Wiederverwendung von `qlang-runtime` Provider-Registry/Orchestrator | Mehrere Agenten bauen parallel |
| 5 — Self-Hosting | QAL-Compiler in QAL | Rust entfällt |

Nur Phase 1 ist Gegenstand des folgenden Implementierungsplans. Phasen 2–5 bekommen jeweils eine eigene Spezifikation.

---

## 3. Phase 1 — Sprache

### 3.1 Typen

| Typ | Bedeutung | Repräsentation (WASM) |
|---|---|---|
| `Int` | 64-bit Ganzzahl, **Überlauf = Trap** | `i64` |
| `Float` | 64-bit IEEE | `f64` |
| `Bool` | Wahrheitswert | `i32` (0/1) |
| `Unit` | kein Wert | — |
| `Str` | unveränderlicher UTF-8-String | Zeiger auf `[len: i32][bytes]` |
| `List[T]` | unveränderliche Liste | Zeiger auf `[len: i32][elem…]` |
| `Option[T]`, `Result[T,E]` | eingebaute Summentypen | wie Enums |
| `{a: T, …}` | Record (nominal über `type`) | Zeiger auf Feldblock |
| `A \| B(T) \| …` | Enum mit Payload | Zeiger auf `[tag: i32][payload]` |
| `fn(T…) -> R !e` | Funktionsreferenz (ohne Capture) | Tabellenindex `i32` |

- Phase 1 hat **keine benutzerdefinierten Generics** und **keine Closures**; generisch sind nur die eingebauten `List`, `Option`, `Result`.
- Kein `null`, keine impliziten Konversionen, keine Ausnahmen. Fehler werden über `Result` behandelt.
- Top-Level-Definitionen brauchen vollständige Signaturen; innerhalb von Körpern werden Typen lokal inferiert.

### 3.2 Effekte

Jede Funktion deklariert ihre Effektmenge. Fehlt `!…`, ist sie `pure`.

| Effekt | Erlaubt (Phase 1) |
|---|---|
| (keiner) | nur Berechnung |
| `io` | stdout schreiben, stdin lesen |
| `fs` | Dateien lesen/schreiben (nur in freigegebenen Verzeichnissen) |
| `time` | aktuelle Zeit |
| `rand` | Zufallszahlen |

**Regel:** Effekte(Aufgerufene) ⊆ Effekte(Aufrufer). Verletzung ist ein Prüffehler, kein Laufzeitfehler.

**Berechtigungsmodell:** Das erzeugte WASM-Modul importiert ausschließlich die Host-Funktionen der Effekte, die der Einstiegspunkt deklariert. Die Laufzeit verweigert den Start, wenn der Aufrufer diese Effekte nicht freigibt (`--allow io,fs:./data`). Eine `pure`-Funktion hat physisch keinen Zugriff auf das System.

### 3.3 Contracts

```
fn f(x: Int) -> Int
  req x >= 0
  ens r >= x
= …
```

- `req`: Vorbedingung über Parameter. `ens`: Nachbedingung, Ergebnis heißt `r`.
- Contracts sind reine Ausdrücke (keine Effekte).
- **Phase 1:** werden als Laufzeitprüfungen kompiliert; Verletzung = Trap mit strukturiertem Fehler (Definition, Contract, Werte).
- Contracts sind Teil des Hashes einer Definition.

### 3.4 Ausdrücke

Ausdrucksorientiert, keine Anweisungen:
`let`-Bindung, `if … then … else …`, `match` mit Mustern (Literale, Enum-Varianten, Records, `_`, Guards `if`), Funktionsaufruf, Feldzugriff `a.b`, Record-Konstruktion mit Typnamen `User { id: 1, name: "a" }`, Sequenz `{ a; b }` (ohne Typnamen und ohne `feld:`), Listenliteral `[a, b]`, Operatoren `+ - * / % == != < <= > >= && || ! ++` (Str- und List-Verkettung), Rekursion (inkl. Endrekursion → Schleife).

### 3.5 Eingebaute Funktionen (Phase 1)

- **Str:** `len`, `slice`, `split`, `join`, `trim`, `contains`, `starts_with`, `to_int → Option[Int]`, `to_float → Option[Float]`, `from_int`, `from_float`, `chars → List[Str]`
- **List:** `len`, `get → Option[T]`, `push`, `concat`, `map`, `filter`, `fold`, `range`, `reverse`, `sort_int`
- **Effekte:** `print(Str) !io`, `read_line() -> Option[Str] !io`, `read_file(Str) -> Result[Str,Str] !fs`, `write_file(Str,Str) -> Result[Unit,Str] !fs`, `now_ms() -> Int !time`, `rand_int(Int,Int) -> Int !rand`

### 3.6 Definitionsarten

- `type` — Record oder Enum
- `fn` — Funktion
- `test` — benannter Test: reiner Ausdruck vom Typ `Bool`, wird bei `qal_test` ausgeführt
- Einstiegspunkt: eine Funktion `main(args: List[Str]) -> Int` mit beliebigen Effekten; Rückgabe = Exit-Code

---

## 4. Phase 1 — Textform (für LLMs)

Die Textform ist eine **Sicht** auf den Graphen, nicht seine Quelle. Sie ist token-sparsam, nutzt sprechende Namen und kurze, in gängigen Tokenizern einzeln kodierte Schlüsselwörter (`fn type test let if then else match req ens`). Einrückung ist bedeutungslos; der Drucker erzeugt eine kanonische Form.

```
type Err = Empty | NotNumber(Str) | Negative(Int)

fn parse_age(s: Str) -> Result[Int, Err]
  ens match r { Ok(n) => n >= 0, _ => true }
= if len(trim(s)) == 0 then Err(Empty)
  else match to_int(trim(s)) {
    None => Err(NotNumber(s)),
    Some(n) if n < 0 => Err(Negative(n)),
    Some(n) => Ok(n),
  }

test parse_age_ok = parse_age(" 42 ") == Ok(42)
test parse_age_neg = parse_age("-1") == Err(Negative(-1))

fn main(args: List[Str]) -> Int !io
= match get(args, 0) {
    None => { print("usage: age <n>"); 2 },
    Some(a) => match parse_age(a) {
      Ok(n) => { print("ok " ++ from_int(n)); 0 },
      Err(_) => { print("invalid"); 1 },
    },
  }
```

(`{ a; b }` ist ein Sequenzausdruck: wertet `a` aus, verwirft dessen `Unit`, liefert `b`.)

**Rundlauf-Garantie:** `parse(print(g)) == g` für jeden gespeicherten Graphen.

---

## 5. Phase 1 — Programmgraph und Hashing

- Jede Definition wird nach dem Parsen in einen **kanonischen AST** überführt:
  - lokale Variablen → De-Bruijn-Indizes (Namen entfallen),
  - Verweise auf andere Definitionen → deren Hash,
  - Feldnamen und Variantennamen bleiben (sie sind Semantik).
- **Hash = SHA-256 der kanonischen Binärkodierung.** Konsequenzen:
  - Umbenennen ändert keinen Hash.
  - Identische Logik existiert genau einmal.
  - Abhängigkeiten sind unveränderlich adressiert.
- Lesbare Namen liegen in einer **Namenstabelle** (`name → hash`), getrennt vom Graphen. Lokale Variablennamen werden als Metadaten zum Hash gespeichert, damit die Textsicht sprechende Namen zeigt.
- Rekursive Gruppen (gegenseitige Rekursion) werden als ein Zyklus gemeinsam gehasht.

## 6. Phase 1 — Speicher (Codebase)

SQLite-Datei `.qal/store.db` pro Projekt.

| Tabelle | Inhalt |
|---|---|
| `defs` | `hash`, `kind`, `canonical` (Bytes), `signature`, `effects`, `local_names`, `created_at` |
| `deps` | `hash` → `dep_hash` |
| `names` | `name`, `hash`, `updated_at` |
| `history` | `name`, `old_hash`, `new_hash`, `agent`, `at` |

**Invarianten:**
1. Eine Definition wird nur gespeichert, wenn Parsen, Namensauflösung, Typ- und Effektprüfung fehlerfrei sind.
2. Alle Abhängigkeiten einer gespeicherten Definition sind gespeichert.
3. Speichern ist atomar pro `define`-Aufruf (alle Definitionen oder keine).

**Ersetzen:** `define` mit einem existierenden Namen erzeugt einen neuen Hash. Ist die Signatur (Typ + Effekte) unverändert, werden alle Abhängigen automatisch neu verknüpft und neu geprüft. Ändert sich die Signatur, schlägt `define` mit der Liste betroffener Abhängiger fehl, außer der Aufruf enthält deren angepasste Versionen.

---

## 7. Phase 1 — Prüfung und Diagnosen

Pipeline: **parse → auflösen → Typen → Effekte → speichern**. Contracts werden dabei wie Körper typgeprüft (müssen `Bool` und effektfrei sein); ihre Auswertung baut in Phase 1 die Codegenerierung als Laufzeitprüfung ein.

Jede Diagnose:

```json
{
  "code": "E0203",
  "kind": "type_mismatch",
  "def": "parse_age",
  "path": "body.else.match.arm[1].guard",
  "message": "Int erwartet, Str gefunden",
  "expected": "Int",
  "found": "Str",
  "hints": ["to_int(s) liefert Option[Int]; Wert per match entpacken"]
}
```

- `path` adressiert den Knoten im AST (stabil, unabhängig von Formatierung).
- Diagnosekategorien Phase 1: `parse`, `unknown_name`, `type_mismatch`, `effect_violation`, `non_exhaustive_match`, `signature_change`, `missing_dep`.
- Laufzeitfehler (Trap) haben dasselbe Format mit `kind` ∈ `contract_violation`, `int_overflow`, `div_by_zero`, `out_of_memory`, `index` sowie den beteiligten Werten.

---

## 8. Phase 1 — WASM-Codegenerierung und Laufzeit

- Codegen erzeugt direkt WASM-Binärcode (Crate `wasm-encoder`), kein Zwischenschritt über eine andere Sprache.
- **Speicher:** Bump-Allocator im linearen Speicher, kein Freigeben; Standardlimit 256 MB pro Ausführung, Überschreitung = `out_of_memory`. Ausreichend für CLI-Programme; Speicherverwaltung für langlaufende Services ist Teil von Phase 3.
- Endrekursion wird zu Schleifen; sonst normale WASM-Aufrufe (Stack-Limit der Laufzeit).
- **Host-Schnittstelle:** eigenes Importmodul `qal_host` mit je einer Funktionsgruppe pro Effekt. Keine WASI-Abhängigkeit in Phase 1 (bewusst: volle Kontrolle über das Berechtigungsmodell).
- **Laufzeit** (`qal-run`): bettet wasmtime ein, stellt nur die freigegebenen Effektgruppen bereit, beschränkt `fs` auf freigegebene Pfade, liefert Ergebnis/Fehler als JSON.
- `qal build <name>` schreibt ein eigenständiges `.wasm` plus Manifest (benötigte Effekte, Hash).

---

## 9. Phase 1 — Agenten-Schnittstelle (MCP)

`qal mcp` startet einen MCP-Server über stdio. Alle Antworten sind JSON.

| Tool | Eingabe | Ausgabe |
|---|---|---|
| `qal_define` | Textform mit 1..n Definitionen | Hashes **oder** Diagnosen (nichts gespeichert) |
| `qal_view` | Name(n) | Textform + Signaturen der direkten Abhängigkeiten |
| `qal_query` | `name` (Präfix), `type` (Signaturmuster), `uses` (Hash/Name) | Liste `{name, hash, signature, effects}` |
| `qal_test` | Name(n) oder alle | je Test `pass`/`fail` + Fehler |
| `qal_run` | Einstiegspunkt, `args`, `allow` (Effekte/Pfade) | `stdout`, `exit_code`, Fehler-JSON |
| `qal_build` | Einstiegspunkt | Pfad zu `.wasm` + Manifest |
| `qal_rename` | alt, neu | ok (kein Hash ändert sich) |
| `qal_history` | Name | Versionen |

Die Werkzeugbeschreibungen enthalten eine kompakte Sprachreferenz (≤ 1 Seite), damit ein Agent ohne Vorwissen produktiv ist.

Dieselben Operationen gibt es als CLI: `qal init | define <datei> | view | query | test | run | build | rename | history | mcp`.

---

## 10. Phase 1 — Projektstruktur

Eigener Cargo-Workspace `qland/qal/` (unabhängig vom LLVM-lastigen qlang-Build):

| Crate | Verantwortung | Hängt ab von |
|---|---|---|
| `qal-core` | AST, Typen, kanonische Form, Hashing, Typ- & Effektprüfung, Diagnosen | — |
| `qal-text` | Lexer, Parser, kanonischer Drucker | `qal-core` |
| `qal-store` | SQLite-Speicher, Namen, Ersetzen/Neuverknüpfen | `qal-core` |
| `qal-wasm` | Codegenerierung, Contracts als Prüfungen, Allocator | `qal-core` |
| `qal-run` | wasmtime-Host, Berechtigungen, Fehler-JSON | `qal-wasm` |
| `qal-cli` | CLI + MCP-Server | alle |
| `bench/` | Benchmark-Aufgaben und Auswertung | `qal-cli` |

## 11. Phase 1 — Benchmark

- **20 Aufgaben** im CLI-Format (u. a. Wortzählung, römische Zahlen, CSV-Summe, binäre Suche, Klammerprüfung, Zustandsautomat Bankkonto, Tokenizer für Rechenausdrücke, Auswerter, Run-Length-Encoding, Datei-Statistik).
- Jede Aufgabe: Beschreibung, versteckte Ein-/Ausgabe-Tests.
- Zwei Durchläufe pro Aufgabe mit demselben Modell und derselben Tool-Schleife: **(a)** QAL über MCP, **(b)** Rust über Dateischreiben + `cargo run`.
- Gemessen: Erfolg (versteckte Tests grün), Eingabe-/Ausgabe-Tokens, Anzahl Tool-Aufrufe, Anzahl Fehlversuche, Wandzeit.
- Die Agenten-Schleife nutzt die Claude API (`ANTHROPIC_API_KEY` erforderlich).
- Ergebnis: `bench/report.md` mit Tabelle pro Aufgabe und Summen.

---

## 12. Erfolgskriterien Phase 1

1. Alle 20 Benchmark-Aufgaben sind in QAL lösbar, nachgewiesen durch Referenzlösungen in `bench/solutions/`, die ihre versteckten Tests bestehen.
2. Ein Agent kann ausschließlich über die MCP-Tools Programme definieren, testen und ausführen.
3. Rundlauf `parse(print(g)) == g` gilt für alle Referenzlösungen.
4. Umbenennen ändert keine Hashes; identische Definitionen unter zwei Namen haben denselben Hash.
5. Eine `pure`-Funktion bzw. ein Programm ohne `fs` kann nachweislich nicht auf Dateien zugreifen (Test: das Modul importiert die Funktion nicht; Start ohne Freigabe wird verweigert).
6. Der Benchmark-Bericht existiert. **Das Ergebnis (QAL besser oder schlechter als Rust) ist kein Erfolgskriterium**, sondern die Messung, die Phase 2 steuert.

## 13. Teststrategie

- Unit-Tests je Crate (Parser, Typprüfung, Effekte, Hashing, Speicher-Invarianten).
- Golden-Tests: Textform → kanonischer Druck.
- Eigenschaftstest: Rundlauf Parser/Drucker.
- End-to-End: Referenzlösungen bauen, ausführen, versteckte Tests prüfen.
- MCP: Protokolltests über stdio.

## 14. Bewusst nicht in Phase 1

Benutzerdefinierte Generics, Closures, statische Beweise, `net`/`db`, Speicherfreigabe, Mehragenten-Koordination, Signaturen, Self-Hosting, menschenlesbare IDE-Unterstützung.

## 15. Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| Agent ist in QAL anfangs schlechter als in Rust (keine Trainingsdaten) | Kompakte Sprachreferenz in Tool-Beschreibungen; Diagnosen mit Hinweisen; Benchmark macht Ursachen sichtbar |
| Ohne Generics/Closures zu wenig Ausdruckskraft | Eingebaute generische Listenfunktionen; Generics/Closures sind erste Punkte von Phase 2 |
| Bump-Allocator reicht nicht | Limit + klarer Fehler; Phase 3 adressiert Speicherverwaltung |
| Umfang Phase 1 zu groß | Implementierungsplan schneidet in lauffähige Zwischenstände (zuerst `pure`-Ausdrücke bis WASM, dann Effekte, dann Speicher, dann MCP, dann Benchmark) |
