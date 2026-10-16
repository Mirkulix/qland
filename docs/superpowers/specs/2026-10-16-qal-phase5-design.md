# QAL Phase 5 — Self-Hosting: Design-Spezifikation

**Datum:** 2026-10-16
**Status:** verbindlich für die Implementierungspläne 5a, 5b, 5c, 5d, 5e
**Bezug:** `2026-10-15-qal-design.md` (§1 „Rust ist ausschließlich das Bau-Gerüst des Compilers, bis QAL sich selbst kompiliert“, §2 Phase 5, §8 WASM), `2026-10-15-qal-roadmap.md` (Zeile 5, Risiko „Phase 5 ist der größte Brocken“), `2026-10-15-qal-phase2-design.md` (Entscheidungen 11–13, 16: Map/Bytes, kanonische Form v2, Codegen unabhängig vom Beweiser), `2026-10-15-qal-phase3-design.md` (§2.2 Instanz pro Anfrage, §4.3 Pakete), Plan 3a (Design Decisions 1–2: Prelude-Referenzen, SQL-Prüfung), `2026-10-16-qal-phase4-design.md` (Team-Verfahren, `todo()`)
**Entscheidungen:** Der Nutzer delegiert alle Designentscheidungen. Jede Entscheidung unten ist getroffen und begründet; Pläne setzen sie um, ohne nachzufragen. Abweichungen nur mit Begründung im Plan-Ledger.

---

## 0. Ziel und Fertig-Kriterien

Der QAL-Compiler — Text → Prüfung → kanonische Form/Hash → WASM — ist ein QAL-Programm. Er übersetzt die QAL-Testsammlung zu identischem Verhalten, übersetzt sich selbst bis zum Fixpunkt, und danach braucht das Standard-Build Rust nur noch für den Host.

**Phase 5 ist fertig, wenn (nachweisbar):**

1. **Compiler in QAL:** Das Projekt `qal/selfhost/` (Paket `qal/selfhost/qalc.qalpkg`, versioniert) enthält Lexer, Parser, Drucker, kanonische Kodierung und Hashing, Namensauflösung, Typ- und Effektprüfung, Generics/Monomorphisierung, Closures, Erschöpfungsprüfung, Prelude und SQL-Literalprüfung (3a), `todo()` (4b) und WASM-Codegenerierung einschließlich Laufzeit-Helfern. Seine Einstiegsfunktion ist **rein** (keine Effekte): Der Compiler hat als WASM-Modul physisch keinen Systemzugriff.
2. **Identisches Verhalten auf der Testsammlung (§10.1):** Für jedes Programm des Korpus liefern QAL- und Rust-Pipeline byte-identische Ergebnisse auf allen Ebenen: gedruckte Textform, kanonische Bytes und Hashes, Diagnosen (alle Felder inkl. `message` und `hints`), typisierte Form (§4.4), WASM-Bytes und Modul-Metadaten; die Ausführung aller Tests, Benchmark-Prüffälle (1e), std-Abnahmetests (3b), Todo- und Shop-Szenarien (3c, 4d) liefert identisches Ergebnis-JSON. Zusätzlich bestehen die Integrationstests von `qal-store`, `qal-run`, `qal-prove`, `qal-serve` und `qal-cli` mit `QAL_COMPILER=qal`.
3. **Bootstrap-Fixpunkt:** `qal selfhost bootstrap` erzeugt Stage 1 (Rust übersetzt `qalc`), Stage 2 (Stage 1 übersetzt `qalc`), Stage 3 (Stage 2 übersetzt `qalc`); **Stage 2 == Stage 3** byte-identisch (Roadmap), und wegen Kriterium 2 auch Stage 1 == Stage 2. Das versionierte Saat-Modul `qal/selfhost/qalc.wasm` ist dieser Fixpunkt (Test: es übersetzt `qalc.qalpkg` zu sich selbst).
4. **Rust nur noch Host:** `cargo build -p qal-cli --no-default-features` baut ohne `qal-text`, `qal-check`, `qal-wasm` und ohne den kodierenden Teil von `qal_core::canon` (geprüft mit `cargo tree -e normal`); in dieser Konfiguration bestehen die Integrationstests aus Kriterium 2, und ein Agent arbeitet über MCP (`qal_define`, `qal_test`, `qal_run`, `qal_request`) wie vorher. Im Standard-Build parst, prüft, hasht und kodiert keine Rust-Funktion QAL-Programme.
5. **Grenzen gemessen und eingehalten (§6):** Selbstübersetzung im Release-Build ≤ 120 s und ≤ 1 GiB Spitzenspeicher je Arbeitspaket; `qal_define` auf den Benchmark-Projekten im Median ≤ 1 s; Messwerte im 5e-Ledger.
6. 0 Warnungen; keine Panik auf erreichbaren Pfaden; Fehler des QAL-Compilers (Trap, Grenze) erscheinen als Diagnose E0951, nie als Absturz des Hosts.

Kriterien 1–4 und 6 sind für ☑ unverzichtbar. Kriterium 5 darf nur auf Stufe R1/R2 der Rückfall-Leiter (§14.2) und nur mit Messwerten gelockert werden. Wird eines der Kriterien 1–4 innerhalb der Sprachgrenzen nicht erreicht, wird die Roadmap-Zeile ◐ mit genau benannter Lücke (Stufe R3/R4), nicht ☑.

---

## 1. Begriffe

- **Compiler** = alles, was aus QAL-Text oder kanonischer Form Bedeutung berechnet: Lexer, Parser, Drucker, kanonische Kodierung/Dekodierung zu AST, Hash-Bildung, Prüfung, Monomorphisierung, Codegen. **Host** = alles, was Bytes bewegt, speichert oder ausführt: wasmtime-Einbettung und Host-Funktionen der Effekte, SQLite-Store, MCP/CLI-Transport, HTTP-Server, Z3-Anbindung, Signaturen (Phase 4), Dekodieren der vom Compiler gelieferten Ergebnisse (typisierte Form → `TProgram` für Beweiser und Wert-Kodierung), SHA-256 über Bytes.
- **Stage 0** = Rust-Pipeline (`qal-text` + `qal-check` + `qal-wasm` + `canon`). **Stage n+1** = das WASM-Modul, das Stage n aus `qalc.qalpkg` erzeugt.
- **Korpus** = die QAL-Testsammlung (§10.1).
- **Arbeitspaket** = ein Aufruf des Compilers in einer frischen WASM-Instanz (§4.3).

---

## 2. Entscheidungen auf einen Blick

| # | Frage | Entscheidung | Begründung (kurz) |
|---|---|---|---|
| 1 | Maßstab „identisch“ | **byte-identisch zur Rust-Pipeline** auf allen Ebenen (Text, Hash, Diagnosen, typisierte Form, WASM) — nicht nur gleiches Laufzeitverhalten | Differenztests werden exakt (erste abweichende Funktion/Knoten wird gefunden); Fixpunkt Stage 2 == 3 folgt dann aus Stage 1 == 2; keine Verhaltens-Orakel nötig |
| 2 | Art des Compilers | **getreue Portierung** der Rust-Crates, Modul für Modul, nicht Neuentwurf | Byte-Identität verlangt dieselben Entscheidungen (Funktionsreihenfolge, Locals, Helfer, Datenlayout) |
| 3 | Reihenfolge | 5b Frontend + kanonische Form/Hash → 5c Codegen (aus typisierter Form der Rust-Prüfung) → 5d Prüfer → 5e Bootstrap und Umschalten | nach Risiko geordnet: das Schwierigste (Inferenz, Monomorphisierung, Diagnosetexte) zuletzt; jeder Zwischenstand ist ein kohärenter Rückfall (§14.2) |
| 4 | Schnittstelle | reine Funktion `qc_main(req: Bytes) -> Bytes`; Anfrage/Antwort in einem eigenen Binärformat (`QALQ`/`QALR`) | Bytes sind der kleinste gemeinsame Nenner; kein JSON-Parser im Hot-Path; versionierbar |
| 5 | Eingabe | **beides**: Quelltext (für `define`, Dateimodus) **und** kanonische Formen mit Namen (der Store-Stand) | der Store hält kanonische Formen; `define` braucht Text plus Stand; Build/Test brauchen nur Formen |
| 6 | Ausgabe | WASM-Bytes als `Bytes` + Modul-Metadaten + typisierte Form + Diagnosen | genau das, was Host heute aus `CompiledModule`, `TProgram`, `Hashed` und `Vec<Diagnostic>` liest |
| 7 | Speicher ohne GC | **Arena pro Arbeitspaket** (frische Instanz je Aufruf, wie Phase 3 pro Anfrage) + Fork/Join-Protokoll, mit dem der Compiler seine Arbeit selbst in Pakete teilen kann | der Bump-Allocator gibt nie frei; frische Instanzen sind das bewährte Freigabemodell; der Host bleibt generisch |
| 8 | Speicherlimit | Compiler-Läufe dürfen bis 1 GiB (Modul-Maximum 16 384 Seiten, Option `memory_max_pages` im Build); Programme der Nutzer bleiben bei 256 MiB | Zeiger bleiben < 2³¹, vorzeichenbehaftete i32-Vergleiche im Laufzeitcode bleiben korrekt (Audit in 5a) |
| 9 | Neue Sprachmittel | nur die Lücken aus §5.2 (Bit-Operationen, Float-Bits, allokationsfreier Indexzugriff, Akkumulator-Typ, `bytes_join`, `sha256`); **keine** veränderlichen Puffer | Unveränderlichkeit trägt Hashing, Beweise und Isolation; die Lücken lassen sich rein schließen |
| 10 | Rust während des Übergangs | Referenz; `--compiler rust|qal|both`, in allen Tests `both` (Abweichung = E0950); Sprachstopp für neue Features außer §5.2 | jede Abweichung wird sofort und am Ort sichtbar |
| 11 | Saat | `qal/selfhost/qalc.wasm` versioniert (+ `qalc.wasm.sha256`); reproduzierbar mit der Rust-Referenz (Feature `reference`) | ohne Saat bräuchte das Standard-Build wieder den Rust-Compiler; Reproduzierbarkeit entschärft „Trusting Trust“ |
| 12 | Autoren | Agenten über MCP im Team-Verfahren von Phase 4 (Architekt legt Signaturen und Stummel aus den Rust-Modulen an, Implementierer portieren, Verifizierer prüfen differenziell); der Controller schreibt keine QAL-Körper | Self-Hosting ist der größte Praxistest für QAL als Agentensprache; gemessen wie 4d |

---

## 3. Was zuerst — und warum

| Teil | Rust heute (Zeilen inkl. Doku, ohne Tests) | Interface zur Prüfung | Risiko |
|---|---|---|---|
| Lexer, Parser, Drucker | `qal-text` ≈ 1 700 | gedruckter Text, Parse-Diagnosen | niedrig |
| Kanonische Form, Hash, Dekoder | `qal-core::canon` ≈ 2 300 | Bytes, Hashes (exakt spezifiziert, §5 Gesamtspezifikation) | mittel (Gruppen-Labelling mit Suche) |
| Codegen inkl. Laufzeit-Helfer | `qal-wasm` ≈ 2 500 (`rt.rs` ≈ 1 000) | WASM-Bytes | mittel (Umfang, Byte-Treue) |
| Prüfer, Monomorphisierung | `qal-check` ≈ 2 500 | typisierte Form, Diagnosen mit englischen Texten | **hoch** (Inferenz, Unifikation, Fehlermeldungen) |

**Entscheidung: Frontend und kanonische Form zuerst (5b), dann Codegen (5c), dann Prüfer (5d).**

- 5b zuerst: es legt die AST-Datentypen in QAL fest, die alle späteren Teile benutzen; Byte-Vergleich von Text und Hashes ist das einfachste exakte Orakel; Identität (Hash) ist der Kern der Sprache und damit der wertvollste erste Schritt.
- 5c vor 5d: Codegen braucht nur die **typisierte Form** (§4.4), die die Rust-Prüfung liefern kann. Damit gibt es nach 5c einen gemischten Compiler (Rust prüft, QAL erzeugt WASM), dessen Ausgabe schon byte-genau mit Stage 0 verglichen und dessen Codegen-Teil bereits einen Fixpunkt durchlaufen kann — Frühwarnung für Byte-Treue und Laufzeitgrenzen, bevor das riskanteste Stück beginnt.
- 5d zuletzt: der größte Diagnose-Oberflächenbereich; scheitert er an den Grenzen, bleibt mit 5b + 5c ein klar definierter Teilerfolg (§14.2 Stufe R3).

---

## 4. Schnittstelle Compiler ↔ Host

### 4.1 Einstieg

```
fn qc_main(req: Bytes) -> Bytes        # rein; kein `!io`, kein `!fs`
```

Der Host lädt das Compiler-Modul (Saat oder frisch gebaute Stage) einmal (`InstancePre`), ruft `qc_main` je Arbeitspaket in einer frischen Instanz auf (Phase-3-Pooling, `Limits::compiler()`), ohne Effekt-Freigaben.

### 4.2 Formate

Alle Ganzzahlen minimale LEB128 (wie die kanonische Form), Strings `len utf8`, Listen `n elem{n}`.

```
req  := "QALQ" ver:u8(1) op:u8 payload
op   := 1 PARSE      text                                        → Programm-AST (für Dateimodus und Tests)
      | 2 DEFINE     store-state text options                    → Prepared (§4.5)
      | 3 PRINT      store-state names                           → Textform je Name
      | 4 TYPED      store-state roots options                   → typisierte Form
      | 5 BUILD      store-state entry options                   → Modul
      | 6 CHECK_FILE text entry? options                         → Diagnosen | typisierte Form | Modul
      | 7 JOIN       continuation results                        (Fortsetzung, §4.3)
      | 8 SIGNATURES store-state names                           → Signaturtexte (für `query --type`)
store-state := n entry{n} m group{m}
entry       := name hash[32] kind:u8 canonical local_names refs   (sortiert nach Name)
group       := hash[32] form
options     := memory_max_pages                                  (feste Feldliste von Version 1, §16)

res  := "QALR" ver:u8(1) status:u8 payload
status := 0 DONE payload | 1 DIAGNOSTICS diag* | 2 FORK continuation item* | 3 INTERNAL message
```

- `DIAGNOSTICS`: dieselben Felder wie `qal_core::diag::Diagnostic` (`code`, `kind`, `def`, `path`, `message`, `expected?`, `found?`, `hints`, `dependents`, `counterexample?`) als Strings; der Host dekodiert sie in `Diagnostic`, indem er `code`/`kind` gegen das Register `qal_core::diag::KNOWN` abbildet (5a: statt `Cow`, §16; ein unbekanntes Paar ist E0952).
- `INTERNAL`: der Compiler hat einen eigenen Fehler erkannt (Invariante verletzt) → Host meldet E0951.
- Trap des Compilers (`out_of_memory`, `timeout`, `contract_violation`, …) → Host meldet E0951 `compiler_failed` mit Stage, Operation, Arbeitspaket und dem Laufzeitfehler; der Host läuft weiter.

### 4.3 Arbeitspakete (Fork/Join)

Eine Antwort `FORK continuation [item…]` bedeutet: der Host ruft `qc_main(item)` für jedes Item in einer **frischen** Instanz (parallel auf bis zu `--jobs` Threads, Standard = Kerne, höchstens 8), und dann `qc_main(JOIN continuation [result…])` mit den Ergebnissen in Item-Reihenfolge. Das darf sich wiederholen (höchstens 8 Runden, höchstens 100 000 Items, höchstens 256 MiB Gesamtgröße je Runde; sonst E0954 `protocol_limit`).

- Der Host versteht keinen Inhalt; die Schleife ist ≈ 60 Zeilen generischer Code in `qal-run` (`selfhost::drive`).
- Determinismus: Ergebnisse in Item-Reihenfolge, unabhängig von Threads; die Ausgabe hängt nie von `--jobs` ab (Test: `--jobs 1` == `--jobs 8`).
- Der Compiler darf mit **einem** Paket beginnen (keine FORK-Antwort). Teilen ist das vorab festgelegte Mittel gegen Speichermangel (§6): naheliegend sind Parsen/Hashen je Definition, Prüfen je Definition (Top-Level-Definitionen haben vollständige Signaturen, ein Körper hängt nur von den Signaturen seiner Abhängigkeiten ab) und Codegen je Funktion mit einem abschließenden Linken.

### 4.4 Typisierte Form (`QALT`)

Neues Binärformat für das geprüfte, monomorphe `TProgram` (Typen, Funktionen mit Instanzen, Tests, Lambdas mit Captures, Muster, Built-in-Aufrufe, Effekte): Kodierer **und** Dekodierer in Rust (`qal_core::tform`, 5a), Kodierer und Dekodierer in QAL (5c Dekodierer, 5d Kodierer).

- Zweck 1: Schnittstelle Prüfer → Codegen in 5c (Rust-Prüfer + QAL-Codegen).
- Zweck 2: der Host braucht `TProgram` weiterhin für Beweiser (Z3-Kodierung) und Wert-Kodierung/-Dekodierung (`qal-run`); nach dem Umschalten kommt es aus `QALT`.
- Zweck 3: exaktes Orakel für 5d (QAL-Prüfer muss byte-gleiches `QALT` wie Rust erzeugen).
- Rundlauf-Test in 5a: `decode(encode(t)) == t` für jedes geprüfte Programm des Korpus.

### 4.5 `DEFINE`-Ergebnis

Genau die Daten, die `qal-store` heute aus `Prepared` liest, damit die Store-Logik (Ersetzungsregel, `identical_types`, `group_aliases`, `alias_references`, `canonical_check`, Schreiben) **unverändert in Rust** bleibt — sie ist Store-Invariante, nicht Übersetzung:

```
prepared := typed:QALT  n canon{n}  m group{m}  sig_old{k}
canon    := name kind hash canonical signature effects local_names refs deps group?
sig_old  := name kind signature_text_with_current_names        (für die Ersetzungsregel der Eingabenamen)
```

---

## 5. Sprachmittel: Bestand und Lücken

### 5.1 Bestand (Phasen 1–4), den der Compiler braucht

| Bedarf | Mittel | Stand |
|---|---|---|
| AST, Typen, Diagnosen | Records, Enums mit Payload, rekursive und generische Typen | ☑ (1b, 2a) |
| Symboltabellen, Substitutionen | `Map[K, V]` (persistenter AVL, O(log n)), `compare` | ☑ (2a) |
| Quelltext, Ausgabe | `Str`, `Bytes`, `to_bytes` (kostenlos), `char_code` | ☑ (1b, 2a) |
| Hilfsfunktionen höherer Ordnung | Closures, `map`/`filter`/`fold` | ☑ (2a) |
| Schleifen | Endrekursion → Schleife | ☑ (1a) |
| Float-Literale | `to_float` (gleiche Rust-`parse::<f64>` im Host wie der Lexer), `from_float` (gleiche Formatierung wie der Drucker) | ☑ |
| Prelude, SQL-Literale | Prelude-Typen (3a), SQL-Tokenizer — beides muss der QAL-Prüfer nachbauen | portieren (5d) |
| Stummel während der Portierung | `todo()` | ☑ (4b) |

### 5.2 Lücken und ihre Schließung (5a)

| Lücke | Warum es ohne nicht geht | Neues Built-in (rein) |
|---|---|---|
| Bit-Operationen | LEB128/SLEB128, IEEE-Bits, Flags; `/`/`%` runden zur Null und sind für negative Werte falsch und langsam | `bit_and(a, b) bit_or(a, b) bit_xor(a, b) bit_not(a) shl(a, n) shr(a, n)` (arithmetisch) `ushr(a, n)` (logisch), `req 0 <= n && n < 64` (sonst Built-in-Vorbedingung) |
| Float-Bits | `f64.const` braucht die IEEE-Bytes; kanonische Form kodiert Bitmuster | `float_bits(x: Float) -> Int`, `float_of_bits(i: Int) -> Float` |
| Allokationsfreier Index | `get`/`bytes_get`/`map_get` liefern `Option` = eine Heap-Allokation je Zugriff; der Lexer liest jedes Byte | `list_at[T](xs: List[T], i: Int) -> T` (5a: statt `at`, §16), `bytes_at(b: Bytes, i: Int) -> Int` (außerhalb → Trap `index`) |
| Listen aufbauen | `push`/`++` kopieren (O(n) je Schritt, mit Bump-Allocator O(n²) Speicher — 20 000 Definitionen ≈ 1,6 GB) | Prelude-Typ `type Accum[T] = AccEnd \| AccPush(Accum[T], T)` (5a: statt `Acc`, §16; O(1) je Anhängen) und `acc_list[T](a: Accum[T]) -> List[T]` (O(n), in Anhänge-Reihenfolge) |
| Bytes zusammenfügen | Modulbytes entstehen aus ≈ 10⁵ Stücken | `bytes_join(bs: List[Bytes]) -> Bytes` (eine Kopie, O(Summe)) |
| SHA-256 | Hash-Bildung; in QAL mit `Option`-Zugriffen speicherhungrig und langsam | `sha256(b: Bytes) -> Bytes` als **reine Host-Hilfe** (wie `float_to_str`, `utf8_valid` seit Phase 1) |

- Kanonische Form: neue Built-in-Namen und der generische Prelude-Typ (Referenz `R_PRELUDE` mit Typargumenten über `T_APP`); bestehende Hashes ändern sich nicht. Beweiser: neue Built-ins sind approximiert (frische Werte mit Bereichsfakten, z. B. `0 <= bytes_at(b, i) <= 255`).
- **Keine veränderlichen Puffer, keine Schleifen-Syntax, keine GC:** Begründung — Unveränderlichkeit ist die Grundlage für Inhaltsadressierung, Beweise und das Isolationsmodell; die Tabelle oben schließt jede Lücke rein und lokal. Ob es reicht, misst 5a (§6.2), bevor portiert wird.

---

## 6. Laufzeitgrenzen und Budget

### 6.1 Heutige Grenzen

Bump-Allocator, Modul-Maximum 4 096 Seiten = 256 MiB, Fuel 2·10⁹, Wandzeit 10 s, WASM-Stack 64 MiB (Host-Thread größer), Parser-Höhe 256.

### 6.2 Schätzung für die Selbstübersetzung (zu messen in 5a)

| Größe | Schätzung | Grundlage |
|---|---|---|
| Quelltext `qalc` | 10 000–14 000 Zeilen, ≈ 450 KB, ≈ 2 000 Definitionen | ≈ 9 000 Zeilen Rust, ausdrucksorientiert 1–1,5× |
| Token / AST-Knoten | ≈ 150 000 / ≈ 300 000 | Verhältnis aus dem Benchmark-Korpus |
| Map-Einfügungen in Prüfung | 1–3 Mio. × ≈ 450 B Pfadkopie | AVL-Tiefe ≈ 14 bei 10⁴ Einträgen |
| Speicher ein Paket (ganzes Programm) | 0,6–1,5 GB | über 256 MiB, nahe 1 GiB → Teilen nötig |
| Speicher je Definition-Paket | ≤ 32 MiB | größte Funktion der Rust-Crates, × Faktor 50 |
| WASM-Instruktionen | 10¹⁰–5·10¹⁰ | Rust ≈ 3·10⁸ native Instruktionen, Faktor 30–150 |
| Wandzeit (8 Threads) | 20–90 s | wasmtime/cranelift ≈ 1–2·10⁹ Instr./s je Kern |

**Messgate 5a (vor jeder Portierung):** ein QAL-Prototyp, der das Korpus lext, ein Synthetik-Test mit 10⁶ `map_put` auf Str-Schlüsseln und ein 1-MiB-Modul aus 10⁵ Stücken über `Acc`/`bytes_join`. Liegen die hochgerechneten Werte über §0 Kriterium 5, wird das Teilen (§4.3) vor 5b für Prüfen und Codegen verbindlich eingeplant statt als Rückfall.

### 6.3 Grenzen der Compiler-Läufe

`Limits::compiler()`: 1 GiB Speicher (Modul mit `memory_max_pages = 16384`), 2·10¹¹ Fuel und 300 s Wandzeit je Arbeitspaket, 600 s gesamt, WASM-Stack 64 MiB. Rekursionsregeln für den Compiler-Code (in 5b als Lint-Test über das Paket): keine Nicht-End-Rekursion über Folgen (Definitionen, Token, Listen) — nur über Baumhöhe (≤ 256 durch den Parser begrenzt) oder kanonische Verschachtelung (≤ 256); ein Test übersetzt eine 1-MiB-Quelle mit 20 000 Definitionen.

---

## 7. Aufbau des QAL-Compilers

| QAL-Modul (Namenspräfix) | Port von | Inhalt |
|---|---|---|
| `qc_lex_*`, `qc_parse_*`, `qc_print_*` | `qal-text` | Bytes-basierter Lexer (`bytes_at`), rekursiver Abstieg mit Höhengrenze, kanonischer Drucker |
| `qc_ast_*` | `qal-core::ast` | AST-Typen |
| `qc_canon_*` | `qal-core::canon` | Kodierung v1/v2, Gruppen-Labelling, Dekodierung, Signaturtext |
| `qc_check_*`, `qc_unify_*`, `qc_mono_*`, `qc_exhaust_*`, `qc_builtin_*`, `qc_prelude_*`, `qc_sql_*` | `qal-check`, `qal-core::{tast, prelude, sql}` | Auflösung, Inferenz, Effekte, Monomorphisierung, Erschöpfung, Diagnosen |
| `qc_tform_*` | `qal-core::tform` (5a) | Kodierer/Dekodierer `QALT` |
| `qc_wasm_*`, `qc_rt_*`, `qc_enc_*` | `qal-wasm`, `wasm-encoder` (nur die benutzten Teile) | Codegen, Laufzeit-Helfer, Modul-Bauer, Sektionskodierung (Reihenfolge und LEB128 wie `wasm-encoder 0.259`, Locals einzeln wie `asm.rs`) |
| `qc_proto_*`, `qc_main` | — | `QALQ`/`QALR`, Fork/Join, Dispatch |

- Eigenes Projekt `qal/selfhost/` (Store nicht versioniert, Export `qalc.qalpkg` versioniert), unqualifizierte Namen mit Präfix `qc_` (kein Konflikt: eigener Store; der `std.`-Namensraum bleibt der std).
- **Kein** Zugriff auf `std`: der Compiler darf sich nicht auf Code stützen, den er selbst übersetzen muss und der sich unabhängig ändert.
- **Verfahren:** Phase-4-Team im Projekt `qal/selfhost/`. Je Rust-Datei ein Arbeitspaket des Architekten mit QAL-Signaturen, Contracts und `todo()`-Körpern sowie dem Rust-Quelltext als Vorlage in der Beschreibung; Implementierer portieren; Verifizierer legen die Differenztests des Pakets an (§10) und geben `verified` nur bei byte-gleichem Ergebnis auf dem Paket-Korpus. Abweichungen werden nie durch Anpassen der Rust-Referenz „gelöst“, außer ein Rust-Fehler ist nachgewiesen (§9).

---

## 8. Staging und Byte-Identität

```
Stage 0  Rust-Pipeline                          qalc.qalpkg ──▶ qalc1.wasm
Stage 1  qalc1.wasm (unter wasmtime)            qalc.qalpkg ──▶ qalc2.wasm
Stage 2  qalc2.wasm                             qalc.qalpkg ──▶ qalc3.wasm
Prüfung  sha256(qalc1) == sha256(qalc2) == sha256(qalc3) und byteweiser Vergleich
```

- **Eingabe jeder Stage:** dieselbe `BUILD`-Anfrage — `qalc.qalpkg` importiert in einen frischen Store, Stand nach Name sortiert serialisiert, Einstieg `qc_main`, `memory_max_pages = 16384`. Die Anfrage-Bytes werden einmal erzeugt und als Datei gehalten; jede Stage bekommt dieselbe Datei (keine Abhängigkeit von Store-Reihenfolge, Zeit oder Threads).
- **Byte-Vergleich:** bei Abweichung meldet `qal selfhost bootstrap` den Abschnitt (Sektion, Funktionsindex, Offset) und dekodiert beide Funktionen mit `wasmparser` (Host-Werkzeug, nicht Compiler) zum Diff.
- **Ausgabe:** `{"stage1": "<sha256>", "stage2": …, "stage3": …, "fixpoint": true, "stage1_equals_stage2": true, "ms": [..], "fuel": [..], "peak_mib": [..], "work_items": [..]}`.
- **Saat:** `qal/selfhost/qalc.wasm` = Stage 2 nach erfolgreichem Fixpunkt; `qal-cli` bettet es per `include_bytes!` ein, ein Build-Skript prüft `qalc.wasm.sha256`. Test `selfhost_seed_is_a_fixpoint`: die Saat übersetzt `qalc.qalpkg` zu sich selbst. Neue Saat nur zusammen mit neuem `qalc.qalpkg` im selben Commit.
- **Trusting Trust:** Weil Stage 1 == Stage 2 gilt, ist die Saat mit der Rust-Referenz (`--features reference`) reproduzierbar; ein CI-Schritt tut genau das.

---

## 9. Rust als Referenz während des Übergangs

- Neue Fassade-Crate `qal-compiler` mit Trait `Backend` (Operationen §4.2) und zwei Implementierungen: `Rust` (Feature `reference`, ruft die bestehenden Crates) und `Qal` (lädt die Saat bzw. eine angegebene Stage). `qal-store`, `qal-run`, `qal-prove`, `qal-serve`, `qal-cli` hängen nur noch von der Fassade ab (5a).
- Auswahl: `--compiler rust|qal|both` und `QAL_COMPILER`; Standard bis 5e `rust`, ab 5e `qal`. `both` führt beide aus und vergleicht **alle** Ausgaben byte-weise; Abweichung → E0950 `compiler_divergence` mit Ebene (Text, Hash, Diagnose, `QALT`, WASM) und erster Abweichung. In CI laufen alle Integrationstests ab 5b mit `both` für die jeweils portierten Operationen (gemischte Pipeline: noch nicht portierte Teile kommen aus Rust).
- **Sprachstopp:** von 5b bis 5e keine Sprachänderung außer §5.2; nötige Fixes erst in Rust (mit Test), dann im selben Teilplan in QAL.
- **Rust-Fehler gefunden:** wird in Rust behoben, nur wenn Spezifikation oder Rundlauf-Garantie ihn als Fehler ausweisen; beide Seiten werden angepasst, im Ledger dokumentiert. Abweichende Hashes wären eine Formatänderung und sind ausgeschlossen (sonst neue Version der kanonischen Form — außerhalb Phase 5).
- Nach 5e bleibt die Rust-Pipeline hinter `reference` erhalten: Orakel für Differenztests, Reproduktion der Saat. „Nötig“ ist sie für den Betrieb nicht mehr (Kriterium 4).

---

## 10. Teststrategie

### 10.1 Korpus (die „QAL-Testsammlung“)

1. alle `.qal`-Dateien in `qal/examples/`, `qal/crates/*/tests/fixtures/`, `bench/solutions/qal/` (20 Lösungen), Pakete `qal/std/std.qalpkg`, `qal/examples/todo/todo.qalpkg`, `qal/examples/shop/shop.qalpkg`, `qal/selfhost/qalc.qalpkg`;
2. alle QAL-Quelltexte aus Rust-Tests, extrahiert per Skript `qal/selfhost/tools/extract_corpus.py` (String-Literale, die an `parse_program`, `define`, `qal_define` usw. gehen), dedupliziert nach SHA-256, als Dateien `qal/selfhost/corpus/<sha-präfix>.qal` versioniert; sie decken die Diagnosepfade ab;
3. generierte Programme: Generator `qal/selfhost/tools/gen` (Rust, nur Test) erzeugt pro CI-Lauf 10 000 zufällige Programme (Seed im Log) — wohltypisierte aus einer Grammatik mit Typen und ill-typisierte durch Mutation (Token löschen/tauschen, Namen ersetzen, Typen vertauschen) von Korpusprogrammen.

Die Zahl der Programme aus (1) und (2) wird in 5a festgeschrieben (erwartet ≥ 1 000; festgeschrieben: 27 + 683 = 710, §16).

### 10.2 Ebenen des Differenztests

| Ebene | ab | Vergleich |
|---|---|---|
| Parse + Druck | 5b | `print(parse(x))` byte-gleich; Parse-Diagnosen gleich |
| Kanonische Form / Hash | 5b | Bytes jeder Definition und Gruppe, Hashes, lokale Namen, `refs`, Signaturtext |
| Dekodieren | 5b | `PRINT` aus Store-Stand byte-gleich zu `qal_view` |
| WASM | 5c | Modulbytes und Metadaten aus Rust-`QALT` |
| typisierte Form | 5d | `QALT` byte-gleich |
| Diagnosen | 5d | alle Felder; Reihenfolge der Diagnosen |
| Verhalten | 5c | Ausführung: Tests, Benchmark-Prüffälle, std-Abnahmetests, Szenarien → Ergebnis-JSON gleich |
| Integration | 5c–5e | Integrationstests mit `QAL_COMPILER=both`, in 5e `qal` |
| Fixpunkt | 5c (Codegen-Teil), 5e (ganz) | §8 |
| Leistung | 5a, 5e | §0 Kriterium 5, `--jobs`-Unabhängigkeit |

### 10.3 Eigene Tests des Compilers

Jedes `qc_`-Modul hat QAL-`test`-Definitionen (rein) und, wo sinnvoll, Contracts mit Beweisen (z. B. `ens` auf LEB128-Längen) und Property-Tests (Phase 2): `qc_decode(qc_encode(d)) == d`, `qc_parse(qc_print(p)) == p`.

---

## 11. Diagnosen

| Code | kind | Wann |
|---|---|---|
| E0950 | `compiler_divergence` | `--compiler both`: Ausgaben unterscheiden sich (`detail`: Ebene, Ort) |
| E0951 | `compiler_failed` | QAL-Compiler trapt, überschreitet Grenzen oder meldet `INTERNAL` |
| E0952 | `bad_compiler_response` | Antwort verletzt `QALR` (Host-seitige strikte Dekodierung) |
| E0953 | `seed_mismatch` | eingebettete Saat ≠ `qalc.wasm.sha256` oder kein Fixpunkt |
| E0954 | `protocol_limit` | Fork/Join-Grenzen §4.3 |

Alle übrigen Diagnosen sind die bestehenden — byte-gleich aus dem QAL-Compiler.

---

## 12. Rust nach Phase 5

| Crate | Standard-Build | mit `reference` |
|---|---|---|
| `qal-core` | AST/`tast`/`diag`-Datentypen, `tform`-Dekoder, Prelude-Layout, Hash-Typ; **ohne** kanonische Kodierung | vollständig |
| `qal-text`, `qal-check`, `qal-wasm` | nicht gelinkt | Orakel |
| `qal-compiler` | Fassade, Backend `Qal`, Saat | + Backend `Rust` |
| `qal-run`, `qal-store`, `qal-prove`, `qal-serve`, `qal-cli` | Host | unverändert |

Der Beweiser bleibt Rust (Z3-Bindung, Host): er kodiert das `TProgram` aus `QALT`; seine Bestätigung von Gegenbeispielen übersetzt über die Fassade.

---

## 13. Bewusst nicht in Phase 5

Optimierungen im Codegen (Phase 2 §7.4: Entfernen bewiesener Prüfungen bleibt offen — es würde die Byte-Identität zur Referenz brechen), ein Beweiser in QAL, Garbage Collection, veränderliche Datenstrukturen, eine neue Version der kanonischen Form, Inkrementalität über Prozessgrenzen hinaus (außer Rückfall R2), WASI.

---

## 14. Risiken und Rückfallplan

### 14.1 Risiken

| Risiko | Wahrscheinlichkeit | Gegenmaßnahme |
|---|---|---|
| Speicher ohne GC reicht nicht (Prüfung eines 2 000-Definitionen-Programms in einem Paket > 1 GiB) | hoch | Messgate 5a; Teilen je Definition über Fork/Join (§4.3) ist vorbereitet und vom Host unabhängig |
| Fuel/Zeit: QAL-Code mit persistenten Maps ist 30–150× langsamer als Rust | mittel | parallele Arbeitspakete; Limits §6.3; `qal_define` im Alltag betrifft kleine Projekte; Rückfall R2 |
| Byte-Identität in Details (Iterationsreihenfolge von Rust-`HashMap`s, Float-Formatierung, Fehlermeldungstexte) | mittel | 5a sucht alle `HashMap`/`HashSet`-Iterationen in Rust-Pfaden, deren Reihenfolge eine Ausgabe beeinflusst, und ersetzt sie durch `BTreeMap`/sortierte Iteration (Ausgabe dabei unverändert oder als Rust-Fehler behandelt); Floats über dieselben Host-Hilfen |
| Umfang (≈ 12 000 Zeilen QAL von Agenten) | hoch | Team-Verfahren, Portierung Datei für Datei mit Vorlage, Differenztests je Paket, harte Reihenfolge 5b → 5c → 5d |
| Gruppen-Labelling (Suche mit Grenzen) weicht ab | mittel | dieselben Grenzen (`MAX_LABEL_NODES`, `MAX_LABEL_WORK`) und dieselbe Suchreihenfolge; eigener Generator-Lauf für rekursive Gruppen |
| Rekursionstiefe (Stack) | niedrig | Lint §6.3; 1-MiB-Test |
| Saat-Binärdatei im Repo (≈ 1–3 MB) | niedrig | reproduzierbar (§8); nur mit Paket zusammen geändert |
| Agentenkosten explodieren | mittel | gemessen (Tokens, Aufrufe je Paket) im 5b–5d-Ledger; Abbruchkriterium: > 3 Runden je Paket → neuer Agent mit Fehlerklasse (wie 3b) |

### 14.2 Rückfall-Leiter

Jede Stufe ist ein definierter, getesteter Zustand; die höchste erreichte wird in der Roadmap genannt.

| Stufe | Zustand | Roadmap |
|---|---|---|
| R0 | alle Kriterien §0 | ☑ |
| R1 | Kriterien 1–4 und 6 erfüllt; Kriterium 5 nur gelockert: Selbstübersetzung > 120 s (bis 600 s) oder `qal_define` > 1 s, mit Messwerten | ☑ mit dokumentierter Lockerung von Kriterium 5 |
| R2 | wie R1, aber `qal_define` zu langsam → Ergebnis-Cache je Definitions-Hash (`QALT`-Fragmente und Funktions-Code; Store-Schema 6 mit `compiled(hash, compiler, data)`), Prüfung nur der betroffenen Definitionen | ☑, wenn danach Kriterium 5 hält |
| R3 | Prüfer in QAL scheitert an Grenzen: QAL-Frontend + kanonische Form + QAL-Codegen, Rust-Prüfer liefert `QALT`; Fixpunkt Stage 2 == 3 für das gemischte Modul | ◐ „Self-Hosting ohne Prüfer“, Kriterium 4 verfehlt |
| R4 | nur 5b erreicht: Identität (Text, Hash) self-hosted, Differenztests grün | ◐ |

Abbruchsignale für den Wechsel auf die nächste Stufe: Messgate 5a verfehlt auch mit Teilen; oder ein Teilplan überschreitet zweimal das geplante Volumen ohne konvergierende Differenzen.

---

## 15. Teilpläne

- **5a — Fundament (Rust und Sprachmittel):** Built-ins §5.2 (Prüfer, Codegen, Host-Hilfe `sha256`, Beweiser-Approximation, Referenz); `memory_max_pages` und `Limits::compiler()` mit i32-Audit; `qal_core::tform` (Kodierer/Dekodierer, Rundlauf über das Korpus); Fassade `qal-compiler` mit Backend `Rust`, Umstellung aller Host-Crates auf die Fassade; `QALQ`/`QALR`-Format und `selfhost::drive` (Fork/Join, `--jobs`); `Diagnostic` mit `Cow`; `HashMap`-Iterationsaudit; Korpus-Extraktion und Generator; Messgate §6.2 mit Prototyp. **Fertig:** alle bestehenden Tests grün über die Fassade, Messgate dokumentiert, Korpusgröße festgeschrieben.
- **5b — Frontend und Identität in QAL:** Projekt `qal/selfhost/`, Team-Einrichtung, `qc_ast`, `qc_lex`, `qc_parse`, `qc_print`, `qc_canon` (inkl. Gruppen-Labelling und Dekoder), Ops `PARSE`, `PRINT`, `SIGNATURES`, Hash-Teil von `DEFINE`; Backend `Qal` für diese Ops; Differenzebenen Text/Hash/Dekodieren; Lint §6.3. **Fertig:** Korpus + 10 000 generierte Programme ohne Abweichung.
- **5c — Codegen in QAL:** `qc_tform`-Dekoder, `qc_enc`, `qc_wasm`, `qc_rt`, Op `BUILD` aus Rust-`QALT`; Differenz WASM + Verhalten; gemischter Fixpunkt (Rust-Prüfer, QAL-Codegen: das Codegen-Modul übersetzt sich zweimal byte-gleich). **Fertig:** WASM-Ebene und Verhaltensebene ohne Abweichung, gemischter Fixpunkt.
- **5d — Prüfer in QAL:** `qc_check`, `qc_unify`, `qc_mono`, `qc_exhaust`, `qc_builtin`, `qc_prelude`, `qc_sql`, `qc_tform`-Kodierer, Ops `DEFINE`, `TYPED`, `CHECK_FILE` vollständig; Diagnosen byte-gleich; ggf. Teilen in Pakete. **Fertig:** Ebenen typisierte Form und Diagnosen ohne Abweichung, Integrationstests mit `both` grün.
- **5e — Bootstrap und Umschalten:** `qal selfhost bootstrap`, Saat `qalc.wasm` mit Prüfsumme und Fixpunkt-Test, Standard-Backend `qal`, Feature `reference` aus dem Standard-Build, `--no-default-features`-Konfiguration in CI, Leistungsmessung §0.5, Rückfallentscheidung §14.2, Roadmap Zeile 5 mit Nachweisen, Abschluss-Review. **Fertig:** Kriterien 1–6 (bzw. die erreichte Stufe benannt).

Abhängigkeiten: Phase 3 (Prelude, SQL-Prüfung, Pakete) und Phase 4 (Team-Verfahren, `todo()`) vor 5b; `5a → 5b → 5c → 5d → 5e`; 5c kann nach 5b parallel zu den ersten 5d-Paketen laufen (getrennte Arbeitspakete im Team).

---

## 16. Entscheidungen aus 5a (nachgetragen)

Abweichungen von dieser Spezifikation, die 5a getroffen hat; 5b–5e bauen auf ihnen auf (Bericht `.superpowers/sdd/phase5/5a-report.md`).

| Thema | Spezifikation | 5a |
|---|---|---|
| Index-Built-in | `at` | `list_at` — `bench/solutions/qal/07-tokenizer.qal` definiert `at` |
| Akkumulator | `Acc[T]` | `Accum[T]` mit `AccEnd`/`AccPush` und `acc_list` — `03-csv-sum.qal` definiert `type Acc`; kodiert als `T_APP R_PRELUDE "Accum"` |
| Diagnosen über `QALR` | `code`/`kind` als `Cow` | `&'static str` bleibt; Dekodieren gegen das Register `diag::KNOWN` (E0950–E0954 eingeschlossen), unbekannt → E0952 |
| Abhängigkeit Fassade | — | `qal-run` → `qal-compiler`; das Backend `Qal` (5b) bekommt den Ausführer injiziert (`qal_run::selfhost::Invoke`/`Compiler`), weil `qal-compiler` nicht von `qal-run` abhängen kann |
| `options` von `QALQ` v1 | `memory_max_pages entry_kind tparams-limits …` | nur `memory_max_pages`; Prüfergrenzen (z. B. `MAX_INSTANCES`) sind Konstanten der Sprache |
| `--jobs`, `--compiler` | CLI-Optionen | `jobs` ist Parameter von `selfhost::drive` (`default_jobs()`); die CLI-Optionen kommen mit `qal selfhost bootstrap` (5e) bzw. dem Backend `Qal` (5b) |
| Feature `reference` | in 5a angelegt | noch nicht: `CompiledModule`, `FuncMeta`, `host`, `FAIL_*` müssen dafür erst nach `qal-core` |
| Korpus | ≥ 1 000 erwartet | 27 Dateien + 683 extrahierte = 710 (660 parsen, 465 prüfen; Stand nach Phase 4a); `extract_corpus.py --check` ist ein Test |
| Messgate | — | Speicher eines Ganzprogramm-Pakets hochgerechnet 0,4–1,7 GB → Teilen in Arbeitspakete ist ab 5b verbindlich; Zeit nur für das Lexen gemessen (QAL/Rust ≈ 37×) |
| Beweiser | — | jedes Built-in, das trappen kann, trägt seine Bedingung zu `ok` bei (`PROVER_VERSION` `qal-prove/3`) |

