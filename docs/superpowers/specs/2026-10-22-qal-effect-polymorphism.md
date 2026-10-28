# QAL — Effekt-Polymorphie: Design-Spezifikation

**Datum:** 2026-10-22
**Status:** Entwurf zur Umsetzung (Phase 6a)
**Voraussetzung:** Phase 2 (Generics, Closures), Phase 5 (Compiler in QAL, Fixpunkt)

## 1. Problem

Effekte stehen im Typ jeder Funktion und jedes Funktionswerts (`fn(Int) -> Int !io`). Die Prüfregel ist Teilmengen-Subtyping: ein Funktionswert darf *weniger* Effekte haben als erwartet (`unify.rs`, `unify_in` für `Type::Fn`). Das ist korrekt, aber es gibt keine Möglichkeit, eine Funktion zu schreiben, deren Effekte **von einem Argument abhängen**:

```
fn each[T](xs: List[T], f: fn(T) -> Unit) -> Unit = ...
```

`f` ist hier rein. Ein Aufruf `each(urls, fn(u) => fetch(u))` mit `fetch: !net` scheitert mit `type_mismatch` (`qal-check/src/tests.rs:541`). Nur die eingebauten `map`, `filter`, `fold` sind effektpolymorph, fest verdrahtet in `builtins.rs::fn_arg`: die Effekte des Arguments werden Effekte des Aufrufs.

Folgen: Jede eigene Higher-Order-Funktion muss pro Effektmenge kopiert werden (`each_pure`, `each_io`, `each_net`), die Standardbibliothek kann keine Kombinatoren mit Rückrufen anbieten, und Agenten stoßen beim Schreiben von Services auf eine Grenze, die nicht aus der Aufgabe kommt.

## 2. Ziel

Eine Funktion kann **einen Effektparameter** deklarieren, der an Stelle einer Effektmenge steht. Bei jedem Aufruf wird er aus den Argumenten bestimmt; die Effekte des Aufrufs sind die Effekte der Signatur mit eingesetztem Parameter. Alles andere bleibt: Effekte sind Teil des Typs, Teilmengen-Subtyping, invariante Typargumente, Host-Importe nur für die Effekte des Einstiegspunkts.

Nicht Ziel: Effektzeilen mit mehreren Variablen (Koka), Effekt-Handler, Effektinferenz für Top-Level-Signaturen (die bleiben vollständig annotiert), Änderung der Laufzeit.

## 3. Sprache

### 3.1 Syntax

Ein Effektparameter wird in der Typparameterliste mit `!` deklariert und steht danach wie ein Effektname:

```
fn each[T, !e](xs: List[T], f: fn(T) -> Unit !e) -> Unit !e
= match get(xs, 0) { None => (), Some(x) => { f(x); each(tail(xs), f) } }

fn retry[T, !e](n: Int, op: fn() -> Result[T, Str] !e) -> Result[T, Str] !e !time
  req n >= 1
= match op() { Ok(v) => Ok(v), Err(m) => if n == 1 then Err(m) else { sleep_ms(10); retry(n - 1, op) } }
```

Regeln:
- Höchstens **ein** Effektparameter je Funktion (E0310 bei mehr). Begründung: Mengen mit einer Variablen haben eine eindeutige kleinste Lösung bei der Unifikation; mit zwei Variablen nicht (siehe §4.3).
- Der Parameter darf überall stehen, wo ein Effektname steht: in der eigenen Effektmenge der Funktion, in Funktionstypen der Parameter, des Ergebnisses und in Typen innerhalb von Parametern (`List[fn() -> Unit !e]`).
- Ein Effektparameter, der in keinem Parametertyp vorkommt, ist unbestimmbar und ein Fehler (E0311): `fn f[!e]() -> Unit !e` hat keine Quelle für `e`.
- Namen: wie Typparameter ein Bezeichner; `e`, `eff`, `e1` sind üblich. Ein Effektparameter darf keinen der sechs Effektnamen tragen (E0301 unknown/reserved).
- `type`-Definitionen bekommen **keine** Effektparameter. `type Sink[T] = { put: fn(T) -> Unit }` bleibt wie bisher; wer einen effektbehafteten `put` will, schreibt `fn(T) -> Unit !io` ins Feld.
- Tests und Contracts (`req`/`ens`) sind effektfrei; ein Effektparameter ist dort verboten wie jeder Effekt.

### 3.2 Bedeutung

Ein Aufruf `f(a1, …, an)` einer Funktion mit Effektparameter `e`:
1. Instanziiere `e` mit einer frischen Effektvariablen `ε`.
2. Unifiziere Parametertypen mit Argumenttypen (bestehende Regeln; `ε` wird dabei gebunden, §4).
3. Die Effekte des Aufrufs sind `effects(f)[e := ε]`, aufgelöst.
4. Diese Menge muss in den Effekten des Aufrufers enthalten sein (bestehende Regel `Effekte(Aufgerufene) ⊆ Effekte(Aufrufer)`).

Innerhalb des Körpers von `f` ist `e` eine Konstante: der Aufruf `f(x)` mit `f: fn(T) -> Unit !e` braucht `{e}`, die Funktion deklariert `!e`, also erfüllt. `e` ist mit keinem konkreten Effekt vereinbar: `print` im Körper von `each` braucht `io`, deklariert ist nur `e` → E0301 wie heute.

Ein Funktionswert `each` ohne Aufruf (`let g = each in …`) bekommt den Typ mit frischer Variable, wie bei Typparametern (`FnRefG`).

### 3.3 Beispiele, die typisieren

```
fn each[T, !e](xs: List[T], f: fn(T) -> Unit !e) -> Unit !e = ...
fn main(args: List[Str]) -> Int !io = { each(args, fn(a) => print(a)); 0 }     # e := {io}
fn count[T](xs: List[T]) -> Int = { each(xs, fn(_) => ()); len(xs) }           # e := {}
fn twice[!e](op: fn() -> Unit !e) -> Unit !e = { op(); op() }
fn log_twice(m: Str) -> Unit !io = twice(fn() => print(m))
fn pure_twice() -> Unit = twice(fn() => ())
```

### 3.4 Beispiele, die scheitern

```
fn bad1[!e, !f](a: fn() -> Unit !e, b: fn() -> Unit !f) -> Unit !e !f = ...   # E0310: ein Effektparameter
fn bad2[!e]() -> Unit !e = ()                                                 # E0311: e kommt in keinem Parameter vor
fn bad3[T, !e](xs: List[T], f: fn(T) -> Unit !e) -> Unit = each(xs, f)       # E0301: braucht e, deklariert nichts
fn bad4[!e](op: fn() -> Unit !e) -> Unit !e = { op(); print("x") }           # E0301: io nicht deklariert
fn bad5() -> Unit = twice(fn() => print("x"))                                 # E0301: Aufruf braucht io
```

`bad5` ist die Garantie, dass nichts gewaschen wird: die Effekte der Closure landen über `ε` in den Effekten des Aufrufs und müssen vom Aufrufer gedeckt sein.

## 4. Prüfung

### 4.1 Repräsentation

- `FnDef` bekommt `eparams: Vec<String>` (0 oder 1 Eintrag). `Sig` in `qal-check` ebenso.
- Effektmengen bleiben `Vec<String>` in `Type::Fn` und `FnDef.effects`. Ein Eintrag ist entweder ein bekannter Effektname (`KNOWN_EFFECTS`), der Name eines Effektparameters der umgebenden Definition, oder – nur checker-intern – eine Effektvariable `?N` (ein Name, der kein Bezeichner ist, damit er mit nichts kollidiert).
- `Subst` bekommt eine Tabelle `evars: Vec<Option<BTreeSet<String>>>` für Effektvariablen; `resolve` auf `Type::Fn` ersetzt gebundene Variablen durch ihre Menge und sortiert. Unbekannte Variablen am Ende einer Definition werden zu `∅` (ein Funktionswert, den niemand aufruft, ist rein).

### 4.2 Unifikation (`unify_in` für `Type::Fn`)

Sei eine Effektmenge `S = (C, v)` mit konkreten Effekten `C` und höchstens einer Variablen `v` (Invariante aus §3.1; Instanziierung erzeugt je Signatur genau eine). Die heutige Regel `found ⊆ expected` wird:

| expected `(C1, v1)` | found `(C2, v2)` | Regel |
|---|---|---|
| `v1` ungebunden | beliebig | binde `v1 := (C2 \ C1) ∪ {v2}`; Erfolg |
| `v1` fehlt | `v2` fehlt | `C2 ⊆ C1` |
| `v1` fehlt | `v2` ungebunden | `C2 ⊆ C1` und binde `v2 := ∅` |
| gebunden | — | zuerst auflösen, dann die Zeile darüber |

Exakte Unifikation (`unify_exact`, invariante Positionen wie Typargumente eigener Typen): beide Richtungen; zwei ungebundene Variablen werden vereinigt (`v1 := v2`), eine ungebundene Variable gegen eine Menge ohne Variable wird gebunden, danach Mengengleichheit.

Die erste Zeile bindet minimal: `v1` nimmt genau das auf, was `expected` noch nicht hat. Das ist die kleinste Lösung; eine spätere Verwendung derselben Variablen, die weniger verlangt, scheitert mit `type_mismatch` (reihenfolgeabhängig, aber nie unsolide: eine Bindung macht Mengen nur größer, und am Ende wird gegen die deklarierte Menge geprüft).

`join` (Verzweigungen, Listenelemente): Vereinigung bzw. Durchschnitt der konkreten Teile wie heute; Variablen werden beim Vereinigen exakt unifiziert.

### 4.3 Warum eine Variable je Funktion

Mit `expected = (∅, {v1, v2})` und `found = ({io}, ∅)` gibt es drei kleinste Lösungen (`v1 := {io}`, `v2 := {io}`, beide). Die Wahl wäre willkürlich und Fehlermeldungen unverständlich. Mit einer Variablen ist die Lösung eindeutig. Wer zwei unabhängige Rückrufe braucht, nimmt einen Parameter für beide (`fn both[!e](a: fn() -> Unit !e, b: fn() -> Unit !e)`): `e` wird die Vereinigung, was für die Aufruferseite dasselbe ergibt.

### 4.4 Effektprüfung (`effect_check`)

`needed` wird vor der Prüfung aufgelöst. Innerhalb einer Funktion mit Effektparameter `e` enthält `self.effects` den Namen `e`; ein `needed`, das `e` enthält, ist damit gedeckt. Ein `needed`, das eine noch ungebundene Variable `?N` enthält, wird im Rahmen (`frames`) gesammelt und am Ende der Closure-Inferenz aufgelöst; auf Top-Level (kein Rahmen) wird die Variable zu `∅` gebunden, bevor geprüft wird.

### 4.5 Monomorphisierung, Codegen, Laufzeit

Unverändert. `mono.rs` ignoriert Effekte; Effekte erzeugen keinen Code. Host-Importe eines Moduls bestimmen sich aus den aufgelösten Effekten des Einstiegspunkts; ein Einstiegspunkt kann keinen Effektparameter haben (`main` ist nicht generisch). Die Laufzeit-Stubs für nicht gewährte Effekte (`host.rs::link`) bleiben die zweite Schicht.

### 4.6 Prover

Unverändert: Contracts sind effektfrei, Effektparameter kommen dort nicht vor.

## 5. Kanonische Form und Hashes

Invariante: **Kein bestehender Hash ändert sich.** Die kanonische Kodierung (`canon.rs`, Version 2) bleibt:

- `fn body := n type{n} ret:type effects req … ens … body` – unverändert.
- `effects := n str{n}` sortiert – unverändert. Ein Effektparameter wird als `#0` kodiert (allgemein `#k`, `k` = Position in der Liste der Effektparameter; bei einem Parameter immer `#0`). `#` ist kein Buchstabe, kann also kein Effektname sein; `#0` sortiert vor allen Namen. Bestehende Definitionen enthalten keinen `#`-Eintrag, ihre Bytes bleiben gleich.
- Die Deklaration des Effektparameters selbst wird **nicht** kodiert: sie folgt aus dem Vorkommen von `#0`. Der Name des Parameters ist wie lokale Namen Metadaten (`local_names`), damit die Textsicht `!e` statt `!#0` zeigt.
- `sorted_effects` und der Decoder (`effects()`, „sorted set") akzeptieren `#k`. Der Decoder lehnt `#k` mit `k > 0` ab (eine Variable je Funktion).
- Ein Typ (`type`) hat keine Effektparameter; `#k` in einem `type` ist ein Dekodierfehler.

Nachweis: `phase1_definitions_keep_their_version_1_bytes` und `phase1_definitions_keep_their_hashes` bleiben grün; ein neuer Test hasht `each` mit `e` und mit `eff` umbenannt gleich.

## 6. Textform, QALT, Protokoll

- Parser: `[T, !e]` in `fn`-Signaturen; `!e` an Effektpositionen. Ein `!name`, der weder bekannt noch deklariert ist → E0301 mit Hinweis „deklariere `!name` in `[...]` oder meine einen der sechs Effekte".
- Printer (`qal-core/src/printer.rs`): gibt `[T, !e]` und `!e` aus; Rundlauf `parse(print(p)) == p`.
- QALT (`tform.rs`): `FnDef` bekommt `eparams` als Stringliste nach `tparams`; Version des QALT-Formats wird erhöht (es ist kein Hash-Eingang).
- QALQ/QALR: keine Änderung der Nachrichten; `new_names`, `CHECK_FILE`, `TYPED` transportieren Text bzw. QALT.

## 7. Compiler in QAL (`selfhost/qalc.qal`)

Jede Änderung in `qal-text`, `qal-check`, `qal-core` muss **gleichzeitig** in den gespiegelten Modulen landen, sonst bricht der Fixpunkt:

| Rust | QAL-Modul |
|---|---|
| `parser.rs` (`[T, !e]`, `effects()`) | `qc_parse` |
| `printer.rs` | `qc_print` |
| `ast.rs` (`eparams`), `tform.rs` | `qc_ast`, `qc_tform` |
| `unify.rs` (`evars`, Tabelle §4.2) | `qc_unify` |
| `infer.rs` (`instantiate`, `effect_check`, `lambda`) | `qc_check` |
| `canon.rs` (`#0`) | `qc_canon` |

Abnahme: `full_selfhosted_fixpoint` grün (Stage 1 == 2 == 3), `seed_module_is_the_fixpoint` nach `qal selfhost bootstrap --out selfhost/qalc.wasm`, `qalc.wasm.sha256` aktualisiert, Differential-Tests (`frontend_diff`, `checker_diff`, `codegen_diff`) mit neuen Korpus-Fällen für §3.3/§3.4.

## 8. Standardbibliothek und Werkzeuge

- Die Sprachreferenz in den MCP-Tool-Beschreibungen (`qal_define`) bekommt einen Absatz: „Effektparameter: `fn each[T, !e](xs: List[T], f: fn(T) -> Unit !e) -> Unit !e`".
- `std`: keine Pflichtänderung. Empfohlen, in einem eigenen Schritt von Agenten über MCP: `std.list.each`, `std.list.try_map` (`fn(T) -> Result[U, E] !e`), `std.http.with_retry`.
- `bench/`-Aufgaben bleiben gültig; keine nutzt eigene HOFs mit Effekten.

## 9. Diagnosen

| Code | kind | Wann |
|---|---|---|
| E0301 | effect_violation | wie heute; zusätzlich: unbekannter Effektname mit Hinweis auf Deklaration |
| E0310 | effect_params | mehr als ein Effektparameter |
| E0311 | effect_param_unused | Effektparameter kommt in keinem Parametertyp vor |

Neue Codes gehören in `diag::KNOWN`.

## 10. Tests (Pflicht)

`qal-check/src/tests.rs`:
1. `effect_parameters_take_the_arguments_effects`: §3.3 vollständig.
2. `effect_parameters_cannot_launder`: §3.4 `bad3`–`bad5`, plus: Closure mit `ε` in Record-Feld, Variant-Payload, Option, Liste – überall bleibt der Effekt sichtbar; Rückgabe eines `fn() -> Unit !e` aus `twice`-artiger Funktion und Aufruf ohne Deckung → E0301.
3. `one_effect_parameter_per_function`: `bad1`, `bad2`.
4. `effect_variables_resolve_to_pure_when_unused`: `let g = each in 0` typisiert, `g` ist rein.
5. `generic_type_arguments_stay_invariant_with_effect_variables`: `Sink[fn() -> Unit !e]` gegen `Sink[fn() -> Unit]` scheitert.

`qal-core/src/canon.rs`: 6. Hash-Stabilität (§5), Rundlauf Encoder/Decoder mit `#0`, Ablehnung von `#1`.
`qal-text`: 7. Parser/Printer-Rundlauf für §3.3.
`qal-run`: 8. Ein Programm, dessen `main !io` über `each` druckt, importiert `print`; dasselbe mit reiner Closure importiert nur `fail`.
`selfhost`: 9. Korpus-Fälle, Fixpunkt.

## 11. Aufwand und Reihenfolge

1. Rust: AST, Parser, Printer, Canon (`#0`), QALT – ½ Tag; Tests 6, 7.
2. Rust: Checker (`Subst.evars`, `unify_in`, `instantiate`, `effect_check`, `lambda`) – 1–1½ Tage; Tests 1–5, 8.
3. QAL: Spiegelung in sechs Modulen – 1–2 Tage; Differential-Tests, Fixpunkt, Seed – ½ Tag.
4. MCP-Referenz, Roadmap-Zeile, Ledger – ¼ Tag.

Gesamt etwa eine Woche. Schritt 2 ist vor Schritt 3 vollständig zu reviewen, weil Schritt 3 ihn wörtlich nachbaut.

## 12. Entscheidungen

- **Explizit statt implizit.** Ein `fn(T) -> U` ohne Annotation bleibt rein. Die Alternative (unannotierte Funktionsparameter sind implizit polymorph wie die Built-ins) hätte die Bedeutung jeder gespeicherten Signatur geändert und Hashes semantisch umgedeutet.
- **Eine Variable je Funktion.** §4.3.
- **`#0` in der kanonischen Form** statt neuer Version: Hashes bleiben, Decoder alter Stände lesen neue Definitionen nicht (sie kennen `#` nicht) – das ist das gewünschte Verhalten für ein neues Sprachmittel.
- **Kein Effektparameter an Typen.** Hält Typargumente invariant und die Unifikation linear; Felder mit effektbehafteten Funktionen werden konkret annotiert.

## 13. Umsetzungsstand (2026-10-22)

Umgesetzt wie spezifiziert, mit einer Abweichung von §4.2: Effektvariablen sind nicht „einmal gebunden“, sondern **wachsen** (untere Schranke), solange sie auf der Erwartungsseite einer Prüfung stehen, und werden **eingefroren**, sobald eine Prüfung von ihrem Wert abhängt (Gefundenseite, Effektprüfung eines Aufrufs). Grund: `both(fn() => print(..), fn() => now_ms())` bindet `e` beim ersten Argument an `{io}`; mit einer einmaligen Bindung scheiterte das zweite Argument. Die Vereinigung über alle Argumente ist die in §3.2 gewünschte Bedeutung; das Einfrieren stellt sicher, dass ein späteres Wachstum keine frühere Prüfung ungültig macht (`generic_type_arguments_stay_invariant_with_effect_variables` prüft beides).

- Rust: `ast.rs` (`FnDef.eparams`, `Type::map_effects`, `mentions_effect`), `parser.rs`/beide Printer (`[T, !e]`), `tform.rs` (`!e` in der Typparameterliste, byte-identisch ohne Effektparameter), `canon.rs` (`#0`, Name als lokaler Name, Effektspalte ohne Parameter), `diag.rs` (E0310, E0311), `qal-check` (`Subst.evars`, `unify_effects`, `instantiate`, `effect_check`).
- QAL-Spiegel: `qc_ast`, `qc_parse`, `qc_print`, `qc_tform`, `qc_canon`, `qc_prelude`, `qc_unify`, `qc_check`, `qc_mono`; `qalc.qal` neu gebündelt, Seed `qalc.wasm` neu (Stage 1 == 2 == 3), `qalc.wasm.sha256` aktualisiert.
- Tests nach §10: 1–5 in `qal-check/src/tests.rs`, 6 `effect_parameters_hash_by_position`, 7 `prints_effect_parameters`, 8 `effect_parameters_import_only_what_the_call_needs`, 9: 23 neue Korpusfälle (793 extrahierte Programme), `full_selfhosted_fixpoint`, `seed_module_is_the_fixpoint`.
- MCP-Referenz (`reference.md`) um den Absatz zu Effektparametern ergänzt. `std` unverändert (§8).
