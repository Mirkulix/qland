# QAL Phase 2 — Beweise & Ausdruckskraft: Design-Spezifikation

**Datum:** 2026-10-15
**Status:** verbindlich für die Teilpläne 2a und 2b (der Nutzer delegiert alle Designentscheidungen; sie sind hier getroffen und begründet)
**Bezug:** `2026-10-15-qal-design.md` (§2 Phase 2, §3.3 „Phase 2: per Beweis“, §14), `2026-10-15-qal-roadmap.md` (Zeile 2, Risiko Phase 5)
**Teilpläne:** `docs/superpowers/plans/2026-10-15-qal-phase2a-generics-closures-maps.md`, `docs/superpowers/plans/2026-10-15-qal-phase2b-proofs-properties.md`

---

## 1. Ziel und Fertig-Kriterien

Phase 2 macht Contracts zu Beweisen und gibt QAL die Ausdruckskraft, die Agenten (und später der QAL-Compiler in QAL, Phase 5) brauchen. Fertig ist Phase 2, wenn nachweisbar gilt (Roadmap Zeile 2):

1. `req`/`ens` werden für Ganzzahl-Arithmetik mit Z3 statisch bewiesen; bei Widerlegung gibt es ein **bestätigtes** Gegenbeispiel; alles nicht Bewiesene bleibt zur Laufzeit geprüft (und ist es auch bewiesen — §7.4).
2. Benutzerdefinierte Generics für Funktionen und Typen, kompiliert durch Monomorphisierung.
3. Closures (anonyme Funktionen mit Capture), deren Effekte im Typ stehen.
4. Generierte Property-Tests mit reproduzierbaren Seeds und Schrumpfen.
5. Zusätzlich (Roadmap-Risiko Phase 5): `Map[K, V]` und `Bytes`.
6. Alles über CLI und MCP erreichbar; keine Panik auf erreichbaren Pfaden; 0 Warnungen; die Teilpläne sind gegen den Repository-Stand nach 1d/1e nachgespielt.

## 2. Entscheidungen auf einen Blick

| # | Frage | Entscheidung | Begründung (Kurz) |
|---|---|---|---|
| 1 | Generics-Syntax | `fn f[T, U](x: T) -> U`, `type Pair[A, B] = …`, Anwendung `Pair[Int, Str]` | eckige Klammern wie bei `List[T]`; keine neue Syntaxklasse |
| 2 | Typargumente am Aufruf | nie geschrieben, immer inferiert | weniger Tokens; Typen sind an Aufrufen immer aus Argumenten/Erwartung bestimmbar |
| 3 | Kompilierung | Monomorphisierung (eine Instanz je Typargumentliste) | einfacher, schneller WASM-Code; keine Laufzeittypen; Phase-5-Compiler bleibt einfach |
| 4 | Rekursion generischer Funktionen | nur mit den eigenen Typparametern (E0213) | garantiert endlich viele Instanzen |
| 5 | Rekursive generische Typen | nur regulär (E0212) | `Nest[List[T]]` hätte unendlich viele Instanzen |
| 6 | `==` auf Typparametern | erlaubt; pro Instanz geprüft (Funktionstyp ⇒ Fehler) | `contains[T]`, `index_of[T]` sind ohne Typklassen sonst unmöglich |
| 7 | Effekt-Polymorphie | keine; Funktionstypen tragen feste Effektmengen; `map/filter/fold` übernehmen die Effekte ihres Funktionsarguments | einfach, sound; die Built-ins decken den häufigsten Fall |
| 8 | Closure-Syntax | `fn(x, y: T) => body` | wiederverwendet `fn`, Typannotation optional |
| 9 | Capture | per Wert (alle Werte sind unveränderlich, also semantisch gleich „per Referenz“) | keine Lebensdauerfragen |
| 10 | Closure-Repräsentation | jeder Funktionswert ist ein Zeiger auf `[Tabellenindex][pad][Captures…]`; einheitliche Aufrufkonvention `(env, args…)` | ein Funktionstyp, eine Darstellung für benannte Funktionen und Closures |
| 11 | Map/Bytes | **ja** | Phase 5 braucht Symboltabellen und Binärdaten; Kosten ≈ 600 Zeilen |
| 12 | Map-Repräsentation | persistenter AVL-Baum (O(log n) Lesen und Schreiben, O(log n) Speicher je Einfügen) | ein sortiertes Array wäre O(n) je `map_put` und mit dem Bump-Allocator O(n²) Speicher |
| 13 | Kanonisches Format | Version je Form: v1 für reine Phase-1-Konstrukte, v2 sobald ein Phase-2-Konstrukt vorkommt | alle Phase-1-Hashes bleiben gültig, keine Migration bestehender Stores (z. B. aus 1d/1e) |
| 14 | Beweiser | Z3 über das Crate `z3 =0.21.1`, Feature `vendored` (statisch gelinkt) | keine Systeminstallation; deterministisch per Ressourcenlimit |
| 15 | Int im Beweis | mathematische Ganzzahlen + Bereichsbedingungen (Überlauf = Trap), keine Bitvektoren | QAL wrappt nie; exakt und für lineare Arithmetik schnell |
| 16 | Laufzeitprüfungen | bleiben **alle** erhalten, auch für bewiesene Contracts | Codegen unabhängig vom Solver (reproduzierbares WASM), Annahme von Callee-`ens` wird trivial sound |
| 17 | Widerlegt | nur ein durch Ausführung **bestätigtes** Gegenbeispiel heißt `refuted`; eine Funktion mit widerlegtem Contract wird nicht gespeichert, ein Test schon | keine Fehlalarme; „nur Geprüftes existiert“ |
| 18 | Ergebnisse | `proven` / `refuted` / `unknown` je Beweisverpflichtung, gespeichert je Definitions-Hash und Beweiser-Schlüssel | Hash fixiert alles, wovon der Beweis abhängt |
| 19 | Property-Tests | reine Funktionen mit `ens` oder mit Aufrufen von Funktionen mit `req`; 100 Eingaben, die `req` erfüllen; Fehler = Contract-Verletzung; Schrumpfen | ergänzen die Beweise dort, wo der Beweiser `unknown` sagt |

## 3. Generics

### 3.1 Syntax und Bedeutung

```
type Pair[A, B] = { fst: A, snd: B }
type Tree[T] = Leaf | Node(Tree[T], T, Tree[T])
fn swap[A, B](p: Pair[A, B]) -> Pair[B, A] = Pair { fst: p.snd, snd: p.fst }
fn size[T](t: Tree[T]) -> Int = match t { Leaf => 0, Node(l, _, r) => size(l) + 1 + size(r) }
test t = size(Node(Leaf, swap(Pair { fst: 1, snd: true }), Leaf)) == 1
```

- Typparameter sind großgeschriebene Namen, eindeutig, und nicht Name eines deklarierten oder eingebauten Typs (E0202). Jeder Typparameter einer Funktion muss in einem Parameter- oder dem Ergebnistyp vorkommen (E0211 `unused_type_parameter`), sonst wäre sein Argument nie inferierbar.
- Innerhalb der Definition sind Typparameter **starr** (`Type::Param`): `x + 1` mit `x: T` ist ein `type_mismatch`. Ein generischer Typ braucht genau so viele Argumente, wie er Parameter hat (E0204); `Box` statt `Box[T]` ist ein Fehler mit Hinweis.
- Am Aufruf bekommt jeder Typparameter eine frische Inferenzvariable; Konstruktoren (`Node(…)`, `Pair { … }`) und Muster generischer Typen werden ebenso instanziiert. Typargumente werden nie geschrieben.
- `main` darf nicht generisch sein. Einstiegspunkte, Tests und `qal_call` arbeiten mit Instanzen.
- `==`/`!=` auf Werten eines Typparameters sind erlaubt; nach der Monomorphisierung wird jede Instanz geprüft — ein Funktionstyp als Argument ist dann ein `type_mismatch` (Funktionen sind nicht vergleichbar). Das ist die einzige Prüfung, die erst an der Instanz erfolgt (Entscheidung 6).

### 3.2 Monomorphisierung

`qal_check::check` prüft jede generische Funktion einmal (Ergebnis in `TProgram::generics`), erzeugt dann ausgehend von allen monomorphen Funktionen und Tests jede erreichte Instanz (`f[Int, Str]`, angehängt an `TProgram::fns`, mit `TFn::instance = Some((generic, typargs))`) und ersetzt generische Aufrufe und Funktionsreferenzen (`CallG`, `FnRefG`) durch Aufrufe der Instanzen. Geprüfte Programme sind monomorph; der Codegenerator meldet einen verbliebenen generischen Knoten als internen Fehler (fail-closed).

Terminierung und Grenzen:

| Gefahr | Gegenmaßnahme | Diagnose |
|---|---|---|
| polymorphe Rekursion (`f[T]` ruft `f[List[T]]`) | innerhalb einer rekursiven Gruppe generischer Funktionen nur die eigenen Typparameter | E0213 `polymorphic_recursion` (auch wenn nie instanziiert) |
| irreguläre rekursive Typen (`Nest[T] = … Nest[List[T]]`) | rekursive Verwendung nur mit den eigenen Parametern | E0212 `irregular_recursion` |
| Instanzketten (`d0[T]` → `d1[Pair[T, T]]` → …) | höchstens 1000 Instanzen, Typargumente höchstens 4096 Knoten | E0214 `instantiation_limit` |
| große Rümpfe × viele Instanzen (Speicher) | höchstens 500 000 in Instanzen kopierte Ausdrucksknoten, jeder gewichtet mit `1 + Größe der Typargumente` (jeder kopierte Knoten trägt einen substituierten Typ) | E0214 |
| irreguläre Typen in Typ-Durchläufen (`==`, Map-Schlüssel) | Prüfung endet nach E0212; jeder Durchlauf expandiert höchstens 256 verschiedene Typen | E0212 |

Die Kosten der Prüfung bleiben damit linear in Quellgröße plus diesen Konstanten (gemessen im Release-Build: die gewichtete 500 000-Knoten-Grenze hält den schlimmsten Fall unter ≈ 1 s).

### 3.3 Hashing und Speicher

- Generische Definitionen sind Mitglieder der Arten `3` (Typ) bzw. `4` (Funktion) mit der Anzahl Typparameter; ein Typparameter wird als `T_PARAM index` kodiert. Namen der Typparameter sind lokale Namen (wie Parameter): **Umbenennen ändert keinen Hash**.
- Bei Funktionen ist die Reihenfolge der Typparameter bedeutungslos (sie werden nie geschrieben); kanonisch ist die Reihenfolge des ersten Vorkommens in Parameter-, dann Ergebnistyp. `fn k[A, B](a: A, b: B) -> A` und `fn k[B, A](…)` haben denselben Hash. Bei Typen ist die Reihenfolge Bedeutung (sie steht an jeder Verwendung `Pair[Int, Str]`).
- Signaturtext: `fn[A, B](Pair[A, B]) -> Pair[B, A]`, generische Typen `[A, B] { fst: A, snd: B }`; `query --type` vergleicht diesen Text (Muster `fn[_](_) -> _`).
- Instanzen werden nie gespeichert; der Store hält die generische Definition, Aufrufer verweisen auf ihren Hash.

## 4. Closures

### 4.1 Syntax und Typisierung

```
fn adder(n: Int) -> fn(Int) -> Int = fn(x) => x + n
fn logger(prefix: Str) -> fn(Str) -> Unit !io = fn(s) => print(prefix ++ s)
test t = let k = 3 in map([1, 2], fn(x) => x * k) == [3, 6]
```

- `fn(p1, p2: T) => body`: der Rumpf reicht so weit wie möglich (wie bei `let`). Parametertypen sind optional; erwartet der Kontext einen Funktionstyp passender Stelligkeit (Parameter, Annotation, Built-in-Argument), kommen sie von dort (bidirektional), sonst werden sie inferiert.
- Eine Closure erfasst die lokalen Variablen, die ihr Rumpf benutzt (`TLambda::captures`, sortiert), per Wert. Da alle Werte unveränderlich sind, ist das von „per Referenz“ nicht unterscheidbar. Rekursion geht über benannte Funktionen.
- Funktionswerte (benannt oder nicht) sind nicht vergleichbar und keine Map-Schlüssel (`type_mismatch`).

### 4.2 Effekte

- Der Typ einer Closure enthält die Effekte ihres Rumpfs: `fn(s) => print(s)` hat den Typ `fn(Str) -> Unit !io`. **Erzeugen** ist rein, **Aufrufen** verlangt die Effekte (Prüfung am `CallIndirect` wie bei jedem Funktionswert).
- Effekt-Subtyping: ein Wert mit weniger Effekten passt, wo mehr erlaubt sind; in Funktionsparametern umgekehrt (kontravariant); in `List`, `Option`, `Result`, `Map` kovariant; in Argumenten **benutzerdefinierter** generischer Typen **invariant** (`type Sink[T] = { put: fn(T) -> Unit }` verwendet `T` in Parameterposition — Kovarianz würde Effekte verstecken).
- Keine Effekt-Polymorphie: `fn apply[T](f: fn() -> T) -> T` nimmt nur reine Funktionen. Ausnahme sind die Built-ins `map`, `filter`, `fold`: ihr Aufruf hat die Effekte des Funktionsarguments.
- Folge: Effekte können nicht über Closures in Records, Listen, Maps, Optionen, generischen Containern oder Rückgabewerten „entkommen“: jeder Aufruf steht im Typ, jede Prüfung bleibt `Effekte(Aufgerufene) ⊆ Effekte(Aufrufer)`. Tests und Contracts bleiben rein. Das WASM-Modul importiert weiterhin nur die Host-Funktionen der im erreichten Code (inkl. Closure-Rümpfe) benutzten Built-ins.

### 4.3 WASM

- Jeder Funktionswert ist ein `i32`-Zeiger auf einen Closure-Block `[Tabellenindex: i32][pad][Capture-Slot; n]`. Aufruf: `call_indirect` mit dem Tabellenindex aus dem Block und dem Blockzeiger als erstem Argument (`env`), dann die Argumente.
- Der Rumpf einer Closure wird als eigene Funktion `(env, params…) -> ret` erzeugt; sie lädt die Captures aus `env` in ihre Locals.
- Eine benannte Funktion als Wert bekommt einmal einen statischen Block und eine Hülle `(env, params…)`, die `env` ignoriert und per Endaufruf weiterleitet.
- `map/filter/fold` im Laufzeitcode rufen Closures nach derselben Konvention.

### 4.4 Hashing

`E_LAMBDA n (annot?)ⁿ body`: Parameter werden wie `let`-Bindungen gebunden (De-Bruijn); Umbenennen von Closure-Parametern ändert keinen Hash. Annotationen sind Teil des Hashes (sie können Bedeutung haben, z. B. Float statt Int).

## 5. Map und Bytes

### 5.1 Map[K, V]

- Unveränderlich, nach Schlüssel geordnet. Built-ins: `map_new() map_put(m, k, v) map_get(m, k) -> Option[V] map_del(m, k) map_has(m, k) map_size(m) map_keys(m) map_values(m)`; `map_keys`/`map_values` in Schlüsselordnung; `==` ist strukturell (gleiche Einträge).
- `compare(a, b) -> Int` (−1/0/1) macht diese Ordnung für alle Werte ohne Funktionen verfügbar (Sortieren von Listen, Records, Varianten, …; `<` gibt es für Int, Float und — seit 1e — Str; auf Int und Str ist es dieselbe Ordnung, auf Float die IEEE-Totalordnung). Ergänzt nach dem Agententest, der vor 1e lief: ohne sie war `Tree[Str]` nur mit selbstgeschriebenem Bytevergleich sortierbar.
- Schlüsselordnung (total, strukturell): Int und Bool numerisch, Float nach IEEE-Totalordnung (−0.0 < 0.0, NaNs geordnet), Str und Bytes byteweise (Präfix zuerst), Listen und Maps lexikographisch, Varianten nach Tag dann Nutzlast, Records nach Feldern in Deklarationsreihenfolge. Schlüssel dürfen keine Funktionen enthalten (geprüft am geschriebenen Typ und an jeder Instanz). Bewusst: Schlüssel nutzen die Totalordnung (−0.0 und 0.0 sind verschiedene Schlüssel, ein NaN-Schlüssel wird wiedergefunden), `==` auf Float bleibt überall IEEE-Gleichheit (eine Map mit NaN-Schlüssel ist daher nicht `==` zu sich selbst).
- Repräsentation: die leere Map ist `0`; sonst ein Knoten eines persistenten AVL-Baums `[Größe: i32][Höhe: i32][Schlüssel-Slot][Wert-Slot][links: i32][rechts: i32]` (32 Byte). `map_put`/`map_del` kopieren nur den Pfad (O(log n) Zeit und Speicher), `map_get`/`map_has` O(log n), `map_size` O(1). Gleichheit und Vergleich (Maps als Schlüssel) laufen über die Einträge in Ordnung (Bäume gleicher Maps können verschieden aussehen).
- JSON-Form (Ergebnisse und `qal_call`-Argumente): Liste von `[Schlüssel, Wert]`-Paaren in Schlüsselordnung; der Host sortiert übergebene Paare und lehnt doppelte Schlüssel ab.

### 5.2 Bytes

- Unveränderliche Bytefolge, Layout wie `Str` (`[len: i32][Bytes]`), daher ist `to_bytes(s)` kostenlos. Built-ins: `to_bytes(s) bytes_len(b) bytes_get(b, i) -> Option[Int] bytes_slice(b, from, to) bytes_from(xs) bytes_to_list(b) bytes_to_str(b) -> Option[Str]`, `b1 ++ b2`, `==`.
- `bytes_from` verletzt seinen Vertrag (`contract_violation`), wenn ein Element nicht in `0..=255` liegt; `bytes_slice` außerhalb der Grenzen ist `index`; `bytes_to_str` prüft UTF-8 im Host.
- JSON-Form: Liste von Zahlen; gekürzte Werte (Budget von 256 KiB je Wert, wie Strings) enden mit `"...(+N bytes)"`.

## 6. Kanonische Form Version 2

- Neue Tags: Typen `T_APP` (Anwendung eines generischen Typs, Verweis per Hash), `T_PARAM`, `T_MAP`, `T_BYTES`; Ausdruck `E_LAMBDA`; Mitgliedsarten 3/4 (generischer Typ/generische Funktion).
- **Version je Form:** eine Form wird mit der kleinsten Version geschrieben, die alle ihre Konstrukte hat (eine Gruppe: die größte ihrer Mitglieder). Phase-1-Definitionen behalten Bytes und Hash von Version 1. Der Decoder ist in beide Richtungen strikt: eine v1-Form darf kein v2-Konstrukt enthalten, eine v2-Form muss eines enthalten — sonst gäbe es zwei Kodierungen derselben Definition.
- Store-Schema 3 fügt nur die Tabelle `proofs` hinzu; ein Store von Schema 2 (Phase 1) wird beim Öffnen in einer Transaktion hochgestuft. Definitionen werden nicht migriert.

## 7. Statische Beweise mit Z3

### 7.1 Beweisverpflichtungen

Für jede monomorphe Funktion (Instanzen nicht: sie sind durch die generische Definition abgedeckt), jede generische Funktion und jeden Test:

- jedes `ens[k]`: Gelten die `req` und kehrt der Rumpf zurück, wertet `ens[k]` ohne Trap zu `true` aus (die früheren `ens` dürfen angenommen werden, sie werden zur Laufzeit vorher geprüft);
- jeder Aufruf einer Funktion mit `req` im Rumpf — auch in Closures und in Tests, nicht in Contracts (dort prüft die Laufzeit): das `req[k]` des Aufgerufenen gilt, wann immer der Aufruf erreicht wird (frühere `req[j]` des Aufgerufenen dürfen angenommen werden).

### 7.2 Kodierung

Jeder Ausdruck wird zu einem Wert (Z3-`Int` für `Int`, `Bool` für `Bool`, nichts für andere Typen) und einer Bedingung `ok`, die **jede Auswertung ohne Trap erfüllt**.

- **Int:** mathematische Ganzzahlen. Jedes Ergebnis von `+ - *` und unärem `-` trägt `MIN ≤ v ≤ MAX` zu `ok` bei: Da QAL bei Überlauf trapt, ist jeder zur Laufzeit existierende Wert exakt sein mathematischer Wert. (`if x < 0 then -x else x` ist damit `>= 0` beweisbar — mit Wrapping-Semantik wäre es falsch, `-MIN = MIN`.) `/` und `%` schneiden zur Null hin ab wie `i64.div_s`/`i64.rem_s`; `x / 0` und `MIN / -1` sind Traps, `MIN % -1 = 0`. Parameter liegen im Bereich.
- **Bool**, `! && ||` (Kurzschluss: der rechte Operand wird nur unter der entscheidenden Bedingung ausgewertet), Vergleiche, `==`/`!=` auf Int/Bool exakt.
- **Kontrollfluss:** `if`, `let`, Sequenzen, `match` mit Literal-, Bindungs- und Wildcard-Mustern auf Int/Bool exakt; Arme in Reihenfolge (ein Arm gilt nur, wenn kein früherer passte), Guards; genau ein Arm wird genommen (Matches sind erschöpfend, sonst Trap) — ohne diese Tatsache war die Rekursion über Bäume (`size(l) + 1 + size(r)`) nicht beweisbar.
- **Aufrufe** `g(args)`: der Aufrufer muss `g`s `req` herstellen (Verpflichtung), danach gelten `g`s `req` und `ens` für das Ergebnis. Rümpfe der Aufgerufenen werden nicht betrachtet (modular). Contracts, die selbst Funktionen mit Contracts aufrufen, werden bis Tiefe 3 eingebettet.
- **Closures:** der Rumpf wird an der Erzeugungsstelle kodiert, mit frischen Parametern und den Werten der erfassten Variablen; alles, was er außer seinen Parametern sieht, ist unveränderlich und bei der Erzeugung gebunden. Verpflichtungen darin gelten für jeden späteren Aufruf.
- **Approximation (Soundness-Regel):** Was nicht modelliert ist — Float, Str, Listen, Records, Enums, Maps, Bytes, Feldzugriffe, Aufrufe von Funktionswerten, die meisten Built-ins, Nutzlasten nicht-literaler Muster — wird ein **frischer, unbeschränkter Wert**; sein `ok` enthält nur Tatsachen, die jede reale Auswertung erfüllt (`0 ≤ len(s) ≤ 2³²−1`, `lo ≤ rand_int(lo, hi) ≤ hi`, Bereich von Int). Frische Werte sind in jeder Prüfung allquantifiziert: der Solver betrachtet mindestens jede reale Ausführung. Ein Beweis bleibt ein Beweis; nur die Präzision leidet.

**Soundness-Argument:** Für jede reale Ausführung, die eine Verpflichtung erreicht, gibt es eine Belegung der frischen Werte, die Vorbedingung und Pfadbedingung erfüllt (Überapproximation). Ist `pre ∧ hyp ∧ ¬goal` unerfüllbar, gilt `goal` also in jeder realen Ausführung. Die Annahme von Callee-Contracts ist sound, weil jeder Contract zur Laufzeit geprüft bleibt: ein Aufruf, der zurückkehrt, hat `req` und `ens` bestanden (bei Rekursion: Induktion über die Tiefe des Aufrufbaums). Bewiesen wird partielle Korrektheit; Terminierung nicht.

### 7.3 Ergebnisse

| Status | Bedeutung | Diagnose |
|---|---|---|
| `proven` | `pre ∧ hyp ∧ ¬goal` unerfüllbar | — |
| `refuted` | Z3 fand ein Modell **und** die Ausführung der Funktion (des Tests) auf diesen Werten verletzt genau diesen Contract (oder trapt bei seiner Auswertung) | E0801 `contract_refuted` mit `counterexample` |
| `unknown` | Solver aufgegeben (Ressourcenlimit/Zeit) oder Gegenbeispiel nicht bestätigt (die Ausführung erfüllt den Contract, trapt anders, hat Effekte, ist generisch, …); `reason` nennt den Grund und die approximierten Konstrukte | E0802 `contract_unproven` (Warnung) |

- Bestätigung: Parameter, die Z3 nicht modelliert, bekommen ihren kleinsten Wert (`0`, `""`, `[]`, `None`, die Variante mit den wenigsten Feldern; höchstens 10 000 Knoten, sonst keine Ausführung). Funktionen mit Effekten und generische Funktionen werden nicht ausgeführt. Ausführung mit Instruktions- (5·10⁷) und Zeitlimit (5 s). Ein `req` eines Aufgerufenen gilt nur als bestätigt, wenn die Werte der Verletzung den modellierten Argumenten **dieses** Aufrufs gleichen und keinem anderen Aufruf desselben Aufgerufenen mit derselben Verpflichtung.
- Ein Bericht: `{def, obligation ("ens[0]" | "req[1] of g"), path, contract, status, counterexample?, reason?}`.

### 7.4 Laufzeitprüfungen bleiben

Auch bewiesene Contracts bleiben im WASM geprüft. Gründe: (1) das erzeugte Modul hängt nicht vom Solver, seiner Version oder seinem Ressourcenlimit ab — `qal build` liefert überall dasselbe WASM; (2) die Annahme der Callee-Contracts (§7.2) braucht keine globale Konsistenz der Beweisergebnisse; (3) die Kosten einer Prüfung sind gering. Das Entfernen bewiesener Prüfungen ist eine spätere Optimierung (Phase 5), die dann die Konsistenz aller Beweise einer Gruppe nachweisen muss.

### 7.5 Determinismus, Grenzen, Speicherung

- Jede Verpflichtung wird in einem **frischen Z3-Kontext** auf einem eigenen Thread mit 256 MiB Stack (Kodierung und Z3 rekursieren entlang der Verschachtelung) mit `random_seed = 0` und Ressourcenlimit `rlimit = 10 000 000` gelöst (deterministisch für eine Z3-Version; gemessen: ein schweres nichtlineares Ziel gibt nach ≈ 3 s auf, einfache Widerlegungen brauchen 0,2 s; 2·10⁶ war im Review zu knapp). Die Wandzeit-Grenze (5 s je Verpflichtung, nie über das Budget hinaus) ist nur Sicherheitsnetz; ein `prove`-Aufruf hat ein Gesamtbudget von 30 s, danach `unknown` („not attempted“).
- Eine Kodierung umfasst höchstens 50 000 Ausdrücke einschließlich eingebetteter Contracts (k Aufrufe über 3 Ebenen wären k³); darüber erhält die Funktion einen einzigen Bericht `unknown` („too large to encode“).
- Beweiser-Schlüssel: `qal-prove/1 z3-<Version> rlimit-<n>`. Ergebnisse werden in `proofs(hash, prover, results, at)` gespeichert; der Definitions-Hash fixiert alles, wovon ein Beweis abhängt (die Contracts der Aufgerufenen über deren Hashes). Ergebnisse, die von Maschinenlast abhängen (Budget erschöpft, Wandzeit-Timeout, interner Fehler), werden **nicht** gespeichert.
- `define` beweist innerhalb seiner Transaktion alle neuen, ersetzten und neu verknüpften Funktionen und Tests (und unveränderte ohne Ergebnis für den Schlüssel): Ein widerlegter Contract einer **Funktion** (ihr `ens` oder ein `req` an einem ihrer Aufrufe) lässt das ganze `define` scheitern — auch wenn erst eine Neuverknüpfung einen Abhängigen widerlegt. Widerlegte Aufrufe in **Tests** werden gemeldet, aber gespeichert (der Test schlägt beim Ausführen fehl; TDD bleibt möglich).

## 8. Generierte Property-Tests

- **Kandidaten:** reine Funktionen (keine Effekte; auch Instanzen generischer Funktionen), deren Parametertypen generierbar sind (keine Funktionen), mit einem `ens` oder einem Aufruf einer Funktion mit `req`.
- **Eingaben:** SplitMix64, Seed = `seed XOR FNV-1a(Funktionsname)` (Standard-`seed` 0, `--seed`/`seed`). Größe wächst mit jedem Versuch von 0 bis 30: Int meist in `[-s, s]`, jedes zehnte Mal ein Randwert (0, ±1, MIN, MAX, MIN+1, MAX−1); Float analog mit NaN/±∞/−0.0; Str aus einem Alphabet mit Mehrbyte-Zeichen; Listen, Maps, Records, Varianten rekursiv mit halbierter Größe (Größe 0 wählt die kleinste Variante, rekursive Typen terminieren).
- **req-Filter:** eine generierte Funktion `req?f` (Konjunktion der `req` von `f`) entscheidet, ob eine Eingabe gültig ist; ungültige werden verworfen (`discarded`). Weniger als die Hälfte gültige Läufe ⇒ `gave_up`.
- **Fehler:** jede `contract_violation` (das `ens` von `f`, `req`/`ens` von Aufgerufenen, auch `f`s eigenes `req` an einem rekursiven Aufruf, Vorbedingungen von Built-ins). Andere Traps (Überlauf, Division durch null, Index, Stack, Speicher, Zeit) zählen als `traps` und sind kein Fehler — wie die Beweise geht es um Contracts (partielle Korrektheit).
- **Schrumpfen:** gierig, ein Argument nach dem anderen; Kandidaten: Int → 0, x/2, x∓1; Str/Bytes/Listen → leer, Hälften, ohne ein Element, kleinere Elemente; Varianten → kleinste Variante, Teilbaum gleichen Typs, kleinere Nutzlast; Records → kleinere Felder. Akzeptiert, wenn derselbe Contract (Definition, Contract, Pfad) fehlschlägt; höchstens 500 Versuche.
- **Grenzen:** 100 Läufe je Funktion, je Lauf 2·10⁷ Instruktionen und 2 s; 20 s Gesamtbudget (MCP: der Rest des 30-s-Testbudgets), danach `skipped`.
- Bericht: `{def, status: passed|failed|gave_up|skipped, runs, discarded, traps, seed, counterexample?, error?, trap?}` (`error` nur für `failed`, `trap` = erster tolerierter Trap). Ein generierter Wert hat höchstens 10 000 Knoten; gibt es keinen, ist das Ergebnis `gave_up`.

## 9. Diagnosen

| Code | Art | Wann |
|---|---|---|
| E0202 | `duplicate_name` | Typparameter doppelt oder Name eines Typs; Closure-Parameter doppelt |
| E0203 | `type_mismatch` | u. a. Map-Schlüssel mit Funktion; `==` auf Funktionswerten in einer Instanz; `main` generisch |
| E0204 | `type_mismatch` | falsche Anzahl Typargumente |
| E0211 | `unused_type_parameter` | Typparameter kommt in keinem Parameter-/Ergebnistyp vor |
| E0212 | `irregular_recursion` | rekursiver generischer Typ mit anderen Argumenten als seinen Parametern |
| E0213 | `polymorphic_recursion` | rekursiver Aufruf einer generischen Funktion mit anderen Typargumenten |
| E0214 | `instantiation_limit` | > 1000 Instanzen, zu große Typargumente oder > 500 000 kopierte Knoten |
| E0801 | `contract_refuted` | bestätigtes Gegenbeispiel (`counterexample`: Parametername → Wert) |
| E0802 | `contract_unproven` | Warnung mit Grund; Contract bleibt Laufzeitprüfung |

Diagnosen bekommen das Feld `counterexample` (nur E0801/E0802). Fehlgeschlagene Property-Tests sind keine Diagnosen, sondern Testergebnisse mit `counterexample` und dem Laufzeitfehler.

## 10. CLI und MCP

| Operation | Änderung |
|---|---|
| `qal define` / `qal_define` | beweist (§7.5); Ausgabe zusätzlich `proofs` (Berichte) und `warnings` (E0801 für Tests, E0802); Ablehnung mit E0801 |
| `qal prove [<datei> \| <name>…]` / **neu** `qal_prove {names?}` | `{proofs, proven, refuted, unknown, warnings?}`; aus dem Store mit gespeicherten Ergebnissen; Exit 1 bei `refuted` |
| `qal test [--seed n] [<datei> \| <name>…]` / `qal_test {names?, seed?}` | zusätzlich `properties`; Namen dürfen Funktionen sein (deren Property-Test); `failed_properties` macht das Ergebnis zum Fehler; `passed`/`failed` zählen weiter die geschriebenen Tests |
| `qal_call` | JSON-Argumente für alle Datentypen (Records als Objekte, Varianten `"None"`/`{"Some":[3]}`, Maps als Paarlisten, Bytes als Zahlenlisten) |
| Sprachreferenz | Abschnitte Generics, Closures, Map/Bytes, `compare`, Beweise und Property-Tests; reservierte Built-in-Namen; ≤ 60 Zeilen, ≤ 7,5 KB |

Die MCP-Schnittstelle hat damit 10 Werkzeuge. Neue Felder fehlen in der Ausgabe, wenn sie leer sind; bestehende Felder und Exit-Codes bleiben gleich.

## 11. Z3-Abhängigkeit

- Crate `z3 = "=0.21.1"` (veröffentlicht 2026-09-23), Feature `vendored`: `z3-sys 0.13.1` baut `z3-src 501.0.1` (Z3 5.x) mit CMake aus dem Quelltext und linkt **statisch** (geprüft auf aarch64-apple-darwin: das Testbinary hängt nur von `libc++`, `libiconv` und `libSystem` ab). Das Feature `bundled` ist veraltet und erzeugt eine Build-Warnung, daher `vendored`.
- `[profile.dev.package.z3-sys] opt-level = 3, debug = false`: Z3 wird auch in Debug- und Testbuilds optimiert gebaut (ein Debug-Z3 ist um ein Vielfaches langsamer, 1,4 GB groß und schreibt `.z3-trace`-Dateien).
- Voraussetzungen zum Bauen: CMake und ein C++17-Compiler (Xcode Command Line Tools auf macOS; auf Windows Visual Studio C++, vgl. `CLAUDE.md`). Der erste Build dauert ≈ 4–5 Minuten; danach nur bei Versionswechsel. `Cargo.lock` (versioniert) hält `z3-sys`/`z3-src` fest.

## 12. Bewusst nicht in Phase 2

Effekt-Polymorphie; explizite Typargumente; Typklassen/Traits; Beweise über Float, Strings, Listen, Records (nur approximiert); Terminierungsbeweise; Entfernen bewiesener Laufzeitprüfungen; Property-Tests für Funktionen mit Effekten oder Funktionsparametern; Speicherfreigabe (Phase 3).

## 13. Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| Unsoundness im Beweiser (z. B. Überlauf, Division, Arm-Reihenfolge) | Soundness-Regel §7.2; Laufzeitprüfungen bleiben (ein falscher Beweis ist nie ein falsches Programmverhalten); Tests für jede Semantikkante; unabhängiges Review mit ≈ 7 800 zufälligen Funktionen (kein unsoundes `proven`) |
| Fehlalarme blockieren `define` | nur durch Ausführung bestätigte Gegenbeispiele heißen `refuted` |
| Z3-Laufzeit (nichtlineare Ziele, viele Verpflichtungen) | `rlimit` je Verpflichtung, Wandzeit-Netz, 30-s-Budget je Aufruf, Ergebnisse gespeichert |
| C++-Build von Z3 (Zeit, Toolchain) | einmalig, statisch, optimiert; dokumentiert |
| Monomorphisierung explodiert | §3.2 Grenzen |
| Effekte entkommen über Closures | Effekte im Funktionstyp, Invarianz in benutzerdefinierten generischen Typen |
| Phase 5 braucht schnelle Maps | AVL-Baum statt sortiertem Array |
| Nachspielen nach 1d/1e | Pläne sind gegen den frischen HEAD erzeugt; Änderungen an `qal-cli` sind auf je einen Task am Ende gebündelt |

## 14. Teilpläne

- **2a — Generics, Closures, Map, Bytes:** Kern (AST, typisierte Form, kanonische Form v2, Diagnosen), Text, Prüfung und Monomorphisierung; WASM und Laufzeit (Closures, AVL-Maps, Bytes, Kodierung aller Datentypen); Store-Tests, CLI-JSON-Argumente, Sprachreferenz.
- **2b — Beweise und Property-Tests:** Store-Schema 3 mit `proofs` und Prüfschritt in `define`; Crate `qal-prove` (Z3-Kodierung, Lösen, Bestätigen); Property-Tests; CLI/MCP (`qal prove`, `qal_prove`, Beweise in `define`, Properties in `test`).
