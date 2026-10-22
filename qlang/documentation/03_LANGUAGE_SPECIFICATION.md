# 03 — Sprachspezifikation QAL
> **Syntax, Typen, Ausdrücke und das algebraische Effektsystem**

---

## 1. Grundlegende Spracheigenschaften

* **Rein funktionale Basis:** Alles ist ein Ausdruck (`expression`). Es gibt keine Anweisungen (`statements`), keine Null-Werte, keine impliziten Typumwandlungen und keine ungebundenen Schleifen (`for`/`while`).
* **Rekursion:** Schleifen werden über Endrekursion realisiert. Der WASM-Compiler optimiert echte Endrekursionen automatisch in O(1)-Schleifen.
* **Unveränderlichkeit:** Alle Variablenbindungen (`let`) sind strikt unveränderlich.

---

## 2. Das Typ-System

### Primitive Typen
* `Int`: Vorzeichenbehaftete 64-Bit Ganzzahl (`i64`). Überlauf und Division durch Null trappen deterministisch.
* `Float`: 64-Bit Fließkommazahl nach IEEE 754.
* `Bool`: `true` oder `false`.
* `Str`: Unveränderliche UTF-8 Zeichenkette.
* `Bytes`: Bytefolge.
* `Unit`: Leerer Wert, geschrieben als `()`.

### Zusammengesetzte Typen
* **Records (Datensätze):**
  ```qal
  type Account = { id: Int, owner: Str, balance: Int }
  # Konstruktion:
  let acc = Account { id: 1, owner: "Alice", balance: 500 }
  # Zugriff:
  acc.balance
  ```
* **Enums / Summentypen (Algebraische Datentypen):**
  ```qal
  type Vote = Yes | No | Abstain
  type Shape = Circle(Int) | Rect(Int, Int) | Point
  ```
* **Generics:**
  ```qal
  type Pair[A, B] = { fst: A, snd: B }
  fn swap[A, B](p: Pair[A, B]) -> Pair[B, A] = Pair { fst: p.snd, snd: p.fst }
  ```
* **Standard-Prelude-Typen:**
  * `Option[T] = Some(T) | None`
  * `Result[T, E] = Ok(T) | Err(E)`

---

## 3. Ausdrücke & Kontrollfluss

### Bedingungen (`if-then-else`)
Der `else`-Zweig ist stets verpflichtend:
```qal
if a > b then a else b
```

### Pattern Matching (`match`)
Matches müssen vollständig (exhaustive) sein:
```qal
match result {
  Ok(value) if value > 0 => value,
  Ok(_)                  => 0,
  Err(err_msg)           => -1
}
```

### Sequenzierung
Ausdrücke werden mit Semikolon getrennt. Der letzte Ausdruck bestimmt das Gesamtergebnis:
```qal
{
  print("Berechnung gestartet");
  let res = a * b in res + 1
}
```

---

## 4. Das algebraische Effektsystem (Zero-Trust Sandbox)

Funktionen, die mit der Außenwelt interagieren, müssen ihre Effekte explizit im Typ deklarieren:

| Effekt | Erlaubte Operationen | Beispiel-Signatur |
| :--- | :--- | :--- |
| *Kein Effekt* | **Reine Funktion:** Mathematisch beweisbar, deterministisch, kein I/O. | `fn add(a: Int, b: Int) -> Int` |
| `!io` | Terminal-Ausgabe & Eingabe (`print`, `read_line`, `read_all`). | `fn greet(name: Str) -> Unit !io` |
| `!fs` | Dateisystem-Zugriffe innerhalb des Projektverzeichnisses (`read_file`, `write_file`). | `fn save(data: Str) -> Result[Unit, Str] !fs` |
| `!db` | SQLite-Datenbankabfragen (`db_exec`, `db_query`). | `fn get_user(id: Int) -> DbRows !db` |
| `!net` | Ausgehende HTTP-Anfragen (`http`). | `fn fetch(c: HttpCall) -> Result[HttpResponse, NetError] !net` |
| `!time` | Zeitstempel (`now_ms`). | `fn timestamp() -> Int !time` |
| `!rand` | Pseudozufallszahlen (`rand_int`). | `fn roll_dice() -> Int !rand` |

> **Effekt-Garantie (Statische Typ- und Host-Sandbox-Schranke):**
> Eine Funktion ohne deklarierte Effekte (`!io`, `!fs`, `!db`, `!net`) kann weder I/O-Operationen ausführen noch Daten ins Netzwerk übertragen. Diese Garantie wird auf zwei Ebenen erzwungen:
> 1. **Statische Typprüfung:** Der Compiler weist Aufrufe von effektbehafteten Hilfsfunktionen in reinen Kontexten mit `E0301 effect_violation` vor der Codegenerierung ab.
> 2. **Wasmtime-Sandbox:** Das erzeugte Wasm-Modul enthält für reine Funktionen keinerlei Host-Imports für Dateisystem-, Netzwerk- oder Datenbank-Syscalls. Nicht deklarierte Fähigkeiten existieren im Wasm-Gäste-Modul physisch nicht.
