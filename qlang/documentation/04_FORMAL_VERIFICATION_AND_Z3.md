# 04 — Formale Verifikation & der Z3 SMT-Solver
> **Mathematische Beweise von Softwarekontrakten, QF_LIA-Gültigkeitsbereich und Gegenbeispiele**

---

## 1. Das Kontrakt-Modell: `req` und `ens`

In QAL deklarieren Funktionen formale Verträge:
* **`req` (Vorbedingung / Precondition):** Was der Aufrufer vor dem Aufruf sicherstellen muss.
* **`ens` (Nachbedingung / Postcondition):** Was die Funktion nach ihrer Ausführung garantiert. Die Pseudovariable `r` repräsentiert den Rückgabewert.

```qal
fn safe_div(a: Int, b: Int) -> Int
  req b > 0 && a >= 0
  ens r >= 0 && r <= a
= a / b
```

---

## 2. Der genaue mathematische Gültigkeitsbereich (Z3 SMT Presburger QF_LIA)

Der QAL-Compiler nutzt für Verträge den Z3 SMT-Solver im Fragment der **quantorenfreien linearen Ganzzahlarithmetik (QF_LIA)**:

### Was exakt bewiesen wird (Garantiebereich):
* **Lineare Arithmetik ist vollständig garantiert:** Addition (`+`), Subtraktion (`-`), Vergleiche (`<`, `<=`, `>`, `>=`, `==`, `!=`).
* Multiplikation und Division mit ganzzahligen Konstanten (z. B. `(amount * 250) / 10000`).
* Boolesche Aussagenlogik: `&&`, `||`, `!`.
* Zusammengesetzte Aufrufe: Beim Beweis einer Funktion werden **die Verträge (ens) der aufgerufenen Hilfsfunktionen** genutzt, nicht deren Rümpfe.

### Was approximiert wird & Grenzen:
* **Nichtlineare Arithmetik:** Lineare Relationen sind garantiert; nichtlineare Terme (z. B. die Multiplikation zweier freier Variablen `a * b`) sind im Solver zwar punktuell möglich, liegen aber außerhalb von Presburger QF_LIA. Sie sind unvollständig und führen bei Nichtlösbarkeit deterministisch zu `unknown` (`E0802`).
* **Record-Felder:** Record-Strukturen werden im SMT-Solver approximiert. Findet Z3 einen potentiellen Gegenbeispiel-Kandidaten, führt QAL diesen auf der WASM-Runtime aus, um das Gegenbeispiel zu verifizieren. Kann der Kandidat nicht bestätigt werden, wird der Vertrag nicht als refuted markiert, sondern bleibt `unknown` (`E0802 contract_unproven`) und wird zur Laufzeit dynamisch geprüft.

---

## 3. Der Widerlegungs-Mechanismus (`E0801 contract_refuted`)

Widerlegt Z3 einen Vertrag, wird die Definition **nicht im Store abgelegt**. Die JSON-Diagnose enthält das exakte Gegenbeispiel:

```json
{
  "code": "E0801",
  "kind": "contract_refuted",
  "def": "flawed_escrow_settle",
  "path": "ens[0]",
  "message": "`ens r >= 0 && r + fee == balance - amount` (ens[0]) does not hold for the counterexample",
  "hints": [
    "running the function on the counterexample violates the contract: fix the body, the contract, or add a `req`"
  ],
  "counterexample": {
    "amount": 0,
    "balance": 0,
    "fee": 1
  }
}
```

*Verhalten:* Die Datenbank bleibt vollkommen unberührt. Der Agent nutzt die Variablenbelegung des Gegenbeispiels, um die Vorbedingungen anzupassen.

---

## 4. Der Beweis-Cache (`proofs`-Tabelle) & `confirm_fuel`

Beweisergebnisse werden in der SQLite-Tabelle `proofs` gecacht. Der Cache-Schlüssel setzt sich zusammen aus:
1. Dem Inhalts-Hash der Definition (welcher callees über Hashes bindet),
2. Dem Prover-Key (Z3-Version, Solver-Konfiguration),
3. Dem Bestätigungsbudget (`confirm_fuel`).

> **Garantie (Work Package 3):** Ein mit kleinem Budget unbestätigter Eintrag verhindert niemals eine erneute Prüfung mit höherem `confirm_fuel`.
