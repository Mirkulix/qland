# 04 — Formale Verifikation & der Z3 SMT-Solver
> **Mathematische Beweise von Softwarekontrakten, Presburger-Arithmetik und Gegenbeispiele**

---

## 1. Das Kontrakt-Modell: `req` und `ens`

In QAL deklarieren Funktionen formale Verträge:
* **`req` (Vorbedingung / Precondition):** Was der Aufrufer vor dem Aufruf garantieren muss.
* **`ens` (Nachbedingung / Postcondition):** Was die Funktion nach ihrer Ausführung garantiert. Die Pseudovariable `r` steht für den Rückgabewert der Funktion.

```qal
fn safe_div(a: Int, b: Int) -> Int
  req b > 0 && a >= 0
  ens r >= 0 && r <= a
= a / b
```

---

## 2. Der integrierte Z3 SMT-Theorembeweiser

Beim Aufruf von `qal_define` übersetzt der QAL-Compiler die Kontrakte in **Presburger Lineare Arithmetik (QF_LIA - Quantifier-Free Linear Integer Arithmetic)** und übergibt sie an den eingebetteten Z3 SMT-Solver.

### Was Z3 exakt beweisen kann:
* Lineare Ganzzahl-Relationen: `+`, `-`, Vergleiche (`<`, `<=`, `>`, `>=`, `==`, `!=`).
* Multiplikation und Division mit Konstanten (z. B. `(amount * 250) / 10000`).
* Aussagenlogische Verknüpfungen: `&&`, `||`, `!`.
* Wert-Erhaltungssätze über algebraische Identitäten:  
  `(s_before - amount) + (r_before + (amount - fee)) + fee ≡ s_before + r_before`.

### Was Z3 approximiert:
* Beliebige nichtlineare Terme (`a * b` mit zwei freien Variablen).
* Tief verschachtelte Record-Felder. Hier empfiehlt sich die Deklaration skalarer Kontrakte auf `Int`-Ebene.

---

## 3. Der Widerlegungs-Mechanismus (`contract_refuted`)

Wenn ein Kontrakt mathematisch ungültig ist, lehnt Z3 den Code ab und liefert eine Belegung der Variablen, bei der die Nachbedingung verletzt wird:

### Beispiel aus der Praxis:
```qal
# Fehlerhafter Versuch der KI:
fn flawed_calc(balance: Int, amount: Int, fee: Int) -> Int
  req balance >= 0 && amount >= 0 && fee >= 0
  ens r >= 0 && r + fee == balance - amount
= balance - amount - fee
```

**Diagnose des QAL-Compilers:**
```json
{
  "code": "E0801",
  "kind": "contract_refuted",
  "def": "flawed_calc",
  "path": "ens[0]",
  "message": "`ens r >= 0 && r + fee == balance - amount` does not hold for the counterexample",
  "counterexample": {
    "amount": 0,
    "balance": 0,
    "fee": 1
  }
}
```

**Erklärung für KIs:**
Z3 beweist, dass bei `balance = 0`, `amount = 0`, `fee = 1` das Ergebnis `0 - 0 - 1 = -1` ist. Dies verletzt die Nachbedingung `r >= 0`.  
**Sofortige Abhilfe:** Der Agent ergänzt `req balance >= amount + fee` und reicht den Code erneut ein.

---

## 4. Beweis-Caching mit `confirm_fuel`

Um Wiederholungsprüfungen zu beschleunigen, besitzt QAL einen integrierten Beweis-Cache:
* Der Cache-Schlüssel umfasst: Quelltext-Hash, Kontrakt-Hash, Z3-Version und das Bestätigungsbudget (`confirm_fuel`).
* Ein unvollständiges Ergebnis mit kleinem Budget blockiert niemals eine spätere Prüfung mit größerem Budget.
