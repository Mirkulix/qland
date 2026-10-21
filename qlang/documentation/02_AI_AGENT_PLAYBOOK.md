# 02 — Das Handbuch für KI-Agenten (Operating Playbook)
> **Die operative Arbeitsanweisung für autonome KIs zur Steuerung von QAL über MCP**

---

## 1. Die 7 MCP-Werkzeuge im Detail

Als KI-System steuerst du QAL nicht über Bash-Befehle, sondern über das **Model Context Protocol (MCP)**:

### `qal_query`
* **Zweck:** Durchsucht den Code-Index nach Namen, Typmustern oder Abhängigkeiten.
* **Schema:** `{"name": "prefix", "type": "fn(Int) -> Bool", "uses": "#hash"}`
* **Beispiel:** Finde alle Zahlungsfunktionen: `{"name": "escrow"}`

### `qal_view`
* **Zweck:** Ruft den exakten Quelltext bestimmter Definitionen ab.
* **Schema:** `{"names": ["escrow_fee", "settle_total"]}`
* **Rückgabe:** Quelltext, Signatur, SHA-256 Hash und Liste aller direkten Abhängigkeiten (`deps`).

### `qal_define`
* **Zweck:** Erstellt oder aktualisiert Definitionen atomar im Merkle-Store.
* **Schema:** `{"source": "fn ... \n test ...", "agent": "Mein-Agenten-Name"}`
* **Verhalten:**  
  1. Parst und typisiert den Code.  
  2. Startet den Z3 SMT-Solver für alle Verträge (`req` / `ens`).  
  3. Bei Erfolg: Speichert atomar und gibt die Hashes zurück.  
  4. Bei logischem Fehler: Verweigert die Speicherung und liefert ein konkretes Gegenbeispiel (`counterexample`).

### `qal_prove`
* **Zweck:** Expliziter Audit aller mathematischen Beweispflichten im Store.
* **Schema:** `{"names": ["debit_balance", "credit_balance"]}`
* **Rückgabe:** Status pro Obligation (`proven`, `refuted`, `unknown`) mit Z3-Begründung.

### `qal_test`
* **Zweck:** Ausführung von Unit-Tests und automatischem Property-Fuzzing.
* **Schema:** `{"names": ["test_escrow_settlement"]}`
* **Verhalten:** Generiert 100 synthetische Zufallseingaben für jede reine Funktion mit Kontrakt und prüft Grenzfälle (64-Bit Integer Bounds).

### `qal_call`
* **Zweck:** Deterministische Ausführung einer Funktion in der isolierten WebAssembly-Sandbox.
* **Schema:** `{"fn": "escrow_net", "args": [50000, 1250], "allow": []}`
* **Verhalten:** Führt die Funktion aus. Verletzt der Aufrufer eine Vorbedingung (`req`), trappt die Sandbox sofort mit `E0501 contract_violation`.

### `qal_history`
* **Zweck:** Abfrage der unveränderlichen Historie und Herkunft einer Funktion.
* **Schema:** `{"name": "settle_total"}`
* **Rückgabe:** Aktionen (`added`, `replaced`), Zeitstempel, Agenten-Signatur und SHA-256 Hashes.

---

## 2. Der autonome Kognitions-Loop (Self-Healing)

Wenn du als Agent Software in QAL entwickelst, folge strikt diesem 4-Schritte-Algorithmus:

```
[Schritt 1: Entwurf] ──► Formuliere req und ens vor dem Funktionsrumpf.
                             │
[Schritt 2: Einreichen] ─► Sende Definition via qal_define.
                             │
[Schritt 3: Diagnose] ──► Prüfe das Ergebnis:
                             ├─ Status "proven": Fertig! Fahre mit Fuzzing (qal_test) fort.
                             └─ Status "contract_refuted (E0801)":
                                   │
[Schritt 4: Auto-Fix] ──► Lies das `counterexample` im JSON.
                             Beispiel: {"amount": 0, "balance": 0, "fee": 1}
                             Frage dich: Warum scheitert der Kontrakt für diese Werte?
                             Erweitere req oder sichere den Rumpf gegen Division/Überlauf ab.
                             Sende korrigierten Code erneut via qal_define.
```

---

## 3. Die wichtigsten Diagnose-Codes

| Code | Diagnose-Typ | Bedeutung & Sofortmaßnahme für die KI |
| :--- | :--- | :--- |
| **`E0801`** | `contract_refuted` | Z3 hat ein Gegenbeispiel gefunden, das die Nachbedingung verletzt. **Lies das Feld `counterexample`** und passe `req` oder den Body an. |
| **`E0802`** | `contract_unproven` | Der Kontrakt ist für den Solver zu komplex (z. B. nichtlineare Terme). Vereinfache auf lineare Ganzzahlarithmetik. |
| **`E0501`** | `contract_violation` | Laufzeit-Vorbedingung in `qal_call` verletzt. Der Aufrufer erfüllt `req` nicht. |
| **`E0502`** | `int_overflow` | 64-Bit Ganzzahlüberlauf zur Laufzeit getrappt. Füge obere Schranken in `req` ein. |
| **`E0404`** | `alias_reference` | Zwei Funktionen teilen denselben Bytecode-Hash. Führe sie zu einer kanonischen Funktion zusammen. |
| **`E0201`** | `unknown_name` | Name existiert nicht im Store. Nutze `qal_query`, um den genauen Namen zu finden. |
