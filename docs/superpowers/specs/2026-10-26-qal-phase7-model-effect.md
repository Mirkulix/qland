# QAL Phase 7 — Der Effekt `model`: Spezifikation & Implementierungsplan

**Datum:** 2026-10-26  
**Status:** In Umsetzung (Phase 7 der Roadmap §7 / `2026-10-23-qal-weg-design.md`)  
**Ziel:** KI-Modellaufrufe als deterministischer, budgetierter Host-Effekt in QAL.

---

## 1. Übersicht & Zielsetzung

Bisher konnte QAL von KI-Agenten geschrieben und verifiziert werden, aber Programme in QAL konnten selbst keine Modelle aufrufen. Phase 7 schließt diese Lücke nach dem Prinzip **"Determinismus mit Budget"**:

1. **Host-Effekt `!model`:** Modellaufrufe sind keine unkontrollierten Nebeneffekte, sondern deklarierte Effekte im Typ- und Rechtesystem.
2. **Budget als Sicherheitsgrenze:** Jeder Aufruf wird cent-genau gegen ein Hard-Budget (`QAL_MODEL_BUDGET_EUR`, Vorgabe: 20 €) gezählt. Überschreitungen brechen fail-closed ab.
3. **Aufzeichnung & Replay:** Jeder Modellaufruf wird mit `(hash(prompt), hash(schema))` und Antwort in SQLite protokolliert. `qal test` und `qal run --replay` nutzen aufgezeichnete Antworten – Tests von KI-Orchestrierungen sind **100 % deterministisch, reproduzierbar und kostenlos**.
4. **Erster Produktiveinsatz:** Der Benchmark-Auswerter wird in QAL implementiert und analysiert Agenten-Transkripte.

---

## 2. Sprachanbindung & Typen

### 2.1 Prelude-Typen (`qal-core/src/prelude.rs`)

```text
type ModelResponse = {
  text: Str,
  model: Str,
  in_tokens: Int,
  out_tokens: Int,
  cost_microcents: Int
}

type ModelError =
  | ModelDenied(Str)
  | ModelBudgetExceeded(Int)
  | ModelFailed(Str)
  | ModelInvalid(Str)
  | ModelTimeout
```

* `cost_microcents`: 1 Cent = 10.000 Microcents (für sub-cent genaue Abrechnung bei günstigen Modellen wie Haiku).

### 2.2 Builtin-Funktion

```text
fn model_ask(prompt: Str, schema: Str) -> Result[ModelResponse, ModelError] !model
```

* `prompt`: Die an das Modell übergebene Eingabe.
* `schema`: Optional ein JSON-Schema-String. Wenn nicht leer, instruiert der Host das Modell entsprechend und prüft die Antwort.
* `effect`: `"model"`.
* Host-Import im WebAssembly: `"model.ask"`.

---

## 3. Host-Laufzeit & Sandbox (`qal-run`)

### 3.1 Berechtigungen & CLI
* `--allow model` (bzw. über MCP: `"allow": ["model"]`). Ohne Freigabe bricht der Aufruf sofort mit `Err(ModelDenied(...))` ab.

### 3.2 Authentifizierung & Budget
* Schlüssel: Aus `~/.config/qal/secrets.env` (`ANTHROPIC_API_KEY`) oder der Umgebung.
* Budget: `QAL_MODEL_BUDGET_EUR` (Default 20 €).
* Host führt Buch über die kumulierten Ausgaben in `.qal/model_budget.json`.
* Modell: Standard ist `claude-haiku-5-5` ($0.10 / M in, $0.50 / M out).

### 3.3 Cache- & Replay-Mechanismus
* In `.qal/store.db` (oder lokaler SQLite) wird die Tabelle `model_calls` geführt:
  ```sql
  CREATE TABLE IF NOT EXISTS model_calls (
    prompt_hash TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    response_text TEXT NOT NULL,
    in_tokens INTEGER NOT NULL,
    out_tokens INTEGER NOT NULL,
    cost_microcents INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (prompt_hash, schema_hash, model)
  );
  ```
* Im Modus `--replay` oder während `qal test` wird **niemals** das Netzwerk kontaktiert; Antworten müssen aus dem Store stammen.

---

## 4. Standardbibliothek `std.model`

In `std/src/` bzw. `std.qalpkg`:

```text
fn std.model.ask(prompt: Str, schema: Str) -> Result[Str, ModelError] !model
  = match model_ask(prompt, schema) {
      Ok(resp) => Ok(resp.text),
      Err(err) => Err(err)
    }

fn std.model.ask_json(prompt: Str, schema: Str) -> Result[std.json.Json, ModelError] !model
  = match model_ask(prompt, schema) {
      Ok(resp) => match std.json.parse(resp.text) {
        Ok(j) => Ok(j),
        Err(e) => Err(ModelInvalid(std.json.Error_message(e)))
      },
      Err(err) => Err(err)
    }

fn std.model.ask_full(prompt: Str, schema: Str) -> Result[ModelResponse, ModelError] !model
  = model_ask(prompt, schema)
```

---

## 5. Implementierungs-Aufgaben

- [ ] **Aufgabe 1: Core & AST (`qal-core`, `qal-check`, `qal-wasm`)**
  - `KNOWN_EFFECTS`: Hinzufügen von `"model"`.
  - Prelude-Typen: `ModelResponse`, `ModelError`.
  - Builtin `ModelAsk` mit Signatur und Effekt `"model"`.
  - WASM-Host-Import `"model.ask"`.
- [ ] **Aufgabe 2: Host-Integration (`qal-run`)**
  - Implementierung der Host-Funktion `model.ask` in Wasmtime.
  - Budget-Prüfung, `secrets.env`-Parsing, Anthropic-API-Client mit `ureq`.
  - Aufzeichnung & Replay via SQLite (`model_calls`).
- [ ] **Aufgabe 3: CLI & MCP (`qal-cli`, `qal-serve`)**
  - `--allow model` Support und Validierung.
  - MCP-Permissions-Matrix (`policy.rs`, `mcp.rs`).
- [ ] **Aufgabe 4: Self-Hosted Compiler (`selfhost/`)**
  - Eintrag von `QcBModelAsk` und Effekt `"model"` im QAL-Compiler.
  - Re-Bootstrap des Fixpunkts (Stage 1 == Stage 2 == Stage 3).
- [ ] **Aufgabe 5: Standardbibliothek & Tests (`std.model`)**
  - `std.model` Modul mit Tests für deterministisches Replay, Offline-Mocks und Budgetgrenzen.
  - Erster Einsatz: Benchmark-Auswerter-Skript.

---

## 6. Verifikation (2026-10-26, nach `4ac0e50b`)

**Ein echter Aufruf**, von `std.model.ask_full` aus einem QAL-Programm mit `--allow io,model`:

- **Antwort:** `pong` von `claude-haiku-5-5`, 20 Eingabe- und 4 Ausgabe-Tokens.
- **Kosten:** 4 Millionstel € (0,0004 Cent); das Budget in `.qal/model.db` steht auf 4.
- **Wiederholung:** Der zweite Aufruf kommt aus der Aufzeichnung. Das Budget bleibt bei 4, es gibt
  genau einen Eintrag in `model_calls`, und `store.db` enthält keine Modell-Tabellen.
- **`--replay`:** Ohne Schlüssel und ohne `HOME` liefert er die aufgezeichnete Antwort; ein nicht
  aufgezeichneter Aufruf endet mit `ModelFailed("no recorded model response found for replay")`.
- **Ohne `--allow model`:** E0508 `permission_denied`.

**Offene Lücke:** §1.3 verspricht, `qal test` spiele Modellantworten aus der Aufzeichnung ab. Ein
`test` darf in QAL aber keine Effekte haben: `std.model.ask` in einem Test ergibt E0301, und der
Hinweis, `!model` zur Signatur hinzuzufügen, ist für einen Test nicht umsetzbar. Deterministisch
abspielen lässt sich ein KI-Ablauf heute nur über `qal run --replay`.

**Möglichkeiten:**

- erlaubte Effekte für Tests (`test t !model = ...`), immer im Replay-Modus;
- ein Test-Harness, das `run --replay` mit erwarteter Ausgabe vergleicht.
