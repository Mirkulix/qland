# 02 — Das Handbuch für KI-Agenten (Operating Playbook)
> **Die operative Arbeitsanweisung für autonome KIs zur Steuerung von QAL über MCP**

---

## 1. Die 12 MCP-Werkzeuge (Model Context Protocol)

Der native QAL MCP-Server stellt exakt **12 Werkzeuge** bereit:

| Werkzeug | Parameter | Funktion |
| :--- | :--- | :--- |
| **`qal_define`** | `source` (Str), `agent` (Opt[Str]) | Fügt Typen, Funktionen und Tests atomar hinzu. Prüft Typen und startet den Z3-Beweiser für Verträge. |
| **`qal_prove`** | `names` (List[Str]) | Gibt den formalen Beweisstatus der benannten Funktionen zurück (`proven`, `refuted`, `unknown`). |
| **`qal_test`** | `names` (Opt[List[Str]]) | Führt Unit-Tests aus und generiert 100 synthetische Property-Testläufe je Funktion mit Kontrakt. |
| **`qal_call`** | `fn` (Str), `args` (List[Any]), `allow` (Opt[List[Str]]) | Führt eine Funktion in der WASM-Sandbox aus. Trappt bei Kontraktverletzung (`E0501`). |
| **`qal_run`** | `file` / `name`, `args` (List[Str]), `allow` (Opt[List[Str]]) | Startet einen Programmeinstiegspunkt (`fn main(args: List[Str]) -> Int !...`). |
| **`qal_view`** | `names` (List[Str]) | Liefert den Quelltext, die Signatur, den Hash und die direkten Abhängigkeiten (`deps`) der Namen. |
| **`qal_query`** | `name` (Opt[Str]), `type` (Opt[Str]), `uses` (Opt[Str]) | Durchsucht den Definitions-Index nach Namenspräfix, Signaturmuster oder Referenz. |
| **`qal_request`**| `handler` (Str), `requests` (Str/Pfad), `allow` (Opt) | Testet Service-Handler (`fn handle(HttpRequest) -> HttpResponse`) mit Test-Requests. |
| **`qal_rename`** | `old` (Str), `new` (Str), `agent` (Opt[Str]), `breaking` (Bool)| Benennt eine Definition um. Da Hashes unverändert bleiben, folgen alle Referenzen automatisch. |
| **`qal_delete`** | `names` (List[Str]), `agent` (Opt[Str]), `breaking` (Bool)| Entfernt ungenutzte Definitionen. Schlägt fehl (`has_dependents`), wenn andere Namen sie nutzen. |
| **`qal_history`**| `name` (Str) | Gibt die chronologische Änderungshistorie des Namens mit Hashes, Agent und Zeitstempel aus. |
| **`qal_build`**  | `name` (Str) | Kompiliert eine Funktion zu einem eigenständigen Wasm-Artefakt. |

---

## 2. Der autonome Kognitions-Loop (Self-Healing)

Wenn ein Agent ein Software-Problem in QAL löst, durchläuft er folgenden deterministischen Zyklus:

```
[1. Inspektion] ──────► qal_query / qal_view
                             │
[2. Spezifikation] ───► Kontrakte formulieren: req (Vorbedingung), ens (Nachbedingung)
                             │
[3. Einreichen] ──────► qal_define aufrufen
                             │
[4. Diagnose] ────────► Ergebnis auswerten:
                             ├─ Status "proven": Fertig!
                             ├─ Status "contract_refuted (E0801)":
                             │    Lies `counterexample` (z. B. {"x": 0, "y": 1}).
                             │    Analysiere, warum die Nachbedingung verletzt ist.
                             │    Passe req oder den Funktionsrumpf an und sende erneut.
                             └─ Status "alias_reference (E0404)":
                                  Eine Definition referenziert zwei verschiedene Namen, 
                                  die denselben Definitionshash teilen.
                                  Lösung: Verwende nur einen der beiden Alias-Namen.
```

---

## 3. Die exakte Bedeutung von `E0404 alias_reference`

Im QAL-Store dürfen zwei verschiedene Namen (`name_a`, `name_b`) durchaus auf denselben Bytecode-Hash zeigen.  
Der Fehler **`E0404 alias_reference`** tritt ausschließlich auf, wenn:
* Eine einzelne Definition (z. B. ein Test oder eine Funktion) **beide Alias-Namen gleichzeitig im Quelltext referenziert**.
* **Ursache:** Die interne Referenztabelle `names.refs` bildet jeden referenzierten Hash auf genau einen Namen ab.
* **Handlungsanweisung für KIs:** Beziehe dich in deiner Definition auf nur einen der beiden Alias-Namen.

---

## 4. Häufige Diagnose-Codes und Handlungsanweisungen

| Code | Diagnose-Typ | Ursache & Sofortmaßnahme für die KI |
| :--- | :--- | :--- |
| **`E0801`** | `contract_refuted` | Z3 hat ein Gegenbeispiel gefunden. Lies das Feld `counterexample` im JSON. Schärfe `req` ein oder behebe den Grenzwertfehler im Body. |
| **`E0802`** | `contract_unproven` | Der Kontrakt konnte nicht bewiesen werden (z. B. komplexe Record-Projektionen oder Nichtlinearität). Vereinfache auf skalare Relationen auf `Int` / `Bool`. |
| **`E0501`** | `contract_violation` | Ein Aufrufer hat in `qal_call` die Vorbedingung (`req`) einer Funktion verletzt. |
| **`E0502`** | `int_overflow` | Laufzeit-Überlauf bei 64-Bit Ganzzahloperationen (`+`, `-`, `*`). Ergänze realistische Schranken in `req`. |
| **`E0404`** | `alias_reference` | Gleichzeitiges Referenzieren zweier Namen desselben Hashes in einer Definition. |
| **`E0201`** | `unknown_name` | Ein Bezeichner ist weder im Store noch in der Prelude vorhanden. Nutze `qal_query`. |
| **`E0301`** | `type_mismatch` | Typsignatur weicht ab. Untersuche die genaue Signatur mit `qal_view`. |
| **`E0901`** | `lease_conflict` | Ein anderer Agent hält aktuell eine Schreibsperre auf diesem Namen. |
