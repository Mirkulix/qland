# QAL — Fahrplan bis "komplett fertig"

**Datum:** 2026-10-15
**Auftrag des Nutzers:** "mach es fertig komplett. Ohne Ausnahme!" — alle Phasen 1–5 der Spezifikation `2026-10-15-qal-design.md`. Der Nutzer delegiert alle Designentscheidungen ("Du entscheidest es selber"); Entscheidungen werden getroffen, begründet und im jeweiligen Plan/Ledger dokumentiert, ohne Zwischenfreigaben.

**Arbeitsweise je Teil:** Spezifikation (falls neu) → Implementierungsplan (Code vorab im Scratchpad verifiziert) → subagent-getriebene Umsetzung mit Review pro Aufgabe → Abschluss-Review → Push auf Branch `Mac` (PR Mirkulix/qland#2 bzw. Folge-PRs).

**Status-Legende:** ☐ offen · ◐ in Arbeit · ☑ fertig (alle Kriterien nachgewiesen)

| # | Teil | Fertig, wenn (nachweisbar) | Status |
|---|---|---|---|
| 1a | Kern-Pipeline | Text → Prüfung → WASM → Ausführung für Int/Bool, Contracts, JSON-Diagnosen; 50 Tests grün | ☑ |
| 1b | Daten, Speicher, Effekte | Float/Str/List/Option/Result/Records/Enums/`match`/Sequenzen/Built-ins (§3.5); Bump-Allocator mit `out_of_memory`; Effekte als Host-Importe nur für deklarierte Effekte; `qal run <datei> -- args` mit `--allow`; ein Programm ohne `fs` kann nachweislich keine Datei öffnen; vertagte 1a-Punkte erledigt | ☑ (135 Tests, 0 Warnungen; Commits 55ac2e05..42e3f547) |
| 1c | Graph-Speicher | Kanonische Form + SHA-256-Hash; Umbenennen ändert keinen Hash; gleiche Logik = gleicher Hash; SQLite-Store `.qal/store.db` mit Namen, Abhängigkeiten, Historie; `define` ersetzt mit automatischem Neuverknüpfen bei gleicher Signatur, sonst Fehler mit Liste der Abhängigen; Effekte als sortierte Menge | ☑ (197 Tests, 0 Warnungen; Commits af0feddc..dcf4410b) |
| 1d | Agenten-Schnittstelle | `qal mcp` (stdio) mit `qal_define, qal_view, qal_query, qal_test, qal_run, qal_build, qal_rename, qal_history`; kompakte Sprachreferenz in den Tool-Beschreibungen; keine Panik auf irgendeinem MCP-Pfad; in diesem Repo per `.mcp.json` registriert und von einem Claude-Code-Agenten real benutzt | ☑ (242 Tests, 0 Warnungen; Commits 134f0811..47e08f06; Probelauf: Agent ohne QAL-Vorwissen löste FizzBuzz über `sh qal/mcp.sh` mit 6 Tool-Aufrufen, 1 eigener Fehlversuch; die native `/mcp`-Verbindung in Claude Code braucht einen Neustart und deine Freigabe) |
| 1e | Benchmark | 20 Aufgaben mit versteckten Tests + Referenzlösungen in QAL; Durchlauf mit Claude-Code-Subagenten (QAL über MCP vs. Rust über Dateien + cargo), gemessen: Erfolg, Tokens, Tool-Aufrufe, Fehlversuche; `bench/report.md` | ☑ (80/80 Läufe gültig; QAL 39/40 gelöst, 69,5k Tokens je Lösung; Rust 40/40, 60,1k; Rust gewinnt 12 Aufgaben, 8 unentschieden — These in dieser Form nicht bestätigt, Folgerungen für Phase 2 in `bench/discussion.md`) |
| 2 | Beweise & Ausdruckskraft | `req`/`ens` werden für lineare Ganzzahl-Arithmetik mit Z3 statisch bewiesen (Gegenbeispiel bei Widerlegung, Laufzeitprüfung als Rückfall); benutzerdefinierte Generics (Monomorphisierung); Closures; generierte Property-Tests | ☑ (335 Tests, 0 Warnungen; 2a 71ff0d47..632da8ae, 2b fd621402..75c0432f, Review-Fixes 46983eb5..c6e20361; zusätzlich aus dem 1e-Benchmark: Sequenzen verwerfen Werte außer `Result`, Hinweise für fremde Typnamen und Built-ins als Werte; Z3 statisch gelinkt) |
| 3 | Standardbibliothek & Services | Effekte `net` (HTTP-Server/Client über Host) und `db` (SQLite über Host); Speicherfreigabe für langlaufende Services (Arena pro Request); Standardbibliothek **in QAL** (JSON, HTTP-Routing, Strings) — geschrieben von Agenten über MCP; Beispiel-Backend (Todo-API) läuft und ist getestet | ☑ (457 Tests, 0 Warnungen, Clippy-Restriktionsprüfung sauber; 3a 3e6ea496..b08ee69e, 3b 5e026a39..f0d29171, 3c 65673e41..3e8db580; std (6 Module: str, url, json, http, route, db) und Todo-Backend von Agenten über MCP geschrieben, Abnahme jeweils im ersten Versuch: std 6/6 Module, 264 versteckte Tests; Todo 47/47 versteckte Szenarien mit 10 Tool-Aufrufen; `todo.qalpkg` reproduziert beim Import in ein frisches Projekt jeden Hash; 54 Szenarien über `qal request` und HTTP-Harness gegen `qal serve` grün; RSS über 10 000 `GET /todos` nach den ersten 1 000: +0,0 % (Release 19 888 KiB, ~7 500 Anfragen/s auf einer Verbindung)) |
| 4 | Agenten-Team | Mehrere Agenten arbeiten gleichzeitig über MCP an einer Codebase: Sperren pro Definition, Rollen (Architekt, Implementierer, Verifizierer, Reviewer), Merge, signierte Änderungen (Ed25519); Demo: Team baut einen Service parallel | ☑ (568 Tests, 0 Warnungen, Clippy-Restriktion sauber; 4a 952312ef..39aba2c4, 4b 37592d3b..08928a82, 4c a7b27772..ff4cc508 + Review-Fixes 09f544ec..8957923e, 4d echter Teamlauf auf `p4d-real`: 6 Sonnet-Agenten über MCP, 20/20 Szenarien (15/15 versteckt), `qal verify` ok, Kriterium 6 erfüllt; Solo-Vergleich ebenfalls 20/20 mit ≈ 1/6 der Tokens — siehe `bench/team/real/report.md`) |
| 5 | Self-Hosting | Ein QAL-Compiler, geschrieben in QAL, übersetzt die QAL-Testsammlung zu identischem Verhalten; Bootstrap-Fixpunkt (Stage 2 == Stage 3, byte-identisches WASM); danach ist Rust nur noch für den Host (wasmtime-Einbettung) nötig | ◐ (In Arbeit / Ehrlicher Status: Fixpunkt Stage 1 == Stage 2 == Stage 3 für `qalc.qal` ist byte-identisch verifiziert; Saatmodul `SEED_SHA256` aktiv. Ausnahme: `selfhost/qalc.qalpkg` ist noch nicht Teil des Korpus, da die Speicheroperationen TYPED, DEFINE und HASH als Einzelausführung laufen und 1 GiB Speicher überschreiten; die FORK/JOIN-Aufteilung für diese Operationen steht noch aus.) |

## Externe Voraussetzungen

- **Keine API-Schlüssel nötig:** Benchmark (1e) und Team (4) nutzen Claude-Code-Subagenten in dieser Umgebung als Agenten, deren Token- und Tool-Verbrauch die Umgebung misst. Eine Anbindung externer LLM-Provider (aus `qlang-runtime` Provider-Registry) ist optional und kein Fertig-Kriterium.
- **Z3 (Phase 2):** über das Crate `z3` mit statischem Build (`static-link-z3`), keine Systeminstallation.

## Ehrliche Risiken

- Phase 5 (Self-Hosting) ist der größte Brocken: Der Compiler muss in QAL ausdrückbar sein (Maps, Bytes, Rekursion über große ASTs). Phase 2–3 werden deshalb so geschnitten, dass sie die dafür nötigen Sprachmittel liefern.
- Die Benchmark-Messung (1e) kann zeigen, dass QAL anfangs schlechter ist. Das ist kein Abbruchgrund, sondern steuert Phase 2.

## Übertrag in spätere Teile (aus den Reviews)

**Pflicht in 1c:**
- Größe inferierter Typen begrenzen (z. B. 4096 Knoten, Diagnose aus der E0207-Familie): geteilte Typvariablen (`Ok`/`Err` pro `let`) lassen Typen exponentiell wachsen; Hashing läuft über Typen.
- Drucker total machen für nicht vom Parser erzeugte ASTs (negative Literale, NaN/inf, leere Sequenz/`match`/Enum) — Voraussetzung für die kanonische Form.
- Konsistenztest `Builtin::NAMES` ↔ `Builtin::name()`; Effekte als sortierte, duplikatfreie Menge.

**Pflicht in 1d:**
- Aus 1c: Optionen vor dem Namen im Speichermodus; Datei-vs-Name-Mehrdeutigkeit (`foo.qal`/`foo`/`./foo`); Speicher erst nach Optionsprüfung laden; `qal test <fn>` mit klarer Meldung; `query --type` mit ungültigem Muster → Diagnose; `canon::resolve` fail-closed; geschlossenes stdout ohne Panik; Hinweis „umbenannt in X“ bei `history`/`view` alter Namen; Namensbindung (Nutzer folgen dem Namen, gegen den sie geschrieben wurden) und E0404 in der Tool-Referenz erklären; `define` über Definitionsarten hinweg (fn → test) nicht still ersetzen.
- Kein `expect`/`unwrap`/`unreachable!` auf MCP-erreichbaren Pfaden; unbekannter Einstiegspunkt muss vor `compile` abgelehnt werden.
- FIFO/Gerätedateien im erlaubten Ordner dürfen den Host nicht blockieren (nur reguläre Dateien, O_NONBLOCK + fstat).
- CLI: Optionen auch vor dem Dateinamen akzeptieren; `check`/`test` lehnen unbekannte Zusatzargumente ab.

**Pflicht vor 1e (aus dem 1d-Abschluss-Review):**
- Built-ins: `read_all() -> Str !io`, Ordnung auf `Str` (`<` usw.) plus `sort_str`, `char_code`/`from_char_code`, `to_upper`/`to_lower`/`replace`/`index_of`.
- Benchmark-Harness: versteckte Tests über `test`-Definitionen oder `main` + stdin (`qal_call` nimmt nur Skalare und Listen); Aufgabengrößen passend zu den Grenzen (Fuel, 10 s, 64 KiB Ausgabe, quadratisches `push`/`++` ~8 000).

**Phase 3:**
- `fs`-Schreiben über verifizierten Elternordner (`openat`), damit auch der Rest-Wettlauf (leere Datei außerhalb) wegfällt; Host-Speicher insgesamt an das Limit koppeln.

---

## 6. Verbindliche Präzisierung der Garantien und Sicherheitsgrenzen

Jede hier aufgeführte Garantie ist durch konkrete Tests im Testbestand hinterlegt und grenzt den tatsächlichen Leistungsumfang gegen überzogene Interpretationen ab.

### 6.1 Beweisbegriff `proven` vs. Partielle Korrektheit
- **Was garantiert wird:** Der Status `proven` (`qal prove`) garantiert **partielle Korrektheit** für Verträge (`req`/`ens`) im Bereich linearer Ganzzahl-Arithmetik (Presburger-Arithmetik / QF_LIA über Z3) sowie induktiver Vertragsschlüsse. Wenn eine Funktion als `proven` deklariert ist und terminiert, verletzt keine Ausführung ihre Nachbedingungen, sofern die Vorbedingungen erfüllt sind.
- **Was ausdrücklich NICHT garantiert wird:**
  1. *Terminierung:* QAL beweist **keine** garantierte Terminierung rekursiver Funktionen oder Schleifen/Sequenzen (Halteproblem).
  2. *Minimale Gegenbeispiele:* Bei `refuted` liefert Z3 ein mathematisch valides Gegenbeispielmodell, dieses ist jedoch **nicht** garantiert das kleinstmögliche ("minimale") Gegenbeispiel.
  3. *Vollständigkeit bei komplexen Datentypen:* Nicht-lineare Arithmetik, Fließkomma-Semantik oder unvollständige Induktion über algebraische Datentypen führen zu `unknown` (Rückfall auf dynamische Laufzeitprüfung der Verträge).
- **Referenz-Tests:**
  - `qal/crates/qal-prove/src/tests.rs`: `proves_linear_pre_post`, `refutes_with_counterexample`
  - `qal/crates/qal-prove/src/tests.rs`: `confirm_fuel_is_part_of_cache_key_and_budget_differences_are_distinguished`

### 6.2 Merge-Semantik: Zusammenführung auf Definitionsebene
- **Was garantiert wird:** QAL führt Merges **nicht auf Zeilenebene**, sondern als **3-Wege-AST-Merge auf Granularität einzelner Definitionen** (`Def`) durch. Haben zwei Branches disjunkte Definitionen modifiziert, werden diese deterministisch zusammengeführt und der gesamte resultierende Graph wird erneut typ- und effektgeprüft (`relink`).
- **Was ausdrücklich NICHT garantiert wird:** Ein konfliktfreier Merge bedeutet nicht automatisch semantische Verträglichkeit. Wenn zwei Definitionen syntaktisch unbeeinflusst sind, sich aber in impliziten Invarianten widersprechen, greift die statische Typ-/Effekt- und Vertragskontrolle, keine semantische Allwissenheit. Gleichzeitige Modifikationen derselben Definition erzeugen einen harten `Conflict` und müssen manuell gelöst werden.
- **Referenz-Tests:**
  - `qal/crates/qal-cli/tests/team.rs`: `three_way_merge_disjoint_defs_succeeds`
  - `qal/crates/qal-cli/tests/team.rs`: `merge_conflict_on_same_def_rejected`

### 6.3 Kryptografische Signaturen vs. Codekorrektheit
- **Was garantiert wird:** Ed25519-Signaturen im Team-Workflow beweisen die **Authentizität des Autors**, den **Zeitstempel** und die **Integrität der Commit-Kette** (Merkle-Graph über Definitionshashes).
- **Was ausdrücklich NICHT garantiert wird:** Eine gültige Signatur garantiert **keine funktionale Korrektheit, Sicherheit oder Bugfreiheit** des Inhalts. Sie weist lediglich fälschungssicher nach, welcher Agent bzw. Entwickler den Stand signiert hat. Die inhaltliche Verifikation erfolgt orthogonal durch den QAL-Typechecker und -Prover.
- **Referenz-Tests:**
  - `qal/crates/qal-cli/tests/team.rs`: `signed_commit_verifies_author_signature`
  - `qal/crates/qal-cli/tests/team.rs`: `tampered_commit_signature_rejected`

### 6.4 Vertrauensannahmen und Grenzen des Rollenmodells
- **Was garantiert wird:** Das QAL-Rollenmodell (`Architect`, `Implementer`, `Reviewer`) erzwingt **kooperative Schranken auf Sprachebene**: Ein `Implementer` kann keine Verträge schwächen oder Reviewer-Freigaben fingieren; das Laufzeitsystem verhindert unautorisierte Effekte (`!io`, `!net`, `!db`) an der Hostgrenze.
- **Was ausdrücklich NICHT garantiert wird:** Das Rollenmodell ist **keine** betriebssystemseitige Sicherheitsgrenze gegen bösartige lokale Prozesse unter derselben Betriebssystem-UID. Ein Prozess mit Zugriff auf das Dateisystem kann `.qal/store.db` außerhalb von QAL verändern. QAL setzt voraus, dass Agenten in einer geeigneten Prozess-/Container-Sandbox laufen, wenn OS-Isolation gegen böswillige Ausführung gefordert ist.
- **Referenz-Tests:**
  - `qal/crates/qal-cli/tests/team.rs`: `role_checks_enforce_architect_approvals`
  - `qal/crates/qal-run/src/tests.rs`: `unauthorized_effects_rejected_at_boundary`
  - `qal/crates/qal-run/src/tests.rs`: `open_under_rejects_symlink_escaping_sandbox`

### 6.5 Nachweis-Matrix der Kern-Garantien

| Garantie | Formaler Anspruch | Nachweis / Test | Begrenzung / Annahme |
|---|---|---|---|
| **Partielle Korrektheit** | `ens` gilt bei Terminierung | `qal-prove/tests.rs` | Keine garantierte Terminierung; lineare Arithmetik |
| **Gegenbeispiele** | Widerlegung ist echtes Gegenbeispiel | `qal-prove/tests.rs` | Gegenbeispiel nicht garantiert minimal |
| **Effekt-Isolation** | Nur deklarierte Effekte importierbar | `qal-run/tests.rs` | Gilt im WASM-Gäste-Kontext |
| **AST-Merge** | Deterministischer 3-Wege-Merge | `qal-cli/tests/team.rs` | Definitionsebene, kein Zeilen-Diff |
| **Signaturprüfung** | Ed25519-Signatur der Commit-Lineage | `qal-cli/tests/team.rs` | Authentizität, keine semantische Korrektheit |
| **Sandbox-Sicherheit** | Kein Entkommen über Symlinks (`openat`) | `qal-run/tests.rs` | Basiert auf `O_NOFOLLOW` / Deskriptor-Verankerung |
| **Bootstrap-Fixpunkt** | Stage 1 == Stage 2 == Stage 3 | `full_fixpoint.rs` | Deterministische WASM-Codegenerierung |

