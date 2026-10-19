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
| 4 | Agenten-Team | Mehrere Agenten arbeiten gleichzeitig über MCP an einer Codebase: Sperren pro Definition, Rollen (Architekt, Implementierer, Verifizierer, Reviewer), Merge, signierte Änderungen (Ed25519); Demo: Team baut einen Service parallel | ☑ (484 Tests, 0 Warnungen, Clippy-Restriktionsprüfung sauber; 4a 87d55959..c3f9a749, 4b b51347d4..265aa66d, 4c 448e6a39..b2a09517, 4d; Team-Demo `shop` REST API Service mit SQLite-Persistenz, Bestandsreservierung und Rollback: 2 Team-Läufe (6 Agenten: arch-1, impl-1..3, ver-1, rev-1) vs. 2 Solo-Läufe; 20/20 Szenarien (5 öffentlich, 15 versteckt) zu 100 % bestanden; 3 parallele Autoren im 5-Minuten-Fenster; automatischer Replayed-Merge bei nicht-fast-forward; Gate-Policy 4/4 eingehalten; `qal verify --trust team.json` kryptografisch verifiziert (38 Records); 0 Definitionen vom Controller; `shop.qalpkg` und `team.json` exportiert; Berichte `bench/team/report.md` und `bench/team/discussion.md`) |
| 5 | Self-Hosting | Ein QAL-Compiler, geschrieben in QAL, übersetzt die QAL-Testsammlung zu identischem Verhalten; Bootstrap-Fixpunkt (Stage 2 == Stage 3, byte-identisches WASM); danach ist Rust nur noch für den Host (wasmtime-Einbettung) nötig | ☑ (5b: Frontend/Lexer/Parser/Printer; 5c: WASM Codegen/QALT/Runtime; 5d: Typechecker/Unify/Mono/Exhaustiveness; 5e: qalc.qal Bundle, volle Proto-Ops, differentieller Abgleich gegen Rust-Referenz, Stage 2 == Stage 3 WASM-Fixpunkt verifiziert; Workspace-Tests und Clippy grün) |

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
