# QAL — Fahrplan bis "komplett fertig"

**Datum:** 2026-10-15
**Auftrag des Nutzers:** "mach es fertig komplett. Ohne Ausnahme!" — alle Phasen 1–5 der Spezifikation `2026-10-15-qal-design.md`. Der Nutzer delegiert alle Designentscheidungen ("Du entscheidest es selber"); Entscheidungen werden getroffen, begründet und im jeweiligen Plan/Ledger dokumentiert, ohne Zwischenfreigaben.

**Arbeitsweise je Teil:** Spezifikation (falls neu) → Implementierungsplan (Code vorab im Scratchpad verifiziert) → subagent-getriebene Umsetzung mit Review pro Aufgabe → Abschluss-Review → Push auf Branch `Mac` (PR Mirkulix/qland#2 bzw. Folge-PRs).

**Status-Legende:** ☐ offen · ◐ in Arbeit · ☑ fertig (alle Kriterien nachgewiesen)

| # | Teil | Fertig, wenn (nachweisbar) | Status |
|---|---|---|---|
| 1a | Kern-Pipeline | Text → Prüfung → WASM → Ausführung für Int/Bool, Contracts, JSON-Diagnosen; 50 Tests grün | ☑ |
| 1b | Daten, Speicher, Effekte | Float/Str/List/Option/Result/Records/Enums/`match`/Sequenzen/Built-ins (§3.5); Bump-Allocator mit `out_of_memory`; Effekte als Host-Importe nur für deklarierte Effekte; `qal run <datei> -- args` mit `--allow`; ein Programm ohne `fs` kann nachweislich keine Datei öffnen; vertagte 1a-Punkte erledigt | ☑ (135 Tests, 0 Warnungen; Commits 55ac2e05..42e3f547) |
| 1c | Graph-Speicher | Kanonische Form + SHA-256-Hash; Umbenennen ändert keinen Hash; gleiche Logik = gleicher Hash; SQLite-Store `.qal/store.db` mit Namen, Abhängigkeiten, Historie; `define` ersetzt mit automatischem Neuverknüpfen bei gleicher Signatur, sonst Fehler mit Liste der Abhängigen; Effekte als sortierte Menge | ☐ |
| 1d | Agenten-Schnittstelle | `qal mcp` (stdio) mit `qal_define, qal_view, qal_query, qal_test, qal_run, qal_build, qal_rename, qal_history`; kompakte Sprachreferenz in den Tool-Beschreibungen; keine Panik auf irgendeinem MCP-Pfad; in diesem Repo per `.mcp.json` registriert und von einem Claude-Code-Agenten real benutzt | ☐ |
| 1e | Benchmark | 20 Aufgaben mit versteckten Tests + Referenzlösungen in QAL; Durchlauf mit Claude-Code-Subagenten (QAL über MCP vs. Rust über Dateien + cargo), gemessen: Erfolg, Tokens, Tool-Aufrufe, Fehlversuche; `bench/report.md` | ☐ |
| 2 | Beweise & Ausdruckskraft | `req`/`ens` werden für lineare Ganzzahl-Arithmetik mit Z3 statisch bewiesen (Gegenbeispiel bei Widerlegung, Laufzeitprüfung als Rückfall); benutzerdefinierte Generics (Monomorphisierung); Closures; generierte Property-Tests | ☐ |
| 3 | Standardbibliothek & Services | Effekte `net` (HTTP-Server/Client über Host) und `db` (SQLite über Host); Speicherfreigabe für langlaufende Services (Arena pro Request); Standardbibliothek **in QAL** (JSON, HTTP-Routing, Strings) — geschrieben von Agenten über MCP; Beispiel-Backend (Todo-API) läuft und ist getestet | ☐ |
| 4 | Agenten-Team | Mehrere Agenten arbeiten gleichzeitig über MCP an einer Codebase: Sperren pro Definition, Rollen (Architekt, Implementierer, Verifizierer, Reviewer), Merge, signierte Änderungen (Ed25519); Demo: Team baut einen Service parallel | ☐ |
| 5 | Self-Hosting | Ein QAL-Compiler, geschrieben in QAL, übersetzt die QAL-Testsammlung zu identischem Verhalten; Bootstrap-Fixpunkt (Stage 2 == Stage 3, byte-identisches WASM); danach ist Rust nur noch für den Host (wasmtime-Einbettung) nötig | ☐ |

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
- Kein `expect`/`unwrap`/`unreachable!` auf MCP-erreichbaren Pfaden; unbekannter Einstiegspunkt muss vor `compile` abgelehnt werden.
- FIFO/Gerätedateien im erlaubten Ordner dürfen den Host nicht blockieren (nur reguläre Dateien, O_NONBLOCK + fstat).
- CLI: Optionen auch vor dem Dateinamen akzeptieren; `check`/`test` lehnen unbekannte Zusatzargumente ab.

**Phase 3:**
- `fs`-Schreiben über verifizierten Elternordner (`openat`), damit auch der Rest-Wettlauf (leere Datei außerhalb) wegfällt; Host-Speicher insgesamt an das Limit koppeln.
