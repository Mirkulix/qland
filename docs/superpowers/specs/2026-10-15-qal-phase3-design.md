# QAL Phase 3 — Standardbibliothek & Services: Design-Spezifikation

**Datum:** 2026-10-15
**Status:** verbindlich für die Implementierungspläne 3a, 3b, 3c
**Bezug:** `2026-10-15-qal-design.md` (§2 Phase 3, §8 Speicher, §14), `2026-10-15-qal-roadmap.md` (Zeile 3, Übertrag „Phase 3“)
**Entscheidungen:** Der Nutzer delegiert alle Designentscheidungen. Jede Entscheidung unten ist getroffen und begründet; Pläne setzen sie um, ohne nachzufragen. Abweichungen nur mit Begründung im Plan-Ledger.

---

## 0. Ziel und Fertig-Kriterien

Ein QAL-Programm wird zu einem langlaufenden HTTP-Service mit Datenbank. Die Standardbibliothek (JSON, HTTP-Helfer, Routing, Strings, DB-Helfer) ist **in QAL** geschrieben — von Agenten über MCP — und wird inhaltsadressiert an Projekte verteilt. Ein Beispiel-Backend (Todo-API) läuft und ist auf HTTP-Ebene getestet.

**Phase 3 ist fertig, wenn (nachweisbar):**

1. `qal serve` startet einen HTTP/1.1-Service aus einem gespeicherten Handler `fn(HttpRequest) -> HttpResponse !e`; jede Anfrage läuft in einer frischen WASM-Instanz (Speicher nach jeder Anfrage freigegeben); 10 000 Anfragen in Folge lassen den Resident-Speicher des Prozesses nach der Aufwärmphase um höchstens 10 % wachsen.
2. Effekt `net` (ausgehender HTTP-Client) mit Host-Allowlist und SSRF-Schutz; Effekt `db` (SQLite über Host) mit ausschließlich parametrisierten Abfragen, deren SQL-Text ein **Literal** sein muss (Prüffehler sonst), eine Transaktion pro Ausführung/Anfrage.
3. Übertrag erledigt: `fs` öffnet über verifizierte Elternverzeichnis-Deskriptoren (`openat`-Kette), Host-Speicher ist an das Speicherlimit gekoppelt, Schreibbudget für `fs`.
4. Standardbibliothek `std.str`, `std.url`, `std.json`, `std.http`, `std.route`, `std.db` existiert als Paket `qal/std/std.qalpkg`, jede Definition laut Historie von einem Agenten (nicht `cli`) über MCP gespeichert, alle Abnahmetests grün.
5. Todo-API (`qal/examples/todo/`) von Agenten über MCP gebaut; QAL-Tests, `qal_request`-Szenarien und ein Rust-HTTP-Harness gegen `qal serve` grün (CRUD, Persistenz über Neustart, Nebenläufigkeit, Limits, Isolation nach Trap).
6. MCP-Tool `qal_request`; CLI `qal serve`, `qal export`, `qal import`; Sprachreferenz und Tool-Beschreibungen aktualisiert; 0 Warnungen, keine Panik auf MCP- oder Serve-Pfaden.

---

## 1. Annahmen über Phase 2

Zum Zeitpunkt dieser Spezifikation existiert `2026-10-15-qal-phase2-design.md` noch nicht. Phase 3 trifft deshalb folgende **explizite Annahmen** und bleibt so schmal wie möglich davon abhängig:

| Phase-2-Merkmal | Annahme | Nutzung in Phase 3 | Rückfall, falls nicht geliefert |
|---|---|---|---|
| Benutzerdefinierte Generics (Typen + Funktionen, Monomorphisierung, Syntax `type Route[A] = …`, `fn f[A](x: A) -> A`) | geliefert | **harte Abhängigkeit** nur für `std.route` (`Route[A]`, `RouteResult[A]`) | `tag: Str` statt `A`; 3b-Plan dokumentiert die Abweichung |
| Effekt-Polymorphie | **nicht** angenommen | — (Router ist bewusst rein, Dispatch per `match` im Nutzercode) | — |
| Closures | nicht benötigt | — | — |
| `Map[K,V]` | nicht benötigt | Header/Parameter/JSON-Objekte sind `List` von Records (ordnungserhaltend, Duplikate sichtbar) | — |
| `Bytes` | nicht benötigt | Bodies sind `Str` (UTF-8); Phase 3 bedient Text-/JSON-APIs | — |
| Property-Tests | geliefert | Abnahme der std (`print(parse(x))`-Rundlauf u. a.) | aufgezählte Tests mit ≥ 50 Fällen pro Eigenschaft |
| Z3-Beweise | geliefert | `req`/`ens` der std werden bewiesen, wo linear | Laufzeitprüfung (Phase-1-Verhalten) |
| Vor-1e-Built-ins (`char_code`, `from_char_code`, `to_upper`, `to_lower`, `replace`, `index_of`, `Str`-Ordnung, `sort_str`, `read_all`) | vor Phase 3 vorhanden (Roadmap „Pflicht vor 1e“) | `std.str`, `std.url`, `std.json` | 3b wartet; kein Ersatz in QAL |

**Diagnose-Codes:** Um Kollisionen mit dem parallel geplanten Phase-2-Bereich zu vermeiden, belegt Phase 3 ausschließlich **E0240–E0249** (Prüfung), **E0520–E0539** (Laufzeit) und **E0800–E0819** (Service/HTTP). Phase 2 bleibt E0211–E0239 und E0512–E0519.

**Rekursive Typen** (z. B. `Json` mit `List[Json]`) werden bereits in Phase 1 unterstützt (`infer.rs`, `seen`-Schutz); Phase 3 verlässt sich darauf.

---

## 2. Service-Modell

### 2.1 Handler und Einstiegspunkte

Neue **eingebaute Typen** im Prelude (wie `Option`/`Result`; Namen reserviert, Layout dem Host bekannt):

```
type HttpHeader   = { name: Str, value: Str }
type HttpRequest  = { method: Str, path: Str, query: Str, headers: List[HttpHeader], body: Str }
type HttpResponse = { status: Int, headers: List[HttpHeader], body: Str }
```

- `method`: Großbuchstaben, wie gesendet (`GET`, `POST`, …; Token-Syntax nach RFC 9110, sonst 400 vom Host).
- `path`: roher Pfad ohne Query (kein Prozent-Dekodieren durch den Host; `std.url.decode` dekodiert). Muss mit `/` beginnen (sonst 400; absolute-form und `*` werden abgelehnt).
- `query`: roher Teil nach `?` ohne `?` (leer, wenn keiner).
- `headers`: Reihenfolge wie empfangen, **Namen klein geschrieben**, Werte ohne umgebende Leerzeichen. Hop-by-hop-Header (`connection`, `keep-alive`, `transfer-encoding`, `upgrade`, `te`, `trailer`) werden entfernt.
- `body`: UTF-8. Ungültiges UTF-8 → 400 vom Host, Handler wird nicht aufgerufen.

**Handler:** eine gespeicherte Funktion mit Signatur exakt `fn(HttpRequest) -> HttpResponse` mit beliebigen Effekten aus `{io, time, rand, db, net, fs}`. Standardname `handle`.

**Init (optional):** `fn() -> Result[Unit, Str]` mit Effekten; läuft einmal beim Start in einer eigenen Instanz und Transaktion (Migrationen, z. B. `CREATE TABLE IF NOT EXISTS`). `Err(m)` oder Trap → Start abgebrochen, Exit-Code 1, Fehler-JSON auf stderr. Option `--init <name>` (CLI) bzw. `init` (MCP).

Falsche Signaturen werden **vor** dem Start abgelehnt: Laufzeit-Diagnose `bad_entry` (E0520) mit erwarteter und gefundener Signatur.

### 2.2 Eine Instanz pro Anfrage („Arena pro Request“)

**Entscheidung:** Jede Anfrage bekommt einen neuen wasmtime-`Store` und eine neue Instanz aus einem einmal vorbereiteten `InstancePre` (Modul kompiliert und gelinkt beim Start). Der lineare Speicher der Instanz ist die Arena der Anfrage: innerhalb der Anfrage bleibt der Bump-Allocator unverändert (kein Freigeben), nach der Antwort wird die gesamte Instanz verworfen.

**Begründung:**
- Isolation durch Konstruktion: kein Zustand (Heap, Heap-Zeiger, nicht initialisierte Bytes) kann eine Anfrage überleben — unabhängig von der Korrektheit des Codegens. Ein Heap-Reset mit Wiederverwendung der Instanz müsste dagegen beweisen, dass kein Laufzeit-Helfer uninitialisierten Speicher liest (z. B. `StrNew`/`ListNew` liefern uninitialisierte Blöcke).
- Der Phase-1-`Runner` instanziiert bereits pro Aufruf; das Modell ist identisch für `qal_run`, `qal_call`, `qal_request` und `qal serve`.
- Kosten: mit dem **Pooling-Allocator** von wasmtime (Feature `pooling-allocator` im Workspace aktivieren; Copy-on-Write-Initialisierung, Speicher-Reset per `madvise`) im Bereich von Mikrosekunden.

**Messgate in 3a:** Mittlerer Overhead Instanziierung+Verwerfen für das Todo-Modul ≤ 200 µs auf der Entwicklungsmaschine (Release-Build). Wird das Gate verfehlt, ist der Rückfall (und nur dann): Instanz pro Worker wiederverwenden, nach jeder Anfrage Heap-Zeiger auf den Startwert zurücksetzen **und** den benutzten Bereich bis zur Hochwassermarke mit Nullen füllen (`data_mut()[base..hw].fill(0)`). Der Rückfall wird im 3a-Ledger mit Messwerten begründet.

Pooling-Konfiguration: `total_memories = total_core_instances = workers + 2` (Init und Reserve), `max_memory_size = memory_limit` der Anfrage, eine Tabelle, keine Stacks-Pooling-Besonderheiten. Fehlschlag der Pooling-Konfiguration (z. B. wenig virtueller Adressraum) → Fallback auf den On-Demand-Allocator mit Warnung im Log (kein Startabbruch).

### 2.3 Nebenläufigkeit

**Entscheidung:** Synchroner Server ohne async-Runtime.
- Ein Accept-Thread (`std::net::TcpListener`), eine **begrenzte** Warteschlange von Verbindungen (Kapazität `--queue`, Standard 64) und `--workers` Worker-Threads (Standard 4, Bereich 1–64). Ist die Warteschlange voll, antwortet der Accept-Thread sofort `503` mit `Retry-After: 1` und schließt.
- Jeder Worker bearbeitet eine Verbindung vollständig (Keep-Alive bis 100 Anfragen oder 5 s Leerlauf), jede Anfrage in eigener Instanz (§2.2), seriell innerhalb der Verbindung (kein Pipelining-Parallelismus; gepipelinte Anfragen werden der Reihe nach beantwortet).
- Worker-Threads werden mit `WASM_STACK_BYTES + 64 MiB` Stack gestartet (wie `Runner::invoke`).
- Gemeinsam: `Engine`, `InstancePre`, Typ-Metadaten (alles `Send + Sync`). Pro Worker: eine SQLite-Verbindung (§3.2), ein Puffer.

**Begründung:** Guest-Ausführung ist CPU-gebunden und synchron (fuel, epoch); tokio/hyper bringen eine große Abhängigkeitskette ohne Nutzen für das Modell „eine Anfrage = ein synchroner WASM-Aufruf“. Parser: Crate **`httparse`** (abhängigkeitsfrei, verbreitet) für Request-Zeile und Header; Body, Keep-Alive und Antwortserialisierung sind eigener, kleiner Code in `qal-serve`.

**HTTP-Umfang:** HTTP/1.1 und 1.0; Body nur mit `Content-Length`. `Transfer-Encoding` in einer Anfrage → `411`/`501` (`chunked` → 411, sonst 501). `Expect: 100-continue` → Host sendet `100 Continue`, wenn `Content-Length` im Limit, sonst 413. Kein TLS (Betrieb hinter Reverse-Proxy; dokumentiert). Kein HTTP/2. Antworten immer mit `Content-Length`; `HEAD` wird als `GET` an den Handler gegeben und der Body verworfen.

### 2.4 Zeitlimits: Epoch-Ticker statt Watchdog

Der Phase-1-Watchdog erhöht die **engine-weite** Epoche — bei mehreren gleichzeitigen Stores würde ein abgelaufener Timer alle laufenden Anfragen abbrechen. **Entscheidung:** Im Serve-Modus erhöht ein einziger Ticker-Thread die Epoche alle 10 ms; jeder Store setzt `set_epoch_deadline(ceil(timeout / 10 ms))`. Der Phase-1-`Runner` (eine Ausführung pro Engine) behält den Watchdog. Beide Wege melden `timeout` (E0507).

### 2.5 Limits pro Anfrage

| Limit | Standard | Option `qal serve` | Folge bei Überschreitung |
|---|---|---|---|
| Wandzeit Handler | 5 s | `--request-timeout <ms>` | Trap `timeout` → 503 |
| Fuel | 500 000 000 | `--fuel <n>` | Trap `timeout` → 503 |
| Guest-Speicher | 64 MiB | `--memory <MiB>` (max. 256) | Trap `out_of_memory` → 500 |
| Host-Speicher (§5.4) | = Guest-Limit | (gekoppelt) | Trap `out_of_memory` mit `host: true` → 500 |
| Request-Zeile + Header | 16 KiB, ≤ 64 Header | `--max-header-bytes` | 431 (Host, ohne Handler) |
| Request-Body | 1 MiB | `--max-body <KiB>` | 413 (Host, ohne Handler) |
| Antwort-Body | 8 MiB | `--max-response <KiB>` | `bad_response` → 500 |
| Header lesen | 5 s ab Verbindungs-/Anfragebeginn | `--header-timeout <ms>` | Verbindung geschlossen (408, wenn noch nichts gesendet) |
| Body lesen | 10 s | `--body-timeout <ms>` | 408 |
| Keep-Alive | 100 Anfragen / 5 s Leerlauf | — | Verbindung geschlossen |
| `print`-Ausgabe im Log | 4 KiB pro Anfrage | — | abgeschnitten, `out_bytes` zählt weiter |
| ausgehende `net`-Aufrufe | 16 pro Anfrage | `--net-calls` | Trap `budget_exceeded` |
| `fs`-Schreiben | 64 MiB, 256 `write_file` pro Ausführung | `--fs-write-budget <MiB>` | Trap `budget_exceeded` |
| DB-Ergebniszeilen | 10 000 pro Abfrage | `--db-max-rows` | `Err(DbTooLarge)` |
| DB-Dateigröße | 1 GiB | `--db-max-mib` | `Err(DbFull)` |

Dieselben Standardwerte gelten für `qal_request`; `qal_run`/`qal_call` behalten ihre Phase-1-Werte (10 s, 2·10⁹ Fuel, 256 MiB), bekommen aber Schreib-, Netz- und Host-Speicherbudget.

### 2.6 Fehlerverhalten (graceful)

| Situation | Status | Body | Log |
|---|---|---|---|
| Handler liefert normal | `status` des Handlers | Handler-Body | `status`, `ms`, `fuel`, `mem` |
| Trap `timeout` | 503 | `{"error":{"kind":"timeout","id":"<req-id>"}}` | volles RunError-JSON |
| andere Traps (`contract_violation`, `int_overflow`, `out_of_memory`, …) | 500 | `{"error":{"kind":"internal_error","id":"<req-id>"}}` | volles RunError-JSON |
| ungültige Antwort (`bad_response`: Status ∉ 100..599, CR/LF/NUL in Headernamen/-werten, ungültiger Headername, Body zu groß, Handler setzt `content-length`/`transfer-encoding`/`connection`) | 500 | wie oben | RunError `bad_response` (E0521) |
| Host lehnt Anfrage ab (400/408/411/413/431/501) | s. §2.3/§2.5 | `{"error":{"kind":"<bad_request|…>","message":"…"}}` | `request_rejected` (E0800) |
| Warteschlange voll | 503 + `Retry-After: 1` | `{"error":{"kind":"overloaded"}}` | `overloaded` (E0801) |

- `--debug-errors` (CLI) gibt das volle RunError-JSON im Body zurück; `qal_request` gibt es **immer** im Feld `error` zurück (Agenten brauchen die Diagnose).
- Der Host setzt `content-length` und, falls der Handler keinen `content-type` setzt, `content-type: text/plain; charset=utf-8`. Kein Trap beendet den Prozess; ein Panic im Worker wird gefangen (`catch_unwind`), mit 500 beantwortet, als `internal` (E0819) geloggt, und der Worker läuft weiter.
- **Request-ID:** monoton steigende Zahl pro Prozess, als `x-request-id` in jeder Antwort.
- **Log:** JSON-Zeilen auf stderr (`--log <datei>` optional): Start `{"event":"listening","addr":…,"entry":…,"hash":…,"workers":…}`, pro Anfrage `{"event":"request","id","method","path","status","ms","fuel","mem_bytes","db":"commit|rollback|none","out"?,"error"?}`, Ende `{"event":"stopped","served":n}`.
- **Herunterfahren:** SIGINT/SIGTERM (Crate `signal-hook`) → keine neuen Verbindungen, laufende Anfragen bis 10 s beenden, dann Exit 0.
- **Kein Hot-Reload:** `qal serve` kompiliert beim Start einen Schnappschuss (Hash des Handlers im Log); neue Definitionen wirken nach Neustart. Begründung: reproduzierbares Verhalten pro Prozess; Hot-Reload ist Phase-4-Thema (Team-Deployments).

### 2.7 Effekte im Serve-Modus

- `io`: `print` schreibt in das Feld `out` der Log-Zeile (und in `qal_request` in `stdout`); `read_line`/`read_all` liefern `None`/`""` (kein stdin).
- `time`, `rand`: wie Phase 1.
- `fs`: wie Phase 1 mit openat (§5.3) und Schreibbudget.
- `db`, `net`: §3.

---

## 3. Effekte `net` und `db`

### 3.1 `net` — ausgehender HTTP-Client

**Entscheidung: Es gibt keinen Guest-Effekt zum Lauschen.** Der Guest berührt nie Sockets; Lauschen ist eine Betreiberentscheidung (`qal serve --listen`). Der Effekt `net` bedeutet ausschließlich **ausgehende HTTP-Anfragen**. Begründung: Ein Handler braucht keine Fähigkeit, um aufgerufen zu werden; die Trennung „Inbound = CLI-Option, Outbound = Effekt“ hält das Effektsystem bei „Effekt = was der Code aktiv tut“.

Prelude:

```
type HttpCall = { method: Str, url: Str, headers: List[HttpHeader], body: Str }
type NetError = NetDenied(Str) | NetInvalid(Str) | NetConnect(Str) | NetTimeout | NetTooLarge
fn http(call: HttpCall) -> Result[HttpResponse, NetError] !net      # eingebaut
```

- Schemata `http` und `https` (rustls, Web-PKI-Wurzeln; kein Abschalten der Zertifikatsprüfung). Userinfo in der URL → `NetInvalid`.
- **Keine Weiterleitungen:** 3xx wird als Antwort zurückgegeben (verhindert Redirect-SSRF).
- Host setzt `host`, `content-length`, `connection: close`, `user-agent: qal/<version>`; Guest-Header mit diesen Namen oder `transfer-encoding` → `NetInvalid`.
- Zeitlimits: Verbinden 5 s, gesamt 10 s (und nie länger als die Restzeit der Anfrage). Antwort-Body ≤ 4 MiB (sonst `NetTooLarge`), muss UTF-8 sein (sonst `NetInvalid("response body is not UTF-8")`). Antwort-Header-Namen klein geschrieben.
- Proxy-Umgebungsvariablen werden ignoriert.
- Implementierung: Crate `ureq` (2.x, rustls) mit eigenem `Resolver` (§5.1), Redirects 0.

**Freigabe:** `net:<host>` oder `net:<host>:<port>`; wiederholbar. `<host>` ist ein DNS-Name (klein, IDNA-ASCII), `*.example.com` (genau Subdomains, nicht `example.com` selbst) oder ein IP-Literal (`net:127.0.0.1:9000`, `net:[::1]:9000`). Ohne Port: nur 80 (http) und 443 (https). Ein bloßes `net` ist ein Fehler (wie `fs`): `` `net` needs a host: `net:<host>[:port]` ``. Nicht freigegebene Ziele → `Err(NetDenied(…))` (kein Trap: der Service kann degradieren; die Meldung nennt die benötigte Freigabe).

### 3.2 `db` — SQLite über Host

Prelude:

```
type SqlValue = SqlNull | SqlInt(Int) | SqlFloat(Float) | SqlText(Str)
type DbRows   = { columns: List[Str], rows: List[List[SqlValue]] }
type DbError  = DbConstraint(Str) | DbSql(Str) | DbBusy | DbFull | DbTooLarge
fn db_exec(sql: Str, params: List[SqlValue]) -> Result[Int, DbError] !db      # Anzahl geänderter Zeilen
fn db_query(sql: Str, params: List[SqlValue]) -> Result[DbRows, DbError] !db
```

**SQL-Injection ist durch das API-Design unmöglich:**
1. **Das Argument `sql` muss an jeder Aufrufstelle ein Str-Literal sein.** Prüfregel im Checker (eingebaute Funktionen sind keine Werte, also kann kein Wrapper ein dynamisches `sql` durchreichen): sonst Diagnose `sql_not_literal` (E0241) mit Hinweis „Werte gehören in `params` als `?`“. Damit ist SQL-Text Teil des Programmgraphen (und des Hashes) und nie aus Daten zusammengesetzt.
2. Werte nur über `?`-Platzhalter (`params`); Anzahl ungleich `sqlite3_bind_parameter_count` → `Err(DbSql("expected N parameters, got M"))`. Benannte Platzhalter (`:x`, `@x`, `$x`, `?NNN`) werden bei der Prüfung des Literals abgelehnt (E0242 `sql_invalid`), damit Zuordnung positional und eindeutig ist.
3. Prüfung des Literals beim `define` (ohne Schema): mit einem leeren In-Memory-SQLite wird nur `sqlite3_complete` + Tokenisierung genutzt — genau **eine** Anweisung (ein abschließendes `;` erlaubt), keine benannten Platzhalter, keine verbotenen Schlüsselwörter am Anweisungsanfang (`ATTACH`, `DETACH`, `PRAGMA`, `VACUUM`, `BEGIN`, `COMMIT`, `END`, `ROLLBACK`, `SAVEPOINT`, `RELEASE`). Verstöße → E0242. Spaltenexistenz wird nicht geprüft (Schema entsteht erst zur Laufzeit).
4. Zur Laufzeit zusätzlich ein **SQLite-Authorizer** (rusqlite-Feature `hooks`): verweigert `SQLITE_ATTACH`, `SQLITE_DETACH`, `SQLITE_PRAGMA`, `SQLITE_TRANSACTION`, `SQLITE_SAVEPOINT`, Funktionsaufrufe `load_extension`, `readfile`, `writefile`, `fts3_tokenizer`, sowie Schreiben auf `sqlite_master`/`sqlite_schema`; `PRAGMA` setzt nur der Host. Extension-Loading ist abgeschaltet. Verbleibender Text hinter der ersten Anweisung → `DbSql`.

**Werte-Abbildung:** `INTEGER`→`SqlInt`, `REAL`→`SqlFloat`, `TEXT`→`SqlText`, `NULL`→`SqlNull`; `BLOB` → `Err(DbSql("BLOB columns are not supported"))` (Phase 3 ohne `Bytes`). Nicht-UTF-8-Text ebenso `DbSql`. `columns` sind die Spaltennamen der Anweisung.

**Fehler-Abbildung:** `SQLITE_CONSTRAINT*` → `DbConstraint(msg)`, `SQLITE_BUSY/LOCKED` nach `busy_timeout` 5 s → `DbBusy`, `SQLITE_FULL` (inkl. `max_page_count`) → `DbFull`, mehr als `--db-max-rows` Zeilen → `DbTooLarge`, alles andere → `DbSql(msg)`. Fehler sind Daten (Result), keine Traps.

**Transaktionen — eine pro Ausführung, kein Transaktions-API:**
- Vor dem Aufruf des Einstiegspunkts (Handler, `init`, `main`, `qal_call`-Funktion) — und nur wenn dessen Effekte `db` enthalten — `BEGIN IMMEDIATE`.
- **Commit**, wenn: Handler normal zurückkehrt **und** `status` ∈ 100..399; `init` `Ok(())` liefert; `main` mit Exit-Code 0 endet; `qal_call` ohne Trap endet. Sonst (Status ≥ 400, Trap, `Err`, Exit ≠ 0, `bad_response`) **Rollback**.
- Begründung: Eine Anfrage ist die natürliche Arbeitseinheit; ohne Closures/Effekt-Polymorphie wäre ein `transaction(f)`-API nicht typisierbar; Teil-Schreibvorgänge bei Fehlern (Validierung nach Insert, Trap mitten im Handler) sind ausgeschlossen. 4xx rollt zurück, weil ein Handler, der nach einem Schreibvorgang einen Client-Fehler meldet, die Anfrage abgelehnt hat.
- `BEGIN IMMEDIATE` serialisiert schreibende Ausführungen (WAL, `busy_timeout` 5 s); Lesende anderer Verbindungen sehen konsistente Schnappschüsse. Bekannte Grenze: auch rein lesende `db`-Handler serialisieren; dokumentiert, Optimierung (Lese-Effekt) ist nicht Teil von Phase 3.

**Freigabe und Einschränkung (Confinement):**
- `db` → Datei `<projekt>/.qal/db/main.sqlite`; `db:<name>` (`[a-z][a-z0-9_]{0,31}`) → `<projekt>/.qal/db/<name>.sqlite`; `db:mem` → frische In-Memory-Datenbank pro Tool-Aufruf (Standard für Agenten-Tests, §7). Höchstens **eine** `db`-Freigabe pro Ausführung (sonst Fehler bei der Freigabe-Prüfung).
- Der Guest wählt nie einen Pfad: der Host öffnet die Datei (Ordner wird angelegt, `O_NOFOLLOW` auf die Datei; Symlink → Startfehler). Die Codebase `.qal/store.db` ist nie erreichbar (anderer Pfad, kein ATTACH). `.qal` bleibt für `fs` versteckt.
- Host-Pragmas beim Öffnen: `journal_mode=WAL`, `foreign_keys=ON`, `busy_timeout=5000`, `max_page_count` gemäß `--db-max-mib`, `trusted_schema=OFF`, `cell_size_check=ON`.

### 3.3 Freigabe über CLI und MCP

Erweiterte Freigabesyntax (`Permissions::parse`/`parse_in`):

```
io  time  rand  fs:<dir>  net:<host>[:port]  db  db:<name>  db:mem
```

- CLI: `--allow io,db,net:api.example.com` (bei `run`, `call`, `test`, `serve`). MCP: `allow: ["db:mem", "net:api.example.com"]` (bei `qal_run`, `qal_call`, `qal_request`).
- `qal_test` bleibt rein (Tests sind reine Ausdrücke). Effektvolles Verhalten testen Agenten mit `qal_request`/`qal_call` und `db:mem`.
- Fehlende Freigabe → `permission_denied` (E0508) vor dem Start, Hinweis im MCP-Format (`pass "allow": ["db:mem"]`), wie in Phase 1.
- Das Modul importiert nur Host-Funktionen der deklarierten Effekte (unverändertes Modell); neue Gruppen `net.http`, `db.exec`, `db.query`.

---

## 4. Standardbibliothek in QAL

### 4.1 Namensraum `std`

**Entscheidung: qualifizierte Namen mit reserviertem Präfix `std`.** `std` ist ein Schlüsselwort; `std.<modul>.<name>` ist ein einziger Namens-Token-Pfad (Lexer/Parser), kein Feldzugriff. Modulsegmente klein; letztes Segment klein (fn/test) oder groß (Typ/Variante). Varianten von std-Typen sind ebenfalls qualifiziert (`std.json.Str("a")`, Muster `std.json.Int(n)`), weil Variantennamen global sind und sonst mit Nutzertypen kollidieren.

- Begründung: Definitionen sind eigenständig und hashbar; ein `use`/Import-Konstrukt würde Namensauflösung kontextabhängig machen. Qualifizierte Namen sind eindeutig, im Store als normale Namen (`names.name = 'std.json.parse'`) speicherbar und `qal_query name: "std.json."` findet alles per Präfix.
- Im Nutzerprojekt sind `std.`-Namen **schreibgeschützt**: `define`/`rename` eines `std.`-Namens → E0240 `reserved_name`. Ebenso Prelude-Namen (`HttpRequest`, `SqlValue`, `DbError`, `NetError`, … und deren Varianten) als Nutzer-Typ-/Variantennamen → E0240.
- Ausnahme: ein Store mit `meta.role = 'std'` (das std-Projekt `qal/std/`, §4.3) darf `std.`-Namen definieren — dort ausschließlich solche (E0240 für unqualifizierte Namen, damit nichts Projektlokales in das Paket gerät).

### 4.2 Module (Signaturen verbindlich, Körper schreiben Agenten)

Notation: `A` ist ein Typparameter (Phase 2). Alle Funktionen sind **rein**, außer wo vermerkt.

**`std.str`** (über die Built-ins hinaus)
`ends_with(s, suf) -> Bool`, `trim_start(s)`, `trim_end(s)`, `repeat(s, n) -> Str` (`req n >= 0`), `pad_left(s, width, fill) -> Str`, `pad_right(…)`, `split_once(s, sep) -> Option[std.str.Pair]` mit `type std.str.Pair = { left: Str, right: Str }`, `lines(s) -> List[Str]` (`\n` und `\r\n`), `is_digit(c)`, `is_alpha(c)`, `is_space(c)` (je ein Zeichen als Str), `eq_ignore_case(a, b) -> Bool`.

**`std.url`**
`decode(s) -> Option[Str]` (Prozent-Dekodierung, `+` bleibt `+`; ungültige Sequenz oder ungültiges UTF-8 → `None`), `decode_form(s)` (wie `decode`, `+` → Leerzeichen), `encode(s) -> Str` (alles außer `A-Z a-z 0-9 - . _ ~` prozentkodiert, Großbuchstaben-Hex), `parse_query(q) -> List[std.url.Param]` mit `type std.url.Param = { name: Str, value: Str }` (Form-Dekodierung; undekodierbare Paare werden ausgelassen), `param(ps, name) -> Option[Str]` (erstes Vorkommen).

**`std.json`**

```
type std.json.Json = std.json.Null | std.json.Bool(Bool) | std.json.Int(Int) | std.json.Float(Float)
                   | std.json.Str(Str) | std.json.Arr(List[std.json.Json]) | std.json.Obj(List[std.json.Field])
type std.json.Field = { key: Str, value: std.json.Json }
type std.json.Error = { pos: Int, message: Str }
```
- `parse(s) -> Result[Json, Error]`: RFC 8259 strikt (kein Kommentar, kein abschließendes Komma, Escapes inkl. Surrogatpaare; einzelne Surrogate → Fehler). Zahlen ohne Bruch/Exponent, die in i64 passen → `Int`, sonst `Float`. **Doppelte Schlüssel → Fehler.** Verschachtelungstiefe ≤ 128 (Fehler sonst). `pos` = Zeichenindex. Führende/abschließende Leerzeichen erlaubt.
- `print(j) -> Str`: kompakt, Schlüssel in gegebener Reihenfolge, Escape von `"` `\` und Steuerzeichen < 0x20 (`\n \r \t \b \f`, sonst `\u00XX`), nicht-ASCII roh. `Float`: endliche Werte wie `from_float`, mit `.0` ergänzt falls ganzzahlig; NaN/±inf → `null`.
- Zugriff: `get(j, key) -> Option[Json]` (Objekt), `at(j, i) -> Option[Json]` (Array), `as_str`, `as_int`, `as_float` (`Int` wird umgewandelt), `as_bool`, `as_arr -> Option[List[Json]]`, `as_obj -> Option[List[Field]]`.
- Bau: `obj(fs: List[Field]) -> Json`, `field(k, v) -> Field`, `str(s)`, `int(n)`, `bool(b)`, `arr(xs)`.
- Eigenschaften (Abnahme): `parse(print(j)) == Ok(j)` für jedes `j` ohne NaN/inf und ohne doppelte Schlüssel; `print` liefert nie einen String, den `parse` ablehnt.

**`std.http`** (auf den Prelude-Typen)
`header(req, name) -> Option[Str]` (Name ohne Groß/Klein), `query(req) -> List[std.url.Param]`, `response(status, ctype, body) -> HttpResponse`, `text(status, body)`, `json(status, j: std.json.Json)` (`content-type: application/json`), `error(status, code, message) -> HttpResponse` (Body `{"error":{"code":…,"message":…}}`), `not_found()`, `bad_request(message)`, `method_not_allowed(allowed: List[Str])` (405 mit `allow`-Header), `with_header(resp, name, value) -> HttpResponse`, `json_body(req) -> Result[std.json.Json, HttpResponse]` (prüft `content-type` beginnt mit `application/json` → sonst 415; Parse-Fehler → 400 mit Position).

**`std.route`**

```
type std.route.Route[A]  = { method: Str, pattern: Str, tag: A }
type std.route.Result[A] = std.route.Matched(A, List[std.url.Param]) | std.route.NotFound | std.route.MethodNotAllowed(List[Str])
fn std.route.route[A](routes: List[std.route.Route[A]], method: Str, path: Str) -> std.route.Result[A]
fn std.route.param(ps: List[std.url.Param], name: Str) -> Option[Str]
```
- Muster: `/todos`, `/todos/:id`, abschließend `/*rest` (Rest inkl. `/`). Segmente werden mit `std.url.decode` dekodiert; undekodierbar → kein Treffer. Ein abschließendes `/` im Pfad ist ein eigenes leeres Segment (keine Normalisierung). Erste passende Route gewinnt. Passt der Pfad, aber keine Methode → `MethodNotAllowed` mit den Methoden aller passenden Routen (sortiert, ohne Duplikate; `HEAD` erlaubt, wo `GET` erlaubt ist).
- **Begründung des Designs:** Der Router ist rein und liefert ein Nutzer-Tag `A` (typisch ein Nutzer-Enum); der Handler dispatcht per `match`. So braucht die std weder Closures noch Effekt-Polymorphie, und `match` erzwingt Vollständigkeit über alle Routen.

**`std.db`** (auf den Prelude-Typen)
`col(rows: DbRows, name) -> Option[Int]` (Spaltenindex), `int(row: List[SqlValue], i) -> Option[Int]`, `text(row, i) -> Option[Str]`, `float(row, i) -> Option[Float]` (`SqlInt` wird umgewandelt), `is_null(row, i) -> Bool`, `bool(row, i) -> Option[Bool]` (0/1), `to_json(rows: DbRows) -> std.json.Json` (Array von Objekten, Schlüssel = Spaltennamen, `SqlNull`→`Null`), `first(rows) -> Option[List[SqlValue]]`.

Nicht in der std: Templates, Auth, Sessions, Datum/Zeit-Formatierung (spätere Phasen bei Bedarf).

### 4.3 Speicherung und Verteilung

- **std-Projekt:** `qal/std/` ist ein normales QAL-Projekt mit `meta.role = 'std'` (neue Tabelle `meta(key TEXT PRIMARY KEY, value TEXT)`, Schema-Version 3 mit Migration 2→3, die nur `meta` anlegt). Sein `.qal/store.db` wird **nicht** eingecheckt (Binärdatei).
- **Paketdatei (eingecheckt):** `qal/std/std.qalpkg`, erzeugt mit `qal export std. --out std/std.qalpkg`. Format (Text, deterministisch):
  ```
  # qal-pkg 1
  # package std <sha256 über alle folgenden Zeilen>
  # def <name> <hash> <agent>
  …                                   (eine Zeile pro Name, nach Name sortiert)
  <kanonische Textform aller Definitionen, in Abhängigkeitsreihenfolge, durch Leerzeilen getrennt>
  ```
  `agent` ist der Agent des letzten `added|replaced`-Historieneintrags — die eingecheckte Herkunftsangabe.
- **Import:** `qal import <datei>` definiert den Textteil in **einem** atomaren `define` (Agent `qal-std@<pakethash-präfix>`), vergleicht danach jeden Namen mit dem aufgezeichneten Hash (Rundlauf-Garantie: identischer Text ⇒ identischer Hash) und rollt bei Abweichung zurück (E0243 `package_mismatch`). Paket-Prüfsumme falsch → E0243 ohne Schreiben.
- **Einbettung:** `qal-cli` bettet `std.qalpkg` per `include_str!` ein. `qal init` importiert es. Beim Öffnen eines Projektstores zum **Schreiben** (define/rename/import) wird verglichen: `meta.std_package` ≠ eingebetteter Pakethash → das eingebettete Paket wird zuerst importiert (eigene Transaktion, Historie mit Agent `qal-std@…`). Lesende Operationen ändern nie etwas; `qal_query`/`qal_view` zeigen dann die zuletzt importierte std. In einem Store mit `role = 'std'` findet kein Auto-Import statt.
- **Versionierungsregel:** std ist in Signaturen **nur additiv**. Ein Import, der die Signatur eines bestehenden `std.`-Namens ändern würde, wird abgelehnt (E0244 `std_signature_change`); Brüche brauchen einen neuen Namen (`std.json.parse2`). Gleiche Signatur, neuer Körper → normales Ersetzen mit automatischem Neuverknüpfen der Nutzer-Definitionen.
- Bootstrapping: 3a legt `std/std.qalpkg` als leeres, gültiges Paket an (nur Kopfzeilen), damit der Build nie von 3b abhängt.

### 4.4 „Geschrieben von Agenten über MCP“ — Verfahren

Ausgeführt vom **Controller** (Hauptsitzung, die Plan 3b abarbeitet). Der Controller schreibt **keinen** QAL-Körper der std selbst.

1. **Vorbereitung (Controller):** pro Modul eine Aufgabenkarte im Plan: Signaturen und Contracts aus §4.2, Verhaltensbeschreibung, **öffentliche Tests** (Textform) und **versteckte Abnahmetests** (Textform; nicht an den Agenten übergeben). Für `std.json` zusätzlich die Property-Tests (Phase 2) bzw. Rückfall-Fälle.
2. **Dispatch:** ein Subagent pro Modul, in Abhängigkeitsreihenfolge `str → url → json → http → route → db` (`http` braucht `json`+`url`; `db` braucht `json`). Der Subagent erhält: die Aufgabenkarte ohne versteckte Tests und **ausschließlich** die MCP-Tools des std-Projekts (`qal mcp` mit Root `qal/std`, über `sh qal/mcp.sh std` bzw. `.mcp.json`-Eintrag `qal-std`); kein Dateischreiben, kein Bash. `agent`-Argument: `std-<modul>-<lauf>`.
3. **Abnahme (Controller):** versteckte Tests per `qal_define` in einen **Wegwerf-Klon** des std-Stores (Kopie von `store.db` in den Scratchpad) definieren und `qal_test` laufen lassen — die Tests landen nie im Paket. Bei Fehlschlag: Fehlschlag-Diagnosen (nicht die Tests) an denselben Agenten zurück (`SendMessage`), max. 3 Runden; danach neuer Agent mit derselben Karte und Hinweis auf die Fehlerklasse.
4. **Review:** ein Reviewer-Subagent liest per `qal_view` das Modul und prüft gegen §4.2 (Signaturen exakt, Grenzfälle, Komplexität: kein quadratisches `++`/`push` über Eingaben > 8 000). Befunde gehen an einen Implementierer-Agenten (wieder nur über MCP).
5. **Nachweis:** `qal history` jedes `std.`-Namens enthält nur Agenten `std-*`; kein Eintrag mit Agent `cli`. Ein Rust-Test in 3b prüft das an der eingecheckten Paketdatei (Spalte `agent` ≠ `cli`, ≠ leer). Messwerte (Tool-Aufrufe, Fehlversuche, Runden pro Modul) im 3b-Ledger.
6. `qal export` → `std.qalpkg` committen.

---

## 5. Sicherheit

### 5.1 SSRF (ausgehendes `net`)
- Auflösung **im Host** (eigener `ureq`-Resolver): alle Adressen des Namens werden geprüft; **eine einzige gesperrte Adresse sperrt den Namen**. Verbunden wird nur mit den geprüften Adressen (keine zweite Auflösung → kein DNS-Rebinding zwischen Prüfung und Verbindung).
- Gesperrt für DNS-Namen: `0.0.0.0/8`, `10/8`, `100.64/10`, `127/8`, `169.254/16` (Cloud-Metadaten), `172.16/12`, `192.0.0/24`, `192.168/16`, `198.18/15`, `224/4`, `240/4`, `255.255.255.255`, `::/128`, `::1`, `fc00::/7`, `fe80::/10`, `ff00::/8`, IPv4-gemappte/kompatible IPv6-Formen (`::ffff:0:0/96`, `64:ff9b::/96`) der gesperrten IPv4-Bereiche.
- Ein **IP-Literal-Eintrag** (`net:127.0.0.1:9000`) erlaubt genau diese Adresse und diesen Port — bewusste Betreiberentscheidung, nötig für lokale Test-Mocks. `localhost` als Name ist gesperrt; Literal verwenden.
- Keine Redirects, keine Proxies, Port-Bindung an den Eintrag, Aufrufbudget (§2.5).

### 5.2 SQL-Injection
Unmöglich per Konstruktion (§3.2): Literal-Pflicht für SQL (Prüffehler), positionale Parameter, Einzelanweisung, Authorizer, kein ATTACH/PRAGMA/Transaktionssteuerung, keine Extensions.

### 5.3 `fs` mit `openat` (Übertrag)
- Jede freigegebene Wurzel wird beim Start einmal als Verzeichnis-Deskriptor geöffnet (`O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`, Crate `rustix`).
- Ein Guest-Pfad wird **lexikalisch** gegen die Wurzel aufgelöst (relativ zur Basis; `..`-Komponenten, die die Wurzel verlassen würden, absolute Pfade außerhalb, NUL → uniforme Ablehnung; versteckte Komponenten bei `no_hidden` → Ablehnung) und dann **Komponente für Komponente** mit `openat(dirfd, comp, O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)` geöffnet; die letzte Komponente mit `O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC` (Schreiben: `| O_CREAT | O_WRONLY`, Modus 0o644, ohne `O_TRUNC`; nach `fstat`-Prüfung „reguläre Datei“ `ftruncate(0)`). Ein Symlink an beliebiger Stelle → `ELOOP`/`ENOTDIR` → uniforme Ablehnung.
- Damit entfallen `canonicalize`, `verify_opened` und der Rest-Wettlauf (leere Datei außerhalb). Bestehende Tests in `host.rs` bleiben grün; neuer Test: Verzeichnis-Tausch gegen Symlink zwischen zwei Schritten erzeugt keine Datei außerhalb.
- Wurzel kann nicht zwischen Freigabe und Nutzung ausgetauscht werden, weil der Deskriptor beim Start gehalten wird.

### 5.4 Host-Speicher und Budgets (Übertrag)
- Pro Ausführung ein **Host-Budget** in Bytes = Guest-Speicherlimit. Belastet werden: gelesene Dateien, gepufferter `print`-Text, eingehender Request-Body + Header (vor dem Guest), ausgehende Antwort-Bodies, DB-Ergebnisse (Summe der Text-/Zeilengrößen vor dem Kodieren), Fehlerwerte. Überschreitung → Trap `out_of_memory` (E0504) mit `values: {requested_bytes, limit_bytes, host: true}`.
- Freigabe des Budgets beim Ende der Ausführung (pro Anfrage neu).
- **Schreibbudget `fs`:** Bytes und Anzahl `write_file` pro Ausführung (§2.5); Überschreitung → Trap `budget_exceeded` (E0522) mit `values: {budget, used, kind: "fs_bytes"|"fs_writes"}`. Begründung für Trap statt `Err`: es ist eine Ressourcengrenze wie OOM, kein behandelbarer E/A-Fehler.

### 5.5 DoS pro Anfrage
Fuel, Epoch-Timeout, Speicher, Host-Budget, Header-/Body-Limits und -Zeitlimits (Slowloris), begrenzte Warteschlange mit 503, Keep-Alive-Grenzen, Netz-Aufrufbudget, DB-Zeilenlimit und `max_page_count`. Kein Limit wird vom Guest beeinflusst. Standard-Lauschadresse `127.0.0.1:8080`; `0.0.0.0` nur explizit.

---

## 6. Diagnosen und Laufzeitfehler (neu)

| Code | kind | Ebene | Bedeutung |
|---|---|---|---|
| E0240 | `reserved_name` | Prüfung/Store | `std.`- oder Prelude-Name im Nutzerprojekt definiert/umbenannt; unqualifizierter Name im std-Projekt |
| E0241 | `sql_not_literal` | Prüfung | `sql` von `db_exec`/`db_query` ist kein Str-Literal |
| E0242 | `sql_invalid` | Prüfung | Literal: mehrere Anweisungen, benannte Platzhalter, verbotenes Schlüsselwort, unvollständig |
| E0243 | `package_mismatch` | Import | Prüfsumme oder Hash nach Import ≠ Paketangabe |
| E0244 | `std_signature_change` | Import | Paket ändert Signatur eines bestehenden `std.`-Namens |
| E0520 | `bad_entry` | Start | Handler/Init mit falscher Signatur oder unbekannt |
| E0521 | `bad_response` | Laufzeit | Handler-Antwort ungültig (§2.6) |
| E0522 | `budget_exceeded` | Laufzeit | fs-Schreib- oder Netz-Aufrufbudget erschöpft |
| E0523 | `db_open` | Start | DB-Datei nicht öffenbar (Symlink, Rechte, beschädigt) |
| E0800 | `request_rejected` | Service-Log | Host lehnte HTTP-Anfrage ab (400/408/411/413/431/501) |
| E0801 | `overloaded` | Service-Log | Warteschlange voll (503) |
| E0802 | `listen_failed` | Start | Adresse nicht bindbar |
| E0819 | `internal` | Service-Log | Panic im Worker (Bug) gefangen |

`out_of_memory` (E0504) bekommt das Wertfeld `host: true` für Host-Budget; `permission_denied` (E0508) deckt `net:`/`db`-Freigaben ab; `NetDenied` bleibt ein `Err`-Wert.

---

## 7. CLI und MCP

### 7.1 CLI

```
qal serve [handler=handle] [--listen 127.0.0.1:8080] [--allow …] [--init <fn>] [--workers 4] [--queue 64]
          [--request-timeout 5000] [--fuel N] [--memory 64] [--max-body 1024] [--max-header-bytes 16384]
          [--max-response 8192] [--header-timeout 5000] [--body-timeout 10000] [--net-calls 16]
          [--fs-write-budget 64] [--db-max-rows 10000] [--db-max-mib 1024] [--debug-errors] [--log <file>]
qal export <prefix> --out <file>       # z. B. qal export std. --out std/std.qalpkg
qal import <file>
qal request [handler] --allow … [--init <fn>] <requests.json>   # gleiche Semantik wie qal_request, Ausgabe JSON
```
- `--listen 127.0.0.1:0` wählt einen freien Port; die `listening`-Logzeile nennt den echten (für das Harness).
- Neue Crate `qal-serve` (HTTP-Server, Service-Pipeline `Service::handle(HttpRequest) -> (HttpResponse, Log)`), abhängig von `qal-run`; `qal-cli` hängt davon ab. `qal-run` bekommt `net`, `db`, Budgets, openat, allgemeines Kodieren von Records/Enums in den Guest-Speicher (bisher nur Str/List) und Dekodieren der Prelude-Typen.

### 7.2 MCP

**Neues Tool `qal_request`** (Annotation: nicht read-only, nicht destruktiv, `openWorldHint: true` nur wenn `net:` freigegeben):

```json
{ "handler": "handle", "init": "setup", "allow": ["db:mem", "time"],
  "requests": [ {"method": "POST", "path": "/todos", "query": "", "headers": [{"name":"content-type","value":"application/json"}],
                 "body": "{\"title\":\"a\"}"},
                {"method": "GET", "path": "/todos", "json": null} ] }
```
- Führt `init` (falls angegeben) und dann die Anfragen **der Reihe nach** gegen **dieselbe** Datenbank aus (bei `db:mem` eine frische pro Tool-Aufruf; `db`/`db:<name>` nur, wenn ausdrücklich freigegeben). Jede Anfrage durchläuft exakt die Serve-Pipeline ohne TCP (gleiche Limits, gleiche Transaktionsregel, gleiche `bad_response`-Prüfung).
- Feld `json` (statt `body`) serialisiert den Wert und setzt `content-type: application/json`. Max. 50 Anfragen pro Aufruf.
- Ergebnis pro Anfrage: `{status, headers, body, json?` (falls `content-type` JSON und parsebar)`, db: "commit"|"rollback"|"none", stdout?, error?}` mit vollem RunError in `error`.
- **Kein `qal_serve`-Tool.** Begründung: ein langlaufender Server-Prozess aus einem stdio-MCP-Server heraus hätte Lebenszyklus-, Port- und Aufräumprobleme und bringt Agenten nichts, was `qal_request` nicht deterministisch liefert. HTTP-Ebene testet das Rust-Harness gegen `qal serve`.

**Bestehende Tools:** `allow` akzeptiert `net:…`, `db`, `db:<name>`, `db:mem`. `qal_define`-Beschreibung (Sprachreferenz) bekommt einen ≤ 12-zeiligen Abschnitt „Services“: Prelude-Typen, `db_exec`/`db_query` mit Literal-Regel, `http`, Transaktionsregel, Verweis auf `qal_query name: "std."`. Die std selbst wird **nicht** in die Referenz kopiert (bleibt ≤ 1 Seite + Abschnitt); `qal_view` zeigt std-Definitionen wie alle anderen.

---

## 8. Beispiel-Backend: Todo-API

Ort: `qal/examples/todo/` (eigenes QAL-Projekt; Store nicht eingecheckt, Export `todo.qalpkg` eingecheckt). Gebaut von Agenten über MCP nach demselben Verfahren wie §4.4 (Karte → Subagent → versteckte `qal_request`-Szenarien → Review).

**Schema** (`fn setup() -> Result[Unit, Str] !db`):
`CREATE TABLE IF NOT EXISTS todos (id INTEGER PRIMARY KEY, title TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0 CHECK (done IN (0,1)), created_at INTEGER NOT NULL)`

**Handler** `fn handle(req: HttpRequest) -> HttpResponse !db !time` über `std.route` mit Nutzer-Enum `Endpoint`.

| Methode + Pfad | Erfolg | Fehler |
|---|---|---|
| `GET /health` | 200 `{"ok":true}` | — |
| `GET /todos[?done=true\|false]` | 200 Array nach `id` aufsteigend | 400 bei anderem `done`-Wert |
| `POST /todos` Body `{"title": Str, "done"?: Bool}` | 201 Todo, Header `location: /todos/<id>` | 415 Content-Type, 400 JSON/Validierung |
| `GET /todos/:id` | 200 Todo | 404, 400 wenn `id` keine positive Int |
| `PATCH /todos/:id` Body mit `title`? und/oder `done`? (mind. eines) | 200 aktualisiertes Todo | 400, 404, 415 |
| `DELETE /todos/:id` | 204 leerer Body | 404 |
| unbekannter Pfad / falsche Methode | — | 404 / 405 mit `allow` |

- Todo-JSON: `{"id":Int,"title":Str,"done":Bool,"created_at":Int}` (Schlüssel in dieser Reihenfolge).
- Validierung: `title` nach `trim` 1–200 Zeichen; unbekannte Felder → 400 (`unknown_field`); falsche Typen → 400 (`invalid_type`). Fehlerformat `std.http.error`: `{"error":{"code":"…","message":"…"}}`.
- Einfügen: `db_query("INSERT INTO todos (title, done, created_at) VALUES (?, ?, ?) RETURNING id, title, done, created_at", …)`.

**Tests:**
1. **QAL-`test`-Definitionen** (rein): Validierung, JSON-Abbildung Zeile→Todo, Routentabelle.
2. **`qal_request`-Szenarien** (in 3c als JSON-Dateien `qal/examples/todo/scenarios/*.json` mit erwarteten Antworten; ein Rust-Test führt sie über `qal request` aus): CRUD-Ablauf, alle Fehlerfälle der Tabelle, Rollback (PATCH mit ungültigem Wert ändert nichts).
3. **HTTP-Harness** (`qal/crates/qal-serve/tests/todo_http.rs`): kopiert das Projekt in ein Temp-Verzeichnis, importiert `todo.qalpkg`, startet `qal serve --listen 127.0.0.1:0 --allow db,time --init setup`, liest den Port aus der Logzeile, und prüft per rohem TCP-Client:
   - CRUD-Ablauf über echte Verbindungen inkl. Keep-Alive;
   - **Persistenz:** Server stoppen (SIGTERM), neu starten, Daten vorhanden;
   - **Nebenläufigkeit:** 50 parallele `POST` → 50 verschiedene `id`s, `GET /todos` liefert 50;
   - **Limits:** Body > 1 MiB → 413; Header > 16 KiB → 431; `Transfer-Encoding: chunked` → 411; Slowloris (Header nie abgeschlossen) → Verbindung nach ~5 s zu; ungültiges UTF-8 → 400;
   - **Isolation:** ein Test-Handler-Fixture mit Endpunkt, der trappt (`int_overflow`) → 500, die nächste Anfrage ist korrekt; Endpunkt mit Endlosrekursion → 503 nach Timeout, Server weiter verfügbar; ein Endpunkt, der 60 MiB allokiert, 20-mal hintereinander → jedes Mal 200 (Speicher pro Anfrage freigegeben);
   - **Speicher:** 10 000 `GET /todos` → RSS-Wachstum nach den ersten 1 000 ≤ 10 % (macOS/Linux über `ps -o rss`);
   - **Herunterfahren:** SIGTERM während einer langsamen Anfrage → Anfrage wird beantwortet, Exit 0.

---

## 9. Phasen der Implementierung

### 3a — Laufzeit & Service (unabhängig von Phase 2)
Prelude-Typen; Built-ins `http`, `db_exec`, `db_query` (Checker inkl. E0241/E0242, Codegen-Importe, Host); Freigabesyntax `net:`/`db`/`db:<name>`/`db:mem`; allgemeines Guest-Kodieren von Records/Enums; Transaktionsregel; SSRF-Resolver; openat-`fs`; Host-Budget und Schreibbudget; Crate `qal-serve` (httparse, Worker-Pool, Epoch-Ticker, Pooling-Allocator, Limits, Log, Shutdown); `qal serve`, `qal request`, MCP `qal_request`; Store-Schema 3 (`meta`), `qal export`/`qal import`, leeres `std.qalpkg`; Messgate §2.2; Referenz-Abschnitt „Services“. Getestet mit handgeschriebenen QAL-Fixtures in `qal/crates/qal-serve/tests/fixtures/` (Fixtures sind Testcode, nicht std).
**Fertig:** alle §5-Sicherheitstests, Limits-Tests, Messgate dokumentiert.

### 3b — Standardbibliothek (braucht 3a und Phase-2-Generics)
Schlüsselwort `std`, qualifizierte Namen in Lexer/Parser/Drucker/Kanonisierung (Rundlauf!), E0240, `meta.role`, Auto-Import beim Schreiben, E0243/E0244; std-Projekt `qal/std/`; Agenten-Verfahren §4.4 für alle sechs Module; `std.qalpkg` eingecheckt; Herkunftstest.
**Fertig:** alle versteckten Abnahmetests + Property-Tests grün, Review ohne offene Befunde, Paket importiert in ein frisches Projekt mit identischen Hashes.

### 3c — Todo-Backend & Abschluss (braucht 3b)
Todo-Projekt durch Agenten; Szenarien; HTTP-Harness §8; RSS-Messung; Roadmap-Zeile 3 auf ☑ mit Nachweisen; Abschluss-Review.

Abhängigkeiten: `3a → 3b → 3c`; 3a kann parallel zu Phase 2 beginnen; 3b beginnt erst, wenn Phase-2-Generics gemergt sind (sonst Rückfall §1 mit `tag: Str`, dokumentiert).

---

## 10. Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| Instanz pro Anfrage zu langsam | Messgate §2.2 mit vorab festgelegtem Rückfall (Heap-Reset + Nullfüllen) |
| Pooling-Allocator braucht viel virtuellen Adressraum / Feature-Flags von wasmtime 49 | `max_memory_size` = Anfragelimit (64 MiB), Fallback auf On-Demand mit Warnung |
| Agenten schreiben JSON-Parser in QAL schlecht (Performance quadratisch, Grenzfälle) | Aufgabenkarten mit Komplexitätsvorgaben (über `chars` iterieren, kein `++` in Schleifen), versteckte Tests mit 100-KiB-Eingaben, Review-Schritt, max. 3 Runden + Neustart |
| Phase-2-Generics verspätet/anders | Nur `std.route` betroffen; Rückfall `tag: Str` vordefiniert |
| `BEGIN IMMEDIATE` serialisiert alle DB-Handler | Für Phase-3-Last (Todo) ausreichend; dokumentiert; Lese-Effekt als späteres Thema |
| Qualifizierte Namen brechen Rundlauf/Hashing | Namen sind Semantik nur als Referenzen (→ Hash); Golden- und Rundlauf-Tests für `std.`-Namen in 3b zuerst |
| Literal-Pflicht für SQL zu restriktiv (dynamisches `ORDER BY`) | Bewusst: Varianten als getrennte Literale per `match`; dokumentiert in der Referenz |
| Kein TLS im Server | Betrieb hinter Reverse-Proxy dokumentiert; nicht Ziel von Phase 3 |
| macOS vs. Linux bei `openat`/RSS-Messung | `rustix` abstrahiert; RSS-Test toleriert ±10 %, läuft auf beiden; Windows: `fs`, `serve`-Harness ausgenommen (wie Phase 1 `fs`) |
