# QAL Production Operations & Recovery Runbook

**Stand:** 2026-10-20  
**Geltungsbereich:** QAL Self-Hosted Engine, Graph-Store (`.qal/store.db`), Daemon (`qal serve`), MCP-Server (`qal mcp`)  
**Zielgruppe:** SRE, Betreiber und autonome Agenten-Orchestrierungs-Systeme

---

## 1. Speicherarchitektur & Persistenzmodell

Der QAL-Store basiert auf einer relational-kanonischen Architektur:
- **Speicherort:** `<Projektverzeichnis>/.qal/store.db` (SQLite 3 im WAL-Journal-Modus).
- **Integritätsmodell:** Jede Definition ist über ihren kanonischen SHA-256-Hash eindeutig identifiziert. Änderungen erzeugen neue Versionen in einer Merkle-artigen Historie (`history`-Tabelle).
- **Transaktionsgarantie:** Schreibende Operationen (`define`, `rename`, `delete`, `import`, `merge`) laufen strikt in atomaren `BEGIN IMMEDIATE`-Transaktionen. Teilweise geschriebene Stände existieren nicht.
- **Rundlauf-Garantie:** `parse(print(def)) == def` und identischer Text $\Rightarrow$ identischer Hash.

---

## 2. Ausfallmodi & Automatische Wiederherstellung (Crash Recovery)

### 2.1 Prozessabbruch während eines Schreibvorgangs (SIGKILL / Stromausfall)
- **Verhalten:** Bricht der Prozess vor dem Transaktions-Commit ab (z. B. durch SIGKILL, OOM-Killer), verwirft das SQLite-WAL-Journal beim nächsten Öffnen der Datenbank alle unvollständigen Schreibvorgänge automatisch.
- **Historien-Konsistenz:** Der Store bleibt byte- und transaktionskonsistent. Weder Definitionen noch Metadaten gehen verloren; der Stand entspricht exakt dem Zustand vor dem abgebrochenen Schreibvorgang.
- **Nachweis:** Verifiziert im Integrationstest `a_process_killed_in_a_define_leaves_a_consistent_store` (`qal-cli/tests/concurrency.rs`) und `tests::a_failure_while_writing_rolls_everything_back` (`qal-store/src/tests.rs`).

### 2.2 Verwaiste Leases (Locks von getöteten Agenten)
- **Problem:** Ein getöteter Agent hielt Leases auf bestimmte Definitionsnamen.
- **Lösung:** Leases besitzen eine automatische Time-To-Live (`ttl_s`, Standard: 3600 s). Nach Ablauf des Zeitstempels `expires_at` betrachtet der Store die Lease automatisch als frei.
- **Manueller Eingriff (im Notfall):**
  ```sql
  -- Abgelaufene Leases bereinigen (automatisch durch qal)
  DELETE FROM leases WHERE expires_at < (strftime('%s', 'now') * 1000);
  ```

---

## 3. Gleichzeitige Zugriffe & Sperrkonflikte (`STORE_BUSY`)

### 3.1 Das `E0907`-Sperrprotokoll
- Bei hoher Last oder konkurrierenden Agenten-Schreibzugriffen blockiert SQLite den zweiten Schreiber bis zum Timeout (Standard: 10 Sekunden).
- Überschreitet der Schreiber das Zeitfenster, bricht QAL fail-closed mit Fehlercode `E0907` (`kind: "store_busy"`) ab.
- **Behebungsverfahren:**
  1. Der Client führt einen exponentiellen Backoff mit Jitter durch ($50\text{ ms} \cdot 2^k \pm \text{Jitter}$).
  2. Nach Freigabe der Sperre wird der Aufruf wiederholt.
  3. Bei kooperierenden Agenten sind granulare Leases (`qal_lease acquire`) zu nutzen, um Sperrkonflikte auf Store-Ebene zu vermeiden.
- **Nachweis:** Verifiziert in `production_concurrent_locking_and_store_busy_retry` (`qal-cli/tests/production.rs`) und `a_write_lock_held_too_long_is_store_busy` (`qal-cli/tests/concurrency.rs`).

---

## 4. Backup & Disaster Recovery

### 4.1 Logisches Paket-Backup (Empfohlen)
Logische Sicherungen exportieren den vollständigen Definitionen- und Abhängigkeitsgraphen in ein kryptografisch verifiziertes `.qalpkg`-Archiv.

```bash
# Vollständiges Projekt-Backup erstellen
qal export "" --out backup-$(date +%Y%m%d-%H%M%S).qalpkg

# Wiederherstellung in ein frisches Verzeichnis
mkdir restored-service && cd restored-service
qal init
qal import ../backup-20261020-220000.qalpkg
```

- **Garantie:** Der Import validiert jeden Hash (`E0243 package_mismatch` bei Manipulation). Der wiederhergestellte Codebase-Graph ist bit-identisch zum gesicherten Original.
- **Nachweis:** Verifiziert in `production_backup_export_and_restore_reproduces_exact_codebase` (`qal-cli/tests/production.rs`).

### 4.2 Physisches Online-Backup (Hot Backup)
Da SQLite WAL aktiv ist, kann ein Snapshot im laufenden Betrieb ohne Downtime erzeugt werden:
```bash
sqlite3 .qal/store.db "VACUUM INTO 'backup-online.db'"
```

---

## 5. Schema-Migrationen

Der QAL-Store migriert beim Öffnen automatisch und atomar auf das jeweils aktuelle Schema:
- **Versionstracking:** Gespeichert im SQLite `PRAGMA user_version`.
- **Migrationskette:** $2 \rightarrow 3 \rightarrow 4 \rightarrow 5 \rightarrow 6$ (getestet in `qal-store/src/schema.rs`).
- **Gleichzeitigkeitsschutz:** Führen mehrere Prozesse parallel einen Start durch, führt der erste die Migration in einer exklusiven Transaktion aus; nachfolgende Prozesse erkennen die aktualisierte `user_version` und überspringen den Schritt.
- **Aktueller Produktionsstand:** Schema Version 6 (geprüft in `production_schema_migration_version_integrity`).

---

## 6. Ressourcen- & Ausführungsgrenzen (Quotas)

QAL schützt den Host prozessual und deterministisch gegen Denial-of-Service und Endlosschleifen:

| Ressource | Grenzwert (Default) | Fehlerverhalten |
|---|---|---|
| **Instruktionen (Fuel)** | $10^7$ (CLI/Test) bzw. $2\cdot 10^{11}$ (Selfhost) | `timeout: instruction budget exhausted` |
| **Wandzeit (Timeout)** | 10 s (Test) / 300 s (Selfhost-Item) | `timeout: execution timed out` |
| **Host-Speicher (WASM)** | 16 MiB (Default) bis 1 GiB (Compiler) | `out of memory: allocation failed` |
| **Dateisystem** | Pfad-Sandbox via `openat(O_NOFOLLOW)` | `E0301 effect_violation` / OS Access Denied |

---

## 7. Langzeit-Servicebetrieb & Speicherverwaltung

### 7.1 Arena-Speicherrecycling pro Request
- Bei Ausführung unter `qal serve` (HTTP-Daemon) wird für jeden HTTP-Request ein isolierter linearer Speicherbereich (Bump-Arena) alloziiert.
- Nach Fertigstellung der Antwort wird die Arena als Ganzes freigegeben.
- **Ergebnis:** RSS-Speicherverbrauch bleibt auch über $10\,000+$ kontinuierliche Anfragen hinweg konstant ($+0{,}0\%$ Memory Leak nach den ersten $1\,000$ Warmup-Requests bei $\approx 7\,500\text{ Req/s}$).
- **Nachweis:** Verifiziert in `todo_scenarios.rs` und `qal-serve/src/tests.rs`.

---

## 8. Reproduzierbare Releases & Version-Rollback

- **Unveränderliche Historie:** Definitionshashes sind inhaltsadressiert. Eine Änderung an einer Funktion $f$ erzeugt eine neue Version mit neuem Hash.
- **Instant Rollback:** Um auf eine vorherige Version zurückzurollen, wird die Funktion einfach mit dem ursprünglichen Text erneut definiert:
  ```bash
  qal define function_v1.qal
  ```
  Da der Hash derselbe ist wie zuvor, werden alle abhängigen Definitionen ohne Hash-Churn automatisch wieder an den bewährten Stand gebunden (`relink`).
- **Nachweis:** Verifiziert in `production_reproducible_release_and_version_rollback` (`qal-cli/tests/production.rs`).
