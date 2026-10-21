# 05 — Inhaltsadressierter Merkle-Store & Crash-Resilienz
> **Das tatsächliche SQLite-Datenbankschema (.qal/store.db), Merkle-Hashing und Crash-Resilienz**

---

## 1. Das tatsächliche SQLite-Schema (`.qal/store.db`)

Die QAL-Codebasis liegt in einer SQLite-Datenbank (`schema.rs`, Schema-Version 6). Es existieren folgende Kern-Tabellen:

```sql
-- 1. Inhaltsadressierte Definitionen (ASTs & Bytecode)
CREATE TABLE defs (
    hash        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('type', 'fn', 'test')),
    canonical   BLOB NOT NULL,
    signature   TEXT NOT NULL,
    effects     TEXT NOT NULL,
    local_names TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);

-- 2. Namenszeiger auf Hashes
CREATE TABLE names (
    name        TEXT PRIMARY KEY,
    hash        TEXT NOT NULL REFERENCES defs(hash),
    local_names TEXT NOT NULL,
    refs        TEXT NOT NULL,
    updated_at  INTEGER NOT NULL
);

-- 3. Abhängigkeitsgraph zwischen Hashes
CREATE TABLE deps (
    hash     TEXT NOT NULL REFERENCES defs(hash),
    dep_hash TEXT NOT NULL REFERENCES defs(hash),
    PRIMARY KEY (hash, dep_hash)
);

-- 4. Z3-Beweis-Cache
CREATE TABLE proofs (
    hash    TEXT NOT NULL REFERENCES defs(hash),
    prover  TEXT NOT NULL,
    results TEXT NOT NULL,
    at      INTEGER NOT NULL,
    PRIMARY KEY (hash, prover)
);

-- 5. Änderungshistorie (Audit-Trail)
CREATE TABLE history (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    action    TEXT NOT NULL CHECK (action IN ('added', 'replaced', 'relinked', 'renamed', 'deleted')),
    old_hash  TEXT,
    new_hash  TEXT NOT NULL,
    from_name TEXT,
    to_name   TEXT,
    agent     TEXT NOT NULL,
    at        INTEGER NOT NULL,
    space     TEXT NOT NULL DEFAULT 'main',
    record    INTEGER NULL
);

-- 6. Team-Modus: Commit-Bündel (Records) & Leases
CREATE TABLE records (
    seq  INTEGER PRIMARY KEY,
    hash TEXT NOT NULL UNIQUE,
    body BLOB NOT NULL,
    sig  TEXT NOT NULL
);

CREATE TABLE leases (
    name        TEXT PRIMARY KEY,
    change      TEXT NOT NULL,
    member      TEXT NOT NULL,
    acquired_at INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL
);
```

### Wichtige Unterscheidung für KIs:
* **`defs`** speichert die kanonischen AST-Definitionen, indexiert nach ihrem SHA-256 Hash.
* **`names`** bildet Namen (`settle_total`) auf die jeweiligen Definitionshashes ab.
* **`records`** speichert **nicht** einzelne Funktions-ASTs, sondern signierte Commit-Bündel im Team-Modus (`seq`, `hash`, `body BLOB`, `sig`).
* **`proofs`** speichert die Z3-Beweisergebnisse je Definitionshash und Prover-Key.

---

## 2. Kanonische Inhaltsadressierung & Namensauflösung

* **Kanonische Serialisierung:** Definitionen werden vor dem Hashing normalisiert (Signatur, Typen, Rumpf). Kommentare und irrelevante Whitespaces verändern den Hash nicht.
* **Referenzen binden an Hashes:** Das Feld `names.refs` enthält die Liste aller referenzierten Namen. Nutzer folgen dem Namen, gegen den der Code geschrieben wurde, selbst wenn andere Namen denselben Hash teilen.
* **Aliasing (`E0404 alias_reference`):** Mehrere Namen dürfen denselben Definitionshash besitzen. Eine Definition darf jedoch nicht zwei Alias-Namen gleichzeitig im Quelltext referenzieren, da `names.refs` jeden referenzierten Hash genau einem Namen zuordnet.

---

## 3. Belegte Crash-Resilienz & Härtetests (`concurrency.rs`)

Die Transaktions- und Crashsicherheits-Garantien sind durch automatisierte Integrationstests (`qal/crates/qal-cli/tests/concurrency.rs`) belegt:

1. **Atomare Wiederherstellung bei Prozessabbruch (`a_process_killed_in_a_define_leaves_a_consistent_store`):**
   * Ein Prozess wird an verschiedenen Zeitpunkten während eines 1.000-Namen-`define` per `SIGKILL` hart beendet (Parsing, Z3-Beweiser, schreibende Transaktion).
   * **Ergebnis:** Die SQLite-Transaktion hinterlässt entweder den vollständigen neuen Stand oder den unveränderten alten Stand (*All-or-Nothing*). Es entstehen keine halben Definitionen oder korrupten Store-Zustände.
2. **Parallele Last (`eight_processes_with_random_operations_lose_nothing`):**
   * 8 Prozesse führen gleichzeitig 1.600 zufällige Schreib-, Lese- und Merge-Operationen aus.
   * **Ergebnis:** Über WAL-Modus und `BUSY_TIMEOUT` (10s) gehen keine Daten verloren; SQLite-Locks werden deterministisch abgewickelt.
3. **Pfadwechsel-Schutz (TOCTOU-Absicherung):**
   * Dateifreigaben werden über `openat` mit `O_NOFOLLOW` aufgelöst, um Symlink-Angriffe außerhalb des Projektverzeichnisses zu verhindern.
