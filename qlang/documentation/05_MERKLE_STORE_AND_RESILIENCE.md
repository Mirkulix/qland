# 05 — Inhaltsadressierter Merkle-Store & Crash-Resilienz
> **SQLite Merkle DAG, nebenläufige KI-Schwärme und Transaktionssicherheit**

---

## 1. Die Datenbank-Struktur (`.qal/store.db`)

Das Dateisystem wird in QAL nicht für Quelltexte genutzt. Stattdessen verwaltet eine eingebettete SQLite-Datenbank (`.qal/store.db`) im WAL-Modus (Write-Ahead Logging) den gesamten Code:

```
[Store: .qal/store.db]
  ├── records          (Kanonische AST-Definitionen, indexiert nach SHA-256)
  ├── change_names     (Zuordnung: Name -> Hash im aktuellen Branch)
  ├── proofs           (Z3-Beweis-Cache mit Bestätigungsbudget)
  ├── leases           (Schreibsperren für parallele Agenten-Änderungen)
  └── history          (Unveränderlicher Audit-Trail mit Timestamps)
```

---

## 2. Kanonisches Merkle-Hashing (SHA-256)

Jede Definition wird vor dem Hashing in eine standardisierte, kanonische Byterepräsentation übersetzt:
* Kommentare und irrelevante Leerzeichen beeinflussen den Hash nicht.
* Typ-Signaturen und Kontrakte gehen zwingend in den Hash ein.
* **Ergebnis:** Zwei syntaktisch identische Funktionen erhalten exakt denselben SHA-256 Hash.

### Schutz vor Duplikaten (`E0404 alias_reference`)
Wenn eine KI zwei Funktionen mit identischem Rumpf und Kontrakt anlegt:
```qal
fn vote_yes(x: Int) -> Int = x + 1
fn vote_no(x: Int) -> Int = x + 1
```
erhalten beide denselben Hash `#b2fb1ba07d0a`. Wenn ein Test nun beide Namen referenziert, erzwingt QAL über den Fehler `E0404 alias_reference` eine saubere, kanonische Refaktorisierung (z. B. `fn vote_inc(count: Int) -> Int`).

---

## 3. Crash-Resilienz & Härtetest-Garantien

Der QAL-Store wurde in automatisierten Stresstests auf extreme Ausnahmesituationen geprüft:

1. **Harter Prozessabbruch (`SIGKILL` mitten im Schreiben):**
   * Selbst wenn der Compiler während des Schreibens von 1.000 Definitionen per `SIGKILL` getötet wird, hinterlässt SQLite dank atomarer Transaktionen eine vollkommen unkorrumpierte Datenbank. Es gilt: *All-or-Nothing*.
2. **8 Parallele Prozesse unter Volllast:**
   * 8 Prozesse führen gleichzeitig tausende Lese-, Schreib- und Merge-Operationen aus (`eight_processes_with_random_operations_lose_nothing`).
   * Keine Transaktion geht verloren, WAL-Locks werden zuverlässig verwaltet.
3. **Schutz vor Pfadwechsel (TOCTOU-Absicherung):**
   * Dateioperationen bei Dateifreigaben nutzen verankerte Verzeichnisdeskriptoren (`openat` mit `O_NOFOLLOW`), wodurch Symlink-Angriffe und Race Conditions außerhalb des Projektverzeichnisses physisch blockiert sind.
