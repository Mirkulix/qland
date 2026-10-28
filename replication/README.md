# QAL Replication Package

> **Für externe Prüfer, Wissenschaftler und Auditoren**  
> Dieses Paket ermöglicht die Replikation der formalen und kryptographischen Kern-Behauptungen von **QAL (QLANG Agent Language)** auf Linux- oder macOS-Systemen. Der beiliegende `replication_report.md` stammt von der Entwicklungsmaschine und ist damit eine Selbstprüfung; eine **unabhängige** Replikation liegt erst vor, wenn jemand anderes dieses Paket auf einer anderen Maschine ausführt und seinen Bericht beilegt.
>
> **Wichtig für frische Clones:** Die Claims 1 und 2 laufen ohne lokalen Pilot-Zustand; Claim 5 benötigt ein lokal angelegtes `pilot/escrow/qal/.qal`-Verzeichnis (nicht versioniert), baut die Datenbank darin aber selbst neu auf. Die Claims 3 (Audit-Kette) und 4 (Truncation-Detection) sind **nur auf einer Maschine mit vorhandenem Pilot-Zustand** reproduzierbar: Sie benötigen den nicht versionierten Merkle-Store `pilot/escrow/qal/.qal/store.db` und die Trust-Datei `~/.config/qal/sessions/escrow/trust.json`. Beide sind nicht getrackt (`.gitignore` schließt `.qal/` und `*.db` aus; die Trust-Datei liegt außerhalb des Repositories) und in einem frischen Clone nicht vorhanden. Ein frischer Clone muss dieses Fixture erst selbst erzeugen und den dann gültigen Head verwenden; ohne das schlagen Claims 3 und 4 fehl (siehe §1.1).

---

## 1. Übersicht der zu replizierenden Behauptungen

| Behauptung | Gegenstand | Prüfwerkzeug / Soll-Ergebnis |
|---|---|---|
| **Claim 1: Wasm-Fixpunkt** | Der selbstgehostete Wasm-Compiler erzeugt byte-identischen Bytecode zur Rust-Referenz | `cargo test -p qal-gen --test full_fixpoint --release` $\rightarrow$ 2/2 tests OK |
| **Claim 2: 23 Z3-Beweise** | Der Escrow-Kernel beweist 23 Obligations (10 `ens` + 13 Aufruf-`req`; Erhaltungssatz, Schwellen, Nicht-Negativität) | `qal/target/release/qal prove pilot/escrow/qal/escrow.qal` $\rightarrow$ `proven: 23, refuted: 0, unknown: 0` |
| **Claim 3: Audit-Integrität** | SHA-256 Merkle-Kette mit Ed25519-Signaturen und fixiertem Head — **nur Entwicklungsmaschine**, benötigt den nicht versionierten Store + Trust-Datei¹ | `qal verify --trust <trust.json> --head <hash>` $\rightarrow$ 0 Warnungen |
| **Claim 4: Truncation-Detection** | Das unbemerkte Abschneiden neuerer Records wird strikt verhindert — **nur Entwicklungsmaschine**, benötigt den nicht versionierten Store + Trust-Datei¹ | `python3 tools/test_truncation_detection.py` $\rightarrow$ Abbruch mit `E0922 chain_broken` |
| **Claim 5: Zero-Sum Invariante** | Die Geldsumme bleibt unter 50 parallelen Transaktionen und Backups exakt erhalten | `python3 pilot/escrow/ops/load_test.py` $\rightarrow$ 0 Overdrafts, Saldo $\ge 0$ |

¹ Siehe §1.1: Der Store `pilot/escrow/qal/.qal/store.db` und die Trust-Datei `~/.config/qal/sessions/escrow/trust.json` sind nicht versioniert. Claims 3 und 4 sind daher keine Ein-Befehl-Claims für einen frischen Clone, sondern Entwicklungsumgebungs-Claims.

### 1.1 Reproduzierbarkeit aus einem frischen Clone

| Claim | Aus frischem Clone reproduzierbar? | Benötigter Zustand |
|---|---|---|
| **1: Wasm-Fixpunkt** | ✅ ja | Rust-Toolchain; baut `qal-gen` selbst |
| **2: 23 Z3-Beweise** | ✅ ja | Release-Binary `qal/target/release/qal`, sonst `QAL_BIN` setzen |
| **3: Audit-Integrität** | ❌ nein | `pilot/escrow/qal/.qal/store.db` + `~/.config/qal/sessions/escrow/trust.json` (beide nicht versioniert); der Runner pinnt zusätzlich den lokalen Head `4be3934d…` |
| **4: Truncation-Detection** | ❌ nein | derselbe lokale Store + Trust-Datei; der Test liest den aktuellen Head und benötigt die Records des Pilot-Verlaufs |
| **5: Zero-Sum/Load** | ⚠️ bedingt | `verify_ops.py` kopiert `pilot/escrow/qal/.qal/` in seine Runtime (Verzeichnis nicht versioniert), löscht danach die DB und baut den Zustand neu auf; ohne dieses Verzeichnis schlägt der Kopierschritt fehl. Die historischen Records und der Head werden nicht benötigt. |

> **Claims 3 und 4 sind Entwicklungsmaschinen-Claims.** Sie laufen gegen den lokalen Merkle-Store und die Trust-Datei, die beide nicht im Repository liegen. Ein frischer Clone muss das Fixture erst selbst erzeugen (frischen Store anlegen, z. B. `qal init --std` bzw. `qal team init` + `qal team export-trust --out trust.json`, und den dann gemeldeten Head verwenden) oder den Pilot-Store bereitstellen. Da der Runner zusätzlich den historischen Head `4be3934d…` pinnt, ist ein 5/5-Lauf nur mit vorhandenem Pilot-Zustand oder angepasstem Pin erreichbar. Ein 5/5-Ergebnis belegt also die Reproduktion auf der Entwicklungsmaschine, nicht die Unabhängigkeit von einem sauberen Clone.

---

## 2. Voraussetzungen für Replikationsrechner

* **Betriebssystem:** macOS (Apple Silicon oder Intel) oder Linux (x86_64 oder aarch64, z. B. Ubuntu 22.04 / 24.04, Debian 12).
* **Rust Toolchain:** `rustc 1.80+` und `cargo` ([rustup.rs](https://rustup.rs)).
* **Python:** `python 3.8+` mit `requests`.
* **C++ Compiler & CMake:** für die Z3-Kompilierung (z. B. `build-essential cmake clang`).
* **Nur für Claims 3–4:** ein lokaler Pilot-Zustand — der Merkle-Store `pilot/escrow/qal/.qal/store.db` und die Trust-Datei `~/.config/qal/sessions/escrow/trust.json`. Beide sind **nicht versioniert** (`.gitignore`: `.qal/`, `*.db`) und in einem frischen Clone nicht vorhanden; siehe §1.1.
* **Für Claim 5:** ein lokal angelegtes Verzeichnis `pilot/escrow/qal/.qal/` (nicht versioniert). `verify_ops.py` kopiert es in seine Runtime und baut die Datenbank darin neu auf; die historischen Records und der Head werden nicht benötigt.

---

## 3. Automatisierte Replikation (Runner)

Führen Sie im Wurzelverzeichnis des Repositories folgenden Befehl aus:

```bash
python3 replication/replicate.py
```

Oder alternativ über das Shell-Skript:

```bash
chmod +x replication/replicate.sh
./replication/replicate.sh
```

Das Skript führt alle 5 Replikationsstufen aus, erfasst Maschinentelemetrie (OS, CPU, Toolchain) und generiert den formalen Prüfbericht unter:
[`replication/replication_report.md`](./replication_report.md).

**Einschränkung:** Der Runner führt zwar alle 5 Stufen aus, aber Claims 3 und 4 sind **keine Ein-Befehl-Claims für einen frischen Clone**. Ohne den lokalen Pilot-Store und die Trust-Datei (§1.1) schlagen sie fehl; ein 5/5-Lauf setzt den vorhandenen Pilot-Zustand der Entwicklungsmaschine voraus. Ohne lokalen Zustand laufen nur die Claims 1 und 2; Claim 5 benötigt zusätzlich ein lokal angelegtes `pilot/escrow/qal/.qal`-Verzeichnis (nicht versioniert).

Der Bericht wird vom Runner selbst aus einer festen Vorlage erzeugt (Selbstprüfung, keine unabhängige Drittprüfung). Die Soll-Werte sind im Skript hinterlegt (`EXPECTED_PROOFS = 23`, fixierter Head `4be3934d…`); die gemessenen Ergebniswerte übernimmt der Bericht aus dem jeweiligen Lauf.

---

## 4. Manuelle Replikation Schritt für Schritt

### Schritt 1: Fixpunkt unabhängig verifizieren
```bash
cargo test -p qal-gen --test full_fixpoint --release
```
*Erwartetes Ergebnis:*
```text
test seed_module_is_the_fixpoint ... ok
test full_selfhosted_fixpoint ... ok
```

### Schritt 2: 23 Z3-Obligations formal beweisen
```bash
qal/target/release/qal prove pilot/escrow/qal/escrow.qal
```
*Erwartetes Ergebnis:*
```json
{"proven": 23, "refuted": 0, "unknown": 0}
```
Die 23 bewiesenen Obligations setzen sich aus 10 `ens` und 13 Aufruf-`req`-Prüfungen zusammen; die Datei deklariert 15 `req` + 10 `ens`.

### Schritt 3: Kryptographischen Audit-Head prüfen — **nur Entwicklungsmaschine**
```bash
(cd pilot/escrow/qal && ../../../qal/target/release/qal verify --trust ~/.config/qal/sessions/escrow/trust.json --head 4be3934dda9f5a96e207cd9773af5814f5fd66c0c6ac573c0ab1ce63860e9ab7)
```
*Erwartetes Ergebnis:*
```json
{"head":"4be3934dda9f5a96e207cd9773af5814f5fd66c0c6ac573c0ab1ce63860e9ab7","members":5,"ok":true,"records":19,"trust":"file"}
```
*Voraussetzung:* Der Store `pilot/escrow/qal/.qal/store.db` und die Trust-Datei `~/.config/qal/sessions/escrow/trust.json` sind **nicht versioniert** (`.gitignore` schließt `.qal/` und `*.db` aus). Ein frischer Clone enthält sie nicht; der fixierte Head `4be3934d…` gehört zum lokalen Pilot-Verlauf. Auf einem frischen Clone muss erst ein eigenes Fixture mit eigener Trust-Datei erzeugt und der dann gemeldete Head verwendet werden (siehe §1.1).

### Schritt 4: Angriffsresilienz testen (Truncation Attack) — **nur Entwicklungsmaschine**
```bash
python3 tools/test_truncation_detection.py
```
*Erwartetes Ergebnis:*
Der Test klont den Store, schneidet den neuesten Record ab und verifiziert, dass `qal verify --head` die Manipulation mit `E0922 chain_broken` abweist.
*Voraussetzung:* derselbe lokale Store und dieselbe Trust-Datei wie in Schritt 3; der Test liest den aktuellen Head des Pilot-Verlaufs. Ohne dieses Fixture ist der Test nicht ausführbar.

### Schritt 5: Vollständige Betriebs- und Lasttest-Suite
```bash
python3 pilot/escrow/ops/verify_ops.py
```
*Erwartetes Ergebnis:*
Alle 9 Teilschritte (20/20 Benchmark-Tests, 50 parallele Competing Payouts, Online-Backup, Disaster-Recovery Restore) laufen zu 100 % grün durch.
