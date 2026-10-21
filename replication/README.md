# QAL Independent Replication Package

> **Für externe Prüfer, Wissenschaftler und Auditoren (Third-Party Replication)**  
> Dieses Paket ermöglicht die unabhängige, lückenlose Replikation aller formalen und kryptographischen Kern-Behauptungen von **QAL (QLANG Agent Language)** auf beliebigen Linux- oder macOS-Systemen.

---

## 1. Übersicht der zu replizierenden Behauptungen

| Behauptung | Gegenstand | Prüfwerkzeug / Soll-Ergebnis |
|---|---|---|
| **Claim 1: Wasm-Fixpunkt** | Der selbstgehostete Wasm-Compiler erzeugt byte-identischen Bytecode zur Rust-Referenz | `cargo test -p qal-gen --test full_fixpoint --release` $\rightarrow$ 2/2 tests OK |
| **Claim 2: 22 Z3-Beweise** | Der Escrow-Kernel beweist 22 Kontrakte (Erhaltungssatz, Schwellen, Nicht-Negativität) | `qal prove pilot/escrow/qal/escrow.qal` $\rightarrow$ `proven: 22, refuted: 0, unknown: 0` |
| **Claim 3: Audit-Integrität** | SHA-256 Merkle-Kette mit Ed25519-Signaturen und fixiertem Head | `qal verify --trust <trust.json> --head <hash>` $\rightarrow$ 0 Warnungen |
| **Claim 4: Truncation-Detection** | Das unbemerkte Abschneiden neuerer Records wird strikt verhindert | `python3 tools/test_truncation_detection.py` $\rightarrow$ Abbruch mit `E0922 chain_broken` |
| **Claim 5: Zero-Sum Invariante** | Die Geldsumme bleibt unter 50 parallelen Transaktionen und Backups exakt erhalten | `python3 pilot/escrow/ops/load_test.py` $\rightarrow$ 0 Overdrafts, Saldo $\ge 0$ |

---

## 2. Voraussetzungen für Replikationsrechner

* **Betriebssystem:** macOS (Apple Silicon oder Intel) oder Linux (x86_64 oder aarch64, z. B. Ubuntu 22.04 / 24.04, Debian 12).
* **Rust Toolchain:** `rustc 1.80+` und `cargo` ([rustup.rs](https://rustup.rs)).
* **Python:** `python 3.8+` mit `requests`.
* **C++ Compiler & CMake:** für die Z3-Kompilierung (z. B. `build-essential cmake clang`).

---

## 3. Ein-Befehl-Replikation (Automated Runner)

Führen Sie im Wurzelverzeichnis des Repositories folgenden Befehl aus:

```bash
python3 replication/replicate.py
```

Oder alternativ über das Shell-Skript:

```bash
chmod +x replication/replicate.sh
./replication/replicate.sh
```

Das Skript führt alle 5 Replikationsstufen isoliert aus, erfasst Maschinentelemetrie (OS, CPU, Toolchain) und generiert den formalen Prüfbericht unter:
[`replication/replication_report.md`](./replication_report.md).

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

### Schritt 2: 22 Z3-Kontrakte formal beweisen
```bash
bin/qal prove pilot/escrow/qal/escrow.qal
```
*Erwartetes Ergebnis:*
```json
{"proven": 22, "refuted": 0, "unknown": 0}
```

### Schritt 3: Kryptographischen Audit-Head prüfen
```bash
(cd pilot/escrow/qal && ../../../bin/qal verify --trust ~/.config/qal/sessions/escrow/trust.json --head 4be3934dda9f5a96e207cd9773af5814f5fd66c0c6ac573c0ab1ce63860e9ab7)
```
*Erwartetes Ergebnis:*
```json
{"head":"4be3934dda9f5a96e207cd9773af5814f5fd66c0c6ac573c0ab1ce63860e9ab7","members":5,"ok":true,"records":19,"trust":"file"}
```

### Schritt 4: Angriffsresilienz testen (Truncation Attack)
```bash
python3 tools/test_truncation_detection.py
```
*Erwartetes Ergebnis:*
Der Test klont den Store, schneidet den neuesten Record ab und verifiziert, dass `qal verify --head` die Manipulation mit `E0922 chain_broken` abweist.

### Schritt 5: Vollständige Betriebs- und Lasttest-Suite
```bash
python3 pilot/escrow/ops/verify_ops.py
```
*Erwartetes Ergebnis:*
Alle 9 Teilschritte (20/20 Benchmark-Tests, 50 parallele Competing Payouts, Online-Backup, Disaster-Recovery Restore) laufen zu 100 % grün durch.
