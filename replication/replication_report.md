# Replikationsbericht (Selbstprüfung): QAL Core & Escrow Engine

**Datum (UTC):** 2026-10-26T20:49:57.229760+00:00
**Git Commit:** `ad66b5c3443be1dcb4c1a22cb85efd77d757d1fe`
**Plattform:** macOS-26.6.2-arm64-arm-64bit (arm64)  
**Compiler / Toolchain:** `rustc 1.97.1 (8bab26f4f 2026-07-14) (Homebrew)` | Python `3.11.7`  

**Art des Berichts:** Selbstprüfung. Dieser Bericht wird vom Runner selbst aus einer festen Vorlage erzeugt; eine **unabhängige** Replikation liegt erst vor, wenn eine externe Person den Lauf auf ihrer Maschine wiederholt und ihren eigenen Bericht beilegt.
**Hinweis zu Claims 3–4:** Nur auf einer Maschine mit vorhandenem Pilot-Zustand reproduzierbar. Benötigt werden der nicht versionierte Store `pilot/escrow/qal/.qal/store.db` und die Trust-Datei `~/.config/qal/sessions/escrow/trust.json`; ein frischer Clone muss dieses Fixture erst selbst erzeugen (siehe `replication/README.md` §1.1).

---

## 1. Zusammenfassung der Replikations-Ergebnisse

| Formale Behauptung | Gegenstand | Soll | Gemessen | Status |
|---|---|---|---|---|
| **Claim 1: Wasm-Fixpunkt** | Selbstgehosteter Compiler | Byte-identisch | Byte-identisch (13.882 s) | ✅ Bestanden |
| **Claim 2: 23 Z3-Beweise** | Escrow-Obligations (10 `ens` + 13 Aufruf-`req`) | 23 bewiesen, 0 refuted, 0 unknown | 23 proven, 0 refuted, 0 unknown (0.406 s) | ✅ Bestanden |
| **Claim 3: Audit-Integrität** | SHA-256 Merkle-Kette mit Ed25519-Signaturen¹ | Ungebrochen, 0 Warnungen | 19 Records, Head fixiert (32.9 ms) | ✅ Bestanden |
| **Claim 4: Truncation-Detection** | Abgeschnittene Historie¹ | `E0922 chain_broken` | Erkannt und abgewiesen (0.106 s) | ✅ Bestanden |
| **Claim 5: Zero-Sum Invariante** | 50 parallele Payouts + Recovery | $\sum Balances + Escrow + Disbursed == Deposited$ | Invariante erhalten, 0 Overdrafts (2.259 s) | ✅ Bestanden |

¹ **Nur Entwicklungsmaschine:** Claim 3 und 4 laufen gegen den nicht versionierten Store `pilot/escrow/qal/.qal/store.db` und die Trust-Datei `~/.config/qal/sessions/escrow/trust.json` (beide gitignored). Aus einem frischen Clone sind sie ohne selbst erzeugtes Fixture nicht reproduzierbar.

---

## 2. Diskrepanz-Analyse

**Keine inhaltlichen oder mathematischen Diskrepanzen bei den auf dieser Maschine ausgeführten Prüfungen festgestellt.**

* **Mathematische Soundness:** Der Z3-SMT-Solver lieferte exakt 23 formal bewiesene Obligations (10 `ens` + 13 Aufruf-`req`) ohne Approximations-Unknowns.
* **Kryptographische Kette:** Die Hash-Kette mit Head `4be3934dda9f5a96e207cd9773af5814f5fd66c0c6ac573c0ab1ce63860e9ab7` ist auf dieser Maschine byte-genau konsistent. Der Store ist nicht versioniert; der Head ist ein Schnappschuss des lokalen Pilot-Verlaufs.
* **Claims 3–4:** Beide prüfen lokalen, nicht versionierten Zustand (Store + Trust-Datei). Ihre Ergebnisse sind nur mit diesem Fixture reproduzierbar, nicht aus einem frischen Clone heraus.
* **Laufzeit-Varianz:** Geringfügige Zeitabweichungen im Millisekundenbereich resultieren aus CPU-Taktung und I/O-Latenzen des Zielrechners, berühren jedoch nicht die logische Äquivalenz.

---

## 3. Fazit für externe Auditoren

Alle fünf zentralen Kern-Behauptungen von QAL wurden auf dieser Maschine ohne Diskrepanzen nachvollzogen. Das ist eine **Selbstprüfung**: Der Bericht wird vom Runner selbst erzeugt, und die Claims 3–4 setzen den lokalen Pilot-Store samt Trust-Datei voraus (beide nicht versioniert).
Die Soll-Werte sind im Runner hinterlegt (`EXPECTED_PROOFS = 23`, fixierter Head); die gemessenen Ergebniswerte übernimmt der Bericht aus dem jeweiligen Lauf. Der Runner führt keine externen Telemetrie-Calls aus. Eine **unabhängige** Replikation liegt erst vor, wenn eine externe Person den Lauf auf ihrer Maschine wiederholt und ihren Bericht beilegt.
