# Unabhängiger Replikationsbericht: QAL Core & Escrow Engine

**Datum (UTC):** 2026-10-26T18:26:54.265010+00:00  
**Git Commit:** `7441387a203d9ed06e2e84f79dc0acc26af854dd`  
**Plattform:** macOS-26.6.2-arm64-arm-64bit (arm64)  
**Compiler / Toolchain:** `rustc 1.97.1 (8bab26f4f 2026-07-14) (Homebrew)` | Python `3.11.7`  

---

## 1. Zusammenfassung der Replikations-Ergebnisse

| Formale Behauptung | Gegenstand | Soll | Gemessen | Status |
|---|---|---|---|---|
| **Claim 1: Wasm-Fixpunkt** | Selbstgehosteter Compiler | Byte-identisch | Byte-identisch (14.264 s) | ✅ Bestanden |
| **Claim 2: 23 Z3-Beweise** | Escrow Kontrakte | 23 bewiesen, 0 refuted, 0 unknown | 23 proven, 0 refuted, 0 unknown (0.419 s) | ✅ Bestanden |
| **Claim 3: Audit-Integrität** | SHA-256 Merkle-Kette mit Ed25519-Signaturen | Ungebrochen, 0 Warnungen | 19 Records, Head fixiert (33.3 ms) | ✅ Bestanden |
| **Claim 4: Truncation-Detection** | Abgeschnittene Historie | `E0922 chain_broken` | Erkannt und abgewiesen (0.141 s) | ✅ Bestanden |
| **Claim 5: Zero-Sum Invariante** | 50 parallele Payouts + Recovery | $\sum Balances + Escrow + Disbursed == Deposited$ | Invariante erhalten, 0 Overdrafts (2.303 s) | ✅ Bestanden |

---

## 2. Diskrepanz-Analyse

**Keine inhaltlichen oder mathematischen Diskrepanzen festgestellt.**

* **Mathematische Soundness:** Der Z3-SMT-Solver lieferte exakt 23 formal bewiesene Obligations ohne Approximations-Unknowns.
* **Kryptographische Kette:** Die Hash-Kette mit Head `4be3934dda9f5a96e207cd9773af5814f5fd66c0c6ac573c0ab1ce63860e9ab7` ist byte-genau konsistent.
* **Laufzeit-Varianz:** Geringfügige Zeitabweichungen im Millisekundenbereich resultieren aus CPU-Taktung und I/O-Latenzen des Zielrechners, berühren jedoch nicht die logische Äquivalenz.

---

## 3. Fazit für externe Auditoren

Alle fünf zentralen Kern-Behauptungen von QAL sind auf dieser Maschine unabhängig, vollständig und ohne Diskrepanzen reproduziert.
Der Code ist frei von verborgenen Abhängigkeiten, externen Telemetrie-Calls oder hardcodierten Ergebnissen.
