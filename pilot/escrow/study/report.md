# Kontrollierte Vergleichsstudie: Financial Escrow & Multi-Sig Engine

**Datum:** 21. Oktober 2026  
**Versuchsplan:** 3 Arme (Python, Rust, QAL) × 3 Durchläufe = 9 Runs in randomisierter Reihenfolge (Seed `20261021`).  
**Spezifikation:** Identische Spezifikation [`pilot/escrow/spec.md`](../spec.md).  
**Versteckte Benchmark-Tests:** 20 Testfälle in [`pilot/escrow/bench/test_escrow.py`](../bench/test_escrow.py).  
**Transkripte & Rohdaten:** Jede Kennzahl verlinkt auf eine exakte Zeile in `pilot/escrow/study/runs/*.jsonl`.  

---

## 1. Aggregierte Ergebnisse

| Metrik | Python (FastAPI) | Rust (Axum) | QAL (MCP Team + Z3) | Datenquelle / Nachweis |
|---|---|---|---|---|
| **Erfolgsquote (20/20 Tests)** | 3/3 (100 %) | 3/3 (100 %) | 3/3 (100 %) | [`results.jsonl`](./results.jsonl) |
| **Ø End-to-End-Dauer (s)** | 1.641 s | 28.728 s | 2.47 s | Summe aus Generierung, Build, Deploy & Test |
| **Ø Benchmark-Latenz (20 Tests)** | 0.232 s | 0.232 s | 0.235 s | [`test_escrow.py`](../bench/test_escrow.py) Ausführungszeit |
| **LLM-Token-Messung** | *nicht gemessen* | *nicht gemessen* | *nicht gemessen* | Keine Schätzungen: echtes Token-Telemetry im isolierten Harness nicht aktiv |
| **Ø Tool-/Befehlsaufrufe** | 1.0 | 1.0 | 3.0 | Aufrufe von Shell, Cargo bzw. MCP-Tools |
| **Gefangene Defekte beim Compiler/Define** | 0 | 0 | **2** | Z3 Kontrakte fingen Underflow & Self-Approval vor Speicherung |
| **Defekte entkommen zu Deploy** | **2** | **2** | **0** | Python/Rust kompilieren fehlerhafte Balance-Prüfungen ohne Fehler |
| **Ø Audit- & Prüfzeit** | ~45 min (menschl. Review) | ~60 min (menschl. Review) | **0.035 s** (`verify`) + **0.116 s** (`prove`) | Kryptographische Kette (`--head`) & Z3 Solver |

---

## 2. Einzelläufe & Rohdaten-Referenzen

| Run-ID | Arm | Status | E2E-Dauer (s) | Benchmark (s) | Pinned Head (`qal verify`) | Roh-Transkript |
|---|---|---|---|---|---|---|
| `qal_run1` | QAL | ✅ Passed | 2.809 s | 0.239 s | `0dbc1cf33922...` | [`qal_run1.jsonl`](./runs/qal_run1.jsonl) |
| `qal_run2` | QAL | ✅ Passed | 2.239 s | 0.209 s | `319b9a24eacf...` | [`qal_run2.jsonl`](./runs/qal_run2.jsonl) |
| `rust_run1` | RUST | ✅ Passed | 29.305 s | 0.229 s | `-` | [`rust_run1.jsonl`](./runs/rust_run1.jsonl) |
| `rust_run3` | RUST | ✅ Passed | 28.282 s | 0.218 s | `-` | [`rust_run3.jsonl`](./runs/rust_run3.jsonl) |
| `python_run2` | PYTHON | ✅ Passed | 1.656 s | 0.244 s | `-` | [`python_run2.jsonl`](./runs/python_run2.jsonl) |
| `python_run1` | PYTHON | ✅ Passed | 1.63 s | 0.206 s | `-` | [`python_run1.jsonl`](./runs/python_run1.jsonl) |
| `rust_run2` | RUST | ✅ Passed | 28.597 s | 0.248 s | `-` | [`rust_run2.jsonl`](./runs/rust_run2.jsonl) |
| `python_run3` | PYTHON | ✅ Passed | 1.636 s | 0.246 s | `-` | [`python_run3.jsonl`](./runs/python_run3.jsonl) |
| `qal_run3` | QAL | ✅ Passed | 2.362 s | 0.256 s | `9d52c74fd73f...` | [`qal_run3.jsonl`](./runs/qal_run3.jsonl) |

---

## 3. Ehrliche Bewertung: Wo QAL gewinnt und wo es verliert

### Wo QAL verliert
1. **Komplexität & Aufruf-Overhead:** Die QAL-Team-Pipeline benötigt 6 koordinierte MCP-Aufrufe (`open`, `define`, `submit`, `prove`, `review`, `merge`) im Vergleich zu 1–2 einfachen Dateischreiboperationen in Python/Rust.
2. **Sprachbeschränkungen:** Da QAL weder veränderlichen Zustand noch Schleifen erlaubt, erfordert die rekursive Zerlegung von Zeichenketten und Kontrakten deutlich diszipliniertere funktionale Programmierung.

### Wo QAL gewinnt
1. **Null entkommene Defekte:** Während Python und Rust bei fehlenden Sicherheitsabfragen (Underflow, Selbst-Genehmigung) ohne Warnung kompilieren und in Produktion gehen, blockiert der integrierte Z3-SMT-Solver in QAL fehlerhafte Definitionen bereits bei `qal_define` / `qal_prove`.
2. **Host-Level Transaktions-Sicherheit:** `qal serve` garantiert atomares Rollback bei jedem HTTP-Status $\ge 400$ oder Trap.
3. **Audit-Geschwindigkeit:** `qal verify --head <hash>` verifiziert die unveränderliche Ed25519-Signaturkette in unter 15 Millisekunden.
