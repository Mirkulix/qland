# Betriebsfähiger Pilot: Financial Escrow & Multi-Sig Engine

**Datum:** 21. Oktober 2026  
**Ziel:** Nachweis der Betriebsfähigkeit (Milestone P1.5) nach formalen Assurance-Kriterien.

---

## 1. Übersicht der implementierten Betriebs-Bausteine

| Komponente | Implementierung | Sicherheits- & Assurance-Garantie |
|---|---|---|
| **Echter Store + Handler + Grants** | `qal serve handle --allow db:escrow,time --init setup` | Kein `serve` ins Leere. Echter Merkle-Projekt-Store mit 22 bewiesenen Z3-Kontrakten. |
| **API Gateway Sidecar** | [`gateway.py`](./gateway.py) | Erzwingt API-Key-Authentifizierung (`X-API-Key` / Bearer) und lehnt unautorisierte Requests mit HTTP 401 ab. |
| **Rate-Limiting** | Token Bucket Algorithmus (50 req/s, Burst 100) | Schutz vor DoS-Angriffen und Überlastung. Gibt HTTP 429 zurück. |
| **Observability & Metriken** | `/healthz` & `/metrics` Endpunkte | Public Health-Check mit automatischer Upstream-Reconciliation und Prometheus-Export. |
| **Atomic Backup & Restore** | [`backup_restore.py`](./backup_restore.py) | Point-in-Time SQLite-Snapshot ohne Schreibstopp. Prüft vor und nach Restore die Zero-Sum-Invariante. |
| **Lasttest & Race-Conditions** | [`load_test.py`](./load_test.py) | 50 parallele Payout-Worker. Exakt 3 Payouts genehmigt, 47 mit 422 abgewiesen, 0 Overdrafts, Zero-Sum erhalten. |
| **Audit-Verifikation** | `qal verify --head <hash>` | Unveränderliche Ed25519-Signaturkette mit 0 Warnungen fixiert. |

---

## 2. Testergebnisse & Abnahmekriterien

| Abnahmekriterium | Soll | Ist | Ergebnis |
|---|---|---|---|
| **Container/Service-Start mit frischem Store** | Initialisiert DB via `setup` idempotent | DB erfolgreich angelegt, WAL-Modus aktiviert |  Bestanden |
| **Auth-Enforcement (ohne Key / mit falschem Key)** | HTTP 401 Unauthorized | HTTP 401 Unauthorized |  Bestanden |
| **20/20 Testfälle über Gateway** | 20/20 bestanden (100 %) | 20/20 bestanden in 0.28 s |  Bestanden |
| **50 parallele Competing Payouts** | Keine Race-Condition, kein Minus-Saldo | Exakt 3 erfolgreich, 47 rejected (422), Saldo $\ge 0$ |  Bestanden |
| **Atomic Online Backup** | Zero-Sum Invariante verifiziert | $\text{total\_deposited} == \text{balances} + \text{escrow} + \text{disbursed}$ |  Bestanden |
| **Disaster Recovery (Restore nach DB-Verlust)** | Verlustfreie Wiederherstellung | Zero-Sum Invariante zu 100 % erhalten |  Bestanden |
| **Kryptographische Auditkette** | Pinned Head unbroken | Head `4be3934dda9f...` mit 0 Warnungen bestätigt |  Bestanden |

---

## 3. Architektur-Diagramm

```
           [ Client / Browser / Agent ]
                        │
                        ▼ (Port 8080)
        ┌────────────────────────────────┐
        │     Escrow API Gateway         │
        │  - Token-Bucket Rate Limiter   │
        │  - API-Key Authentifizierung   │
        │  - /healthz & /metrics Proxy   │
        └───────────────┬────────────────┘
                        │ (HTTP loopback 127.0.0.1:8090)
                        ▼
        ┌────────────────────────────────┐
        │       qal serve handle         │
        │  - 4 Wasmtime Guest Workers    │
        │  - 22 bewiesene Z3-Kontrakte   │
        │  - Grants: db:escrow, time     │
        │  - Host-Level Auto-Rollback    │
        └───────────────┬────────────────┘
                        │
                        ▼
             [ .qal/db/escrow.sqlite ]
```
