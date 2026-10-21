# 07 — Verifizierte Referenzbeispiele (Arithmetische Kernels)
> **Formal bewiesene Rechen-Kernels mit Z3-Verifikation, Invarianten und kryptografischen Hashes**

---

> [!NOTE] Einordnung des Geltungsbereichs
> Diese Beispiele sind reine mathematische Kernels (pure functions). Sie demonstrieren, wie Z3 lineare Invarianten (Erhaltungssätze, Schranken) statisch garantiert und bei Fehlern Gegenbeispiele generiert. Sie stellen keine kompletten Produktionssysteme mit Persistenz, Netzwerkprotokollen oder Authentifizierung dar, sondern isolierte, formale Verifikations-Benchmarks.

---

## Fallstudie 1: Formaler Verifikations-Kernel — Treuhand-Wert-Erhaltung (Escrow & Settlement)

Dieser Algorithmen-Kernel demonstriert die statische Verifikation arithmetischer Erhaltungssätze in Presburger QF_LIA:

### 1. Quelltext & Kontrakte
```qal
fn escrow_fee(amount: Int, fee_rate_bps: Int) -> Int
  req amount >= 0 && fee_rate_bps >= 0 && fee_rate_bps <= 1000
  ens r >= 0 && r <= amount
= (amount * fee_rate_bps) / 10000

fn escrow_net(amount: Int, fee: Int) -> Int
  req amount >= 0 && fee >= 0 && fee <= amount
  ens r >= 0 && r + fee == amount
= amount - fee

fn debit_balance(balance: Int, amount: Int) -> Int
  req balance >= amount && amount >= 0
  ens r >= 0 && r == balance - amount
= balance - amount

fn credit_balance(balance: Int, amount: Int) -> Int
  req balance >= 0 && amount >= 0
  ens r >= 0 && r == balance + amount
= balance + amount

fn verify_conservation(s_before: Int, r_before: Int, s_after: Int, r_after: Int, fee: Int) -> Bool
  req s_before >= 0 && r_before >= 0 && s_after >= 0 && r_after >= 0 && fee >= 0
  ens r == (s_before + r_before == s_after + r_after + fee)
= s_before + r_before == s_after + r_after + fee

fn settle_total(s_before: Int, r_before: Int, amount: Int, fee: Int) -> Int
  req s_before >= amount && amount >= fee && fee >= 0 && r_before >= 0
  ens r == s_before + r_before
= let s_after = s_before - amount in
  let r_after = r_before + (amount - fee) in
  s_after + r_after + fee
```

### 2. Z3-Beweisstatus
* `escrow_fee`: **proven** (Gebühr ist positiv und nach oben durch den Betrag beschränkt).
* `escrow_net`: **proven** (Nettoauszahlung + Gebühr entspricht exakt dem Bruttobetrag).
* `debit_balance`: **proven** (Überziehungsschutz: Saldo bleibt >= 0).
* `settle_total`: **proven** (Gesamtsumme vor und nach Transfer bleibt byte-identisch erhalten).

### 3. Kanonische SHA-256 Hashes
* `escrow_fee`: `82d44af65250382e8f94b56b00232bb5764a049e7b36d3c44c773d9163fab060`
* `escrow_net`: `54f55afe050cce1458e70ec1e73c7e27250170f2bbcaabddf622a82516ea436f`
* `debit_balance`: `cb883366f4b61fe8bfc0710c78d9ed41d73830211f0f5d2c63c3bcc925fc5cca`
* `settle_total`: `de13b17844949a50a9b3fa85f4ea24d3d2b31378b74c5fa75d65a6e60d1663d6`

---

## Fallstudie 2: Formaler Verifikations-Kernel — Quorum- & Stimmenerhaltungs-Arithmetik (Governance Voting)

Dieser Algorithmen-Kernel demonstriert die Verifikation von Schranken, Quorum-Bedingungen und automatischer Gegenbeispiel-Widerlegung (kein vollwertiges Voting-Backend, sondern die mathematische Kern-Arithmetik):

### 1. Quelltext & Kontrakte
```qal
fn vote_inc(count: Int) -> Int
  req count >= 0
  ens r >= 0 && r == count + 1
= count + 1

fn check_quorum(total_votes: Int, quorum: Int) -> Bool
  req total_votes >= 0 && quorum >= 0
  ens r == (total_votes >= quorum)
= total_votes >= quorum

fn verify_tally(yes: Int, no: Int, abstain: Int, total: Int) -> Bool
  req yes >= 0 && no >= 0 && abstain >= 0 && total >= 0
  ens r == (yes + no + abstain == total)
= yes + no + abstain == total

fn determine_outcome(yes: Int, no: Int) -> Bool
  req yes >= 0 && no >= 0
  ens r == (yes > no)
= yes > no

fn vote_percentage(yes: Int, total: Int) -> Int
  req yes >= 0 && total >= 0 && yes <= total
  ens r >= 0 && r <= 100
= if total == 0 then 0 else (yes * 100) / total
```

### 2. Autonome Fehlerabwehr (Live Execution)
Ein unvorsichtiger Versuch, `vote_percentage` ohne Schranke `yes <= total` zu definieren, wurde vor der Speicherung von Z3 widerlegt:
* **Diagnose:** `E0801 contract_refuted`
* **Gegenbeispiel:** `{"total": 1, "yes": 2}`
* **Reaktion:** Automatische Ergänzung von `yes <= total` und Division-durch-Null-Abfrage. Danach sofortiger Beweis als `status: proven`.

### 3. Kanonischer Hash
* `vote_percentage`: `63624e256a53f557efbbfa33193ba9162cc95914c14e7edeeb8aed70703546d9`
* `determine_outcome`: `0f023ac25ab50fa12e45fe9f5bc21c19a38a3683a1930a3de1206c713741868f`
