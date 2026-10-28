# QAL — Status, Evidenz und Grenzen

**Stand:** 26.10.2026 · **Branch:** `Mac` · **Zweck:** Öffentlicher Bericht (Phase 9, Nr. 2)

QAL ist eine Programmiersprache für KI-Agenten. Ein Agent bearbeitet keine Dateien: Er sendet
Definitionen an einen geprüften, inhaltsadressierten Store (`.qal/store.db`) über MCP. Nur Code,
der parst und typ- sowie effektkorrekt ist, wird gespeichert; Verträge (`req`/`ens`) werden mit Z3
bewiesen, Tests laufen beim Ändern mit, und Programme laufen in einer WebAssembly-Sandbox mit
expliziten Effekten. Jede Antwort ist JSON, jeder Fehler eine Diagnose mit Code, Pfad und Hinweis.

## 1. Was heute belegt ist

| Eigenschaft | Beleg |
|---|---|
| **Geschlossener Kreis** | Ein `qal_define` prüft, beweist, testet, property-testet und verlinkt in einem Aufruf (Median 0,46 s auf 67 Funktionen) |
| **Nur Geprüftes existiert** | Typen, Effekte, Verträge, Tests; widerlegte Verträge werden nicht gespeichert (`E0801`) |
| **Inhaltsadressierung** | SHA-256 je Definition; Umbenennen ändert keinen Hash; Paketimport reproduziert Hashes |
| **Effekte als Berechtigungen** | `!io`, `!fs`, `!db`, `!net`, `!time`, `!rand`, `!model`; ohne Grant startet das Programm nicht |
| **Self-Hosting** | Compiler in QAL (`qalc.qal`), Fixpunkt Stage 1 == 2 == 3 byte-identisch; `qalc.qalpkg` im Korpus |
| **Formale Beweise** | Escrow-Kernel: **23/23** Obligations bewiesen, 0 refuted, 0 unknown |
| **Audit-Kette** | Ed25519-signierte Records, gepinnter Head, Truncation-Erkennung (`E0922`) |
| **Betrieb** | Escrow-Pilot: 20/20 Szenarien, 50 parallele Payouts, Backup/Restore, Auth, Rate-Limit, Zero-Sum |
| **Installer** | `./install.sh` baut, installiert, zeigt die MCP-Konfiguration; Smoke-Test ausgeführt |

### Reproduktion

```sh
./install.sh
python3 replication/replicate.py          # 5/5: Fixpunkt, 23 Proofs, Audit, Truncation, Zero-Sum
cargo test --workspace --release          # volle Suite
python3 tools/verify_docs_and_schema.py   # Doku == MCP-Schemas
```

Der Replikationsbericht ist eine **Selbstprüfung** auf der Entwicklungsmaschine; eine unabhängige
Drittmeinung liegt erst vor, wenn jemand anderes das Paket auf einer anderen Maschine ausführt.

## 2. Was gemessen wurde

### Agenten-Benchmark (20 Aufgaben, QAL vs. Rust)

- Beide Arme lösen **40/40** Läufe.
- QAL verarbeitet **1,42–1,92×** so viele Tokens wie Rust (Reruns 22.–24.10.: 1,92 / 1,42 / 1,52); im Living-Programm
  **1,32×** (Mittel der Runs 8–13), Run 14 bei 1,29×.
- Nach Tokens gewinnt QAL **5 von 20** Aufgaben (Rust 15/20; im älteren Rerun vom 22.10. waren es 0/20).
- Die These „QAL macht Agenten billiger“ ist für Aufgaben dieser Größe **nicht bestätigt**.
- Was QAL liefert, ist ein geprüfter, gespeicherter, hash-adressierter Codebestand – nicht weniger
  Tokens.
- **Lehrer-Baseline (DeepSeek, 27.10.):** 23 DeepSeek-Agenten lösten **23/23 Aufgaben** und
  **232/232 versteckte Testfälle** mit der QAL-CLI (`bench/teacher-2026-10-27/`). Ein starkes
  Modell kann QAL also schreiben; das lokale TinyLlama (7/23 check-valide, Adapter 2/23) ist
  der Engpass, nicht die Sprache.

### Living-Programm (12 Änderungsaufträge, dünne Tickets)

- **95 von 96** Stages grün (Runs 7–14; Run 14: 12/12, 1,29×).
- Der eine Fehlschlag war ein Agent, der den Test umschrieb, der die Regel festhielt; das
  Verhaltensgedächtnis `E0807` meldet diese Klasse jetzt im selben Define.
- Ein Overflow-Bug aus Run 1 ist durch exakte Multiply-Divide-Arithmetik (128-Bit-Zwischenprodukt)
  geschlossen.

### Mutationsstudie

- Bei **dünnen Tests**: QAL fängt **5 von 14** Mutanten, Rust **1 von 14** (Typinvarianten-Setup);
  der ältere A3-Lauf (2 vs. 1) ist überholt.
- Bei **gründlichen Tests**: **beide 14/14**.
- Der Assurance-Vorteil ist also **bedingt**: Er greift dort, wo Tests dünn sind. QAL ersetzt keine
  Tests.

### IGQK (separates Ökosystem)

- Echte 16×-Bit-Packing-Kompression: 1.000.000 Parameter 3,81 MB → 0,24 MB.
- Qualitätspreis: MNIST 94,10 % → 86,25 % (**−7,85 Prozentpunkte**); GPT-Test-Loss +3,88 %.
- Der IGQK-Optimizer lernt (MNIST 90,02 % vs. SGD 89,68 %), aber ein Quantenvorteil ist **nicht
  gemessen**.

## 3. Was QAL nicht ist

- **Kein allgemeiner Ersatz für Python, Rust, Go oder C++.** Keine Schleifen, keine Mutation,
  kein Nebenläufigkeitsmodell im Programm, 256 MiB Gast-Speicher, kleines Ökosystem.
- **Kein bewiesener Produktivitätsgewinn.** QAL ist zuverlässiger und nachvollziehbarer, aber
  teurer.
- **Keine vollständige formale Verifikation.** Der Prover deckt Int/Bool und seit Kurzem
  Record-Felder exakt ab; Str/List/Map/Float und Termination bleiben offen. Unbewiesene Verträge
  werden zur Laufzeit geprüft. **Ein bewiesenes `ens` heißt nicht trap-frei:** der Prover rechnet
  mit unbeschränkten Ints, während `*` auf 64 Bit überlaufen kann. Die Property-Tests melden das
  als `E0804`; wer Trap-Freiheit behauptet, braucht obere `req`-Schranken.
- **Kein fertiges Produkt für offene Mehr-Mandanten-Systeme.** Single-Node-SQLite, kein TLS im
  Gateway, externe Nutzer fehlen.

## 4. Offene Phasen

| Phase | Stand |
|---|---|
| 5 — Self-Hosting | **☑** |
| 6 — Kosten | **◐** 1,32× (Living), 1,42–1,92× (Benchmark-Reruns), 1,313× (92-Dispatch-Pilot); Ziel ≤ 1,2× nicht erreicht |
| 6b — Erfahrungsgedächtnis | **◐** gebaut, im Benchmark unbewiesen |
| 7 — `!model` | **◐** implementiert und getestet; Abnahme (vier Wochen Einsatz) offen |
| 8 — Modell lernt QAL | **◐** Korpus **5.300 Programme** (4.367 DeepSeek-generiert, alle `qal check`+`qal test`), **4.465 SFT-Beispiele**; voller LoRA-Lauf und Drei-Arm-Harness fehlen |
| 9 — Verbreitung | **◐** Installer, Bericht und MIT-Lizenz; externe Nutzer offen |

## 5. Fazit

QAL ist heute eine **spezialisierte, verifizierbare Kontrollschicht für agenten-gebaute Services**,
kein Allzweckwerkzeug. Sein Wert liegt in geprüftem, berechtigtem, nachvollziehbarem Code – nicht in
Geschwindigkeit. Wer diese Kontrolle braucht, findet hier eine technisch saubere, reproduzierbare
Basis. Wer schnell und billig programmieren will, nimmt weiterhin Rust oder Python.

Die ehrliche Kurzfassung: **QAL funktioniert, ist sicherer als der übliche Agenten-Stack und hat
einen schmalen, realen Nutzen. Es ist noch nicht fertig, und es ist nicht billiger.**
