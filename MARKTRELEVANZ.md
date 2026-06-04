# IGQK / qland — Marktrelevanz im "AI Factory / Tokens-pro-Watt"-Zeitalter

**Gutachter-Perspektive:** LLM-Inferenz-Effizienz × KI-Infrastrukturmarkt
**Gegenstand:** Der gesamte qlang-Cluster — `qland`, `neoqlang`, `QlangNeo`, `OQlang`, `A-2A-qlang` (alle dieselbe Codebasis-Linie) + die Python-Vorläufer `IGQK`, `BitNetDefinition`, `TSLM`, der Microsoft-Fork `BitNet` und der leere `TriLLM`.
**Datum:** 2026-06-04
**Charakter:** Markt-/Moat-Bewertung, keine Mathe-Korrektur. Brutal ehrlich, kein Marketing.

> **Befund-Konsistenz:** Diese Bewertung stützt sich auf ein Quellcode-Review von 9 Repos (Rust-Cluster parallel begutachtet). Alle Repos der qlang-Linie teilen denselben Quantisierungskern; Einzelnachweise mit Datei+Zeile siehe `EVALUATION.md`.

---

## Relevanz-Urteil (ein Satz)

**NIEDRIG** — der Quantisierungskern ist eine Commodity-Reimplementierung (Magnitude-Threshold-Ternär = TWN 2016 / BitNet-absmean) ohne lauffähigen Low-Bit-Kernel, ohne reale Modelle, ohne eine einzige Tokens/Watt- oder Perplexity-Zahl; das einzig potenziell Eigene — die information­geometrische *Auswahl*, welche Gewichte wie quantisiert werden — ist genau der **nicht implementierte** Teil und wäre selbst dann in ~3 Monaten von einem Lab reproduzierbar und kaum patentierbar.

---

## PHASE 1 — Was ist wirklich da (Substanz-Check)

### Der Cluster ist EIN Projekt, mehrfach umbenannt
`qland` → `neoqlang` → `QlangNeo` → `OQlang` → `A-2A-qlang` sind dieselbe Codebasis-Linie (Klon-URLs zeigen auf `qland`/`ABarisic/qlang`; hartkodierte Pfade `/home/mirkulix/neoqlang/...`). Single-Author (Aleksandar Barisic) + Claude-Agenten, AI-assistierte Mehrtages-Sprints (April 2026, 29–202 Commits je Repo). 80–112k LoC Rust — aber der Großteil ist **nicht** Quantisierung: DSL/VM, LLVM-JIT, Agent-Cockpit, Telegram-Bot, `qo/`-Companion-Stack, 3D-Graph-UI.

### Was die Methode tatsächlich tut
- **Quantisierung = Magnitude-Threshold-Ternär** (`igqk.rs:611` `max_abs*0.3`; `unified.rs:161` / `igqk_compress.rs:141` `std*0.5`; `bitnet_math.rs:39` absmean). Das ist algorithmisch BitNet-b1.58 / Ternary Weight Networks (2016).
- **Speicherbandbreite ODER Rechenkosten?** → **Keins von beidem, in der Praxis.** Ternärgewichte werden im Compute durchgängig als **`f32`** gehalten. Ein 2-bit-Packer existiert (`ternary_ops.rs:130`, 4 Gewichte/Byte), wird aber **nur beim Datei-Export** genutzt (`lm_export.rs:153`), nie im Rechenpfad. Der „add-basierte Kernel" `ternary_matvec` rechnet real `sum += x[k]*wi as f32` (also FP-Multiplikation) und ist **toter Test-only-Code** (`#[cfg(test)]`). Damit gibt es weder die Bandbreiten- (keine Sub-Byte-Gewichte im RAM) noch die Compute-Ersparnis (kein mult→add/LUT-GEMM).
- **Information­geometrie ist dekorativ:** Fisher-Metrik (`igqk.rs:185`) und Dichtematrix-Evolution werden berechnet, **steuern aber keine Quantisierungsentscheidung** — der Kompressionspfad nutzt eine feste Schwelle und ignoriert Fisher. `dim = min(d,16)` (`igqk.rs:484`) ⇒ die „Quanten"-Maschinerie berührt ohnehin nur 16 Gewichte.

### Lauffähiger Code / Benchmarks / reproduzierbare Zahlen?
- **Build:** `qlang-runtime` (der ML-Kern) kompiliert sauber (~6–13 s). Full-Workspace bricht ohne System-LLVM-18 ab.
- **Reale Modelle:** keine. Kein GGUF/safetensors/ONNX-Inferenzpfad; externe LLMs nur als API-Clients (Ollama/OpenAI/Anthropic). Eigene Modelle = Spielzeug (MNIST-MLP, „QuantumGPT 532K", Tiny-Mamba auf WikiText-2).
- **Zahlen:** **keine** Tokens/s, Tokens/Watt oder Perplexity vs. Baseline. MNIST-Ternär 84,6 % (eigenes Toy-Training); die IGQK-MNIST-„Validierung" meldet **10,40 % = Zufall** als „0 % Verlust, perfekt". „16×" ist ein theoretisches f32→2-bit-Byte-Verhältnis, keine Messung.

### Reifegrad (ehrlich)
**Idee + Prototyp** für die Quantisierung (messbar nur auf Toy-Tasks). Positiv: Der Autor ist selbstkritisch — Commits wie „eliminate fake marketing claims — honest audit pass" und nüchterne `QLANG-STATUS.md`-Disclaimer zeigen Selbstwahrnehmung.

---

## PHASE 2 — Marktrelevanz

### Welches wirtschaftliche Problem würde es lösen?
Der adressierte Schmerzpunkt (Inferenzkosten, HBM-Knappheit, Tokens/Watt, Edge/On-Prem, EU-Souveränität) ist **real und groß**. ABER: Low-Bit-Inferenz löst ihn nur über **gepackten Sub-Byte-Speicher + mult-freien Kernel** — und genau das fehlt hier. Ohne Kernel und ohne reale Modelle löst dieser Code **heute kein** wirtschaftliches Problem; er demonstriert ein bekanntes Prinzip auf Spielzeug.

### Wer hätte Schmerz + Zahlungsbereitschaft?
- **Hyperscaler / Neoclouds:** Nein — haben eigene Quant-Teams (GPTQ/AWQ/FP8/INT4 in TensorRT-LLM, vLLM).
- **Inferenz-Startups (Groq, Together, Fireworks):** Nein — Kernel-Engineering ist deren Kerngeschäft; sie liefern das selbst.
- **Enterprises mit On-Prem-LLMs / EU-Souveränität:** Potenziell — aber sie kaufen **fertige, benchmarkte** Stacks (llama.cpp/BitNet.cpp, vLLM), keinen Toy-Prototyp.
- **Edge/Embedded:** Der plausibelste Schmerz (1,58-bit für Mikrocontroller/NPUs) — aber exakt dort braucht man **echte Kernel + Hardware-Zahlen**, die hier nicht existieren.

### Software-Multiplikator oder mit Verfallsdatum?
Low-Bit *als Konzept* ist ein hardware-übergreifender Software-Multiplikator (dauerhaft wertvoll — BitNet zeigt es). Dieser **konkrete** Code ist es nicht: Er liefert den Multiplikator-Mechanismus (Packing + Add-Kernel) gerade nicht und läuft damit dem hinterher, was BitNet.cpp/llama.cpp/TensorRT bereits kostenlos in Silizium-naher Form liefern.

### Nüchterner Vergleich
| Verfahren | Liefert real | qlang-Cluster |
|---|---|---|
| **BitNet b1.58 / bitnet.cpp** | Gepackte 1,58-bit-Kernel, LLM-Skala, gemessen | absmean-Ternär als f32, kein Kernel, Toy |
| **GPTQ / AWQ** | Fehlerkompensation 2. Ordnung, INT4 auf echten LLMs | Magnitude-Schwelle, keine Kompensation |
| **QuIP#** | Inkohärenz + Gitter-Codebücher, SOTA-2-bit | nicht vorhanden |
| **llama.cpp Quant** | K-Quants, läuft überall, Community | nichts Vergleichbares |

→ **Verteidigbar eigen: nichts in der Effizienz.** Commodity, die andere kostenlos liefern: praktisch alles am Quantisierungskern.

---

## PHASE 3 — Wo ist der Burggraben (bzw. wo fehlt er)

### Der einzig denkbare Vorsprung …
… läge in der **information­geometrischen BEGRÜNDUNG der Auswahl**: Fisher-/krümmungs­gewichtete Entscheidung, *welche* Gewichte ternär dürfen und welche Präzision behalten (Minimierung von `(W−Q)ᵀ G (W−Q)` statt `‖W−Q‖²`). Das wäre eine echte, testbare Differenzierung gegenüber GPTQ (Hessian-basiert) und BitNet (uniform).

### … ist aber nicht der Burggraben, den dieses Repo hat
1. **Nicht implementiert:** Genau diese Auswahl fehlt; die Quantisierung ist uniform-magnitude.
2. **Schwer patentierbar:** Fisher-/natural-gradient-gewichtete Quantisierung grenzt an reichlich Prior Art (OBD/OBS, K-FAC, GPTQ ist faktisch eine Hessian-Approximation). Ein reiner „Fisher statt Hessian"-Twist ist kaum verteidigbar.
3. **In ~3 Monaten reproduzierbar:** Ein kompetentes Lab baut Fisher-gewichtete Quant-Selektion in einem Quartal nach.

### Die EINE Sache, die das Projekt verteidigbar macht?
**Im Effizienzmarkt: keine.** — Die einzige *real gebaute*, halbwegs differenzierte Komponente ist **QLMS**, das signierte binäre Agent-zu-Agent-Handover-Protokoll + Cockpit (`OQlang`/`A-2A-qlang`). Das ist solides Engineering, aber (a) ein **anderer Markt** (Agent-Orchestrierung, bereits hart umkämpft: MCP, Googles A2A) und (b) ebenfalls ohne Moat. Für die Tokens/Watt-Frage ist es irrelevant.

---

## PHASE 4 — Wie teilnehmen (Go-to-Market-Optionen)

> Vorbemerkung: Jede dieser Optionen ist **erst** glaubwürdig, wenn EIN hartes Beweisstück existiert (siehe „Beweisstück"). Ohne das ist jede GTM-Bewegung Marketing.

### a) Open-Source + Reputation/Beratung
- **Aufwand:** mittel. **Zeithorizont:** 1–3 Monate. **Risiko:** niedrig (Downside nur Zeit).
- **Pfad:** Methode + Benchmark sauber als Repo/Blog/Talk veröffentlichen; daraus Consulting-/Speaking-Mandate im EU-/Souveränitäts-/On-Prem-Inferenz-Umfeld ziehen (dort zählt Glaubwürdigkeit + EU-Narrativ).
- **Nächster Schritt:** Das Beweisstück (s. u.) erzeugen, als reproduzierbares Repo + ehrlichen Blogpost („Fisher-weighted ternary vs. GPTQ — was bringt's wirklich") publizieren.
- **Wichtigstes Beweisstück:** Fisher-gewichtete Ternär-Quant **head-to-head gegen BitNet/GPTQ** auf *einem* Standardmodell (z. B. Llama-3.2-1B oder Qwen2.5-0.5B), mit Perplexity + gemessenem RAM-Footprint. Ehrlichkeit ist hier der Asset.
- **Realismus:** Höchste Erfolgswahrscheinlichkeit. Monetarisiert Reputation, nicht IP.

### b) Produkt/Lizenz (Inferenz-Optimierungs-Layer als SDK/Service)
- **Aufwand:** hoch. **Zeithorizont:** 6–12+ Monate. **Risiko:** hoch.
- **Pfad:** Nur tragfähig mit echtem gepacktem Low-Bit-Kernel (CPU-SIMD/CUDA) + GGUF/safetensors-Loader + gemessenem Tokens/Watt-Vorteil, fokussiert auf **eine** Nische (z. B. EU-On-Prem-CPU-Inferenz oder Edge-NPU).
- **Nächster Schritt:** Erst Option (a) als Validierung; nur bei klarem Mess-Vorteil in den Kernel investieren.
- **Wichtigstes Beweisstück:** Gemessene **Tokens/Watt** und €/1M-Tokens vs. llama.cpp-Q4 auf realer Zielhardware — sonst kauft niemand.
- **Realismus:** Konkurriert frontal mit llama.cpp/vLLM/BitNet.cpp. Nur mit echtem Kernel-Vorsprung sinnvoll; aktuell weit entfernt.

### c) Akquise-/Talent-Pfad (IP/Demo für Inferenz-Startups oder Chip-/Cloud-Player)
- **Aufwand:** mittel. **Zeithorizont:** 2–4 Monate. **Risiko:** mittel.
- **Pfad:** Nicht das Repo „verkaufen" (kein Moat), sondern **Fähigkeit demonstrieren**: eine scharfe Demo + Benchmark als Türöffner für eine Rolle/Acqui-Hire bei Groq/Fireworks/Together/Mistral oder einem EU-Inferenz-Startup.
- **Nächster Schritt:** Dasselbe Beweisstück wie (a), aber als prägnante technische Demo + sauberes Write-up für gezielte Ansprache aufbereiten.
- **Wichtigstes Beweisstück:** Ein eleganter, korrekter **gepackter Ternär-Kernel** (auch CPU-SIMD) mit Mikro-Benchmark (GB/s, GFLOP-equiv, Energie) — zeigt Kernel-Engineering-Skill, was diese Firmen tatsächlich einstellen.
- **Realismus:** Pragmatisch. Der Wert ist der Mensch + Skill, nicht die IP.

---

## Abschluss

### 3 stärkste Punkte
1. **Reale Engineering-Substanz im Rust-Stack** (sauberer VM/JIT, Eigensolver, das signierte QLMS-Agent-Protokoll) — zeigt Bau-Fähigkeit.
2. **Selbstkorrigierende Ehrlichkeit** (Audit-Commits, nüchterne STATUS-Dateien) — gute Grundlage für glaubwürdige Veröffentlichung.
3. **Thematische Nähe zu einem echten Milliardenmarkt** (Tokens/Watt) + EU-Souveränitäts-Narrativ, das der Autor authentisch besetzen kann.

### 3 schwächste Punkte
1. **Kein Kernel, kein Sub-Byte-Compute** — der eigentliche Effizienz-Mechanismus fehlt; Ternär läuft als f32.
2. **Null reale Zahlen** — keine Tokens/Watt/Perplexity auf realem Modell; „Validierung" auf Zufallsniveau.
3. **Kein Moat** — Quantkern ist Commodity; die info-geometrische Differenzierung ist unimplementiert, schwer patentierbar und schnell reproduzierbar.

### Der EINE nächste Schritt (bestes Verhältnis Aufwand/Erkenntnis)
**Implementiere die information­geometrische Auswahl wirklich — und miss sie gegen GPTQ und BitNet auf genau einem kleinen Standard-LLM.** Konkret: Fisher-/Diagonal-Krümmungs-gewichtete Entscheidung, welche Gewichte ternär werden, dann Perplexity + gemessener RAM-Footprint von (i) deiner Methode, (ii) BitNet-absmean, (iii) GPTQ-INT4 auf z. B. Qwen2.5-0.5B. Dieser eine Test entscheidet binär über das gesamte Projekt: Schlägt deine geometrische Auswahl die Baselines messbar, hast du ein publizierbares, türöffnendes Ergebnis (Wege a/c). Tut sie es nicht, weißt du mit minimalem Aufwand, dass der Effizienz-Pfad tot ist — und kannst die echte Substanz (QLMS / Agent-Orchestrierung) als getrenntes Vorhaben verfolgen.

---
*Erstellt durch automatisiertes Multi-Repo-Code-Review (9 Repos). Alle Substanzaussagen sind über Datei+Zeile in den jeweiligen Repos nachprüfbar.*
