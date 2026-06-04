# IGQK / qland — Technisches Gutachten

**Gutachter-Perspektive:** Informationsgeometrie, numerische Optimierung, LLM-Inferenz-Effizienz
**Gegenstand:** `Mirkulix/IGQK` (Feb 2026, nur Python) und `Mirkulix/qland` (Apr 2026, Superset: Python `igqk/` + Rust `qlang/`, ~31k LoC)
**Datum:** 2026-06-04
**Methodik:** Quellcode-Review gegen die behauptete Theorie. Jede Bewertung mit Datei + Zeile belegt.

> **Hinweis zur Gültigkeit der Referenzen:** Findings, die auf `qlang/...` (Rust) verweisen, existieren nur in `qland`. Findings, die auf `igqk/...` (Python) verweisen, sind in **beiden** Repos byte-identisch. `qland` ist die spätere, umfassendere Variante und damit der Maßstab.

---

## Zusammenfassung (TL;DR)

Was das README verspricht — eine information­geometrisch *begründete* Quantisierungsentscheidung, getragen von Quantengradientenfluss mit Konvergenz- und Kompressionsschranken — ist im Code **nicht realisiert**. Was der Code *tatsächlich* tut, ist:

1. **Magnitude-Threshold-Ternarisierung** (`{-1,0,+1}` per fester Schwelle) — algorithmisch identisch zu *Ternary Weight Networks* (Li & Liu, 2016).
2. Dazu Standard-Sparse/Low-Rank-Projektion (Top-k bzw. Truncated SVD).
3. Ein dekoratives Dichtematrix-/„Quanten"-Modul, das (a) mathematisch unsauber ist und (b) **gar nicht in den Trainingspfad eingebunden** ist.

Die zentrale, einzig potenziell neue Idee — *die Metrik soll die Quantisierungsentscheidung steuern* — ist genau der Teil, der **fehlt**: Die Projektion ignoriert die Fisher-Metrik vollständig. Damit gibt es keinen messbaren Vorsprung gegenüber dem Stand der Technik (BitNet, GPTQ, AWQ), und ökonomisch (Tokens/Watt) liefert die Implementierung **null** realen Hebel, weil ternäre Gewichte als `f32` gespeichert werden und kein Low-Bit-Kernel existiert.

Erschwerend: Die „Validierung" behauptet *„0 % Genauigkeitsverlust, 16×, perfekt"*, während das Modell auf MNIST bei **10,40 %** liegt — exakt Zufallsniveau bei 10 Klassen. Es wird also ein Modell komprimiert, das nie gelernt hat.

Provenienz: Das gesamte `qland`/`qlang`-Projekt wurde laut Git-Historie (`git log`, 54 Commits) **von Claude-Agenten in ~30 Stunden am 2.–3. April 2026** generiert. Das erklärt das Muster: viel plausibel aussehendes Gerüst, dekorative Mathematik, aspirationale Doku.

---

## PHASE 1 — Bestandsaufnahme (deskriptiv)

### Verzeichnis / zentrale Module

**Eigentliche Mathematik (Python, `igqk/`):**
- `igqk/core/manifold.py` — Statistische Mannigfaltigkeit, Fisher-Metrik, „Riemann-Distanz", Geodäten
- `igqk/core/quantum_state.py` — Dichtematrix ρ (Eigenzerlegung, Entropie)
- `igqk/core/measurement.py` — Messoperatoren / Kollaps auf diskrete Gewichte
- `igqk/core/evolution.py` — Quantengradientenfluss (Python-Variante)
- `igqk/compression/projection.py` — „Optimale Projektion" Π: M → N
- `igqk/integration/pytorch.py` — Trainer-/Optimizer-Anbindung

**Eigentliche Mathematik (Rust, `qlang/crates/qlang-runtime/src/`, nur `qland`):**
- `igqk.rs` (1110 Z.) — „IGQK Math Engine": Fisher, Laplace-Beltrami, Quantenfluss, Kompression, Konvergenz
- `fisher.rs` (469 Z.) — empirische Fisher-Metrik + natürlicher Gradient
- `theorems.rs` (482 Z.) — numerische „Verifikation" der Theoreme 5.1/5.2/5.3
- `quantum_flow.rs` (605 Z.), `qlang-core/src/quantum.rs` — Dichtematrix-Infrastruktur
- `training.rs` (331 Z.) — der **tatsächliche** Trainings-Loop

**Theorie-Quelle:** `Entwicklung_der_IGQK-Theorie_Mathematische_Details.pdf` (Behauptungen), `README.md` (Zusammenfassung).

### (a) Metrik/Geometrie auf dem Parameterraum
Behauptet: Fisher-Informationsmetrik `g_ij = E[∂_i log p · ∂_j log p]` auf statistischer Mannigfaltigkeit (`manifold.py:7-14`, `README.md:9`).
Tatsächlich: Fisher *wird berechnet* (`manifold.py:57`, `fisher.rs:78`, `igqk.rs:180`), aber als Geometrie nirgends *verwendet* (siehe Phase 2). Distanzen, Geodäten und Projektion sind euklidisch/flach.

### (b) Zielfunktion der Quantisierung
Behauptet: Quantengradientenfluss `dρ/dt = -i[H,ρ] - γ{G⁻¹∇L, ρ}` (`README.md:22`, `igqk.rs:265-285`).
Tatsächlich: Verzerrung wird euklidisch gemessen `D = ‖W - Π(W)‖²` (`projection.py:133`, `igqk.rs:649-657`). Die Projektion minimiert nichts Riemannsches; sie ist eine feste Schwellwert-Regel. Der „Fluss" ist nicht ans Training gekoppelt.

### (c) Quantisierungsschema
Ternär `{-1,0,+1}` per Magnitude-Threshold; zusätzlich Low-Rank (Truncated SVD) und Sparse (Top-k).
Schwellen sind **inkonsistent** über die Codebasis: `0.3·max|w|` (`igqk.rs:607`), `0.5·std` (`projection.py:253`, `measurement.py:90`).

### (d) Sätze/Schranken
- **Theorem 5.1 (Konvergenz):** `E_ρ*[L] ≤ min L + O(ℏ)` (`README.md:189`, `theorems.rs:85`)
- **Theorem 5.2 (Kompressionsschranke):** `D ≥ (n-k)/(2β)·ln(1+β·σ²_min)` (`README.md:199`, `theorems.rs:156`)
- **Theorem 5.3 (Verschränkung/Generalisierung):** `E_gen ≤ E_train + √(I(A:B)/n)` (`README.md:213`, `theorems.rs:234`)

### Was die Methode TATSÄCHLICH tut
Post-hoc Magnitude-Threshold-Quantisierung (ternär/sparse/low-rank) eines klassisch (per Finite-Differenzen-SGD) trainierten Modells, garniert mit einem nicht angebundenen Dichtematrix-Modul. **Keine** information­geometrische Entscheidung, **kein** Quanteneffekt, **kein** quantisierungs-bewusstes Training.

---

## PHASE 2 — Korrektheitsprüfung

### 2.1 Die Informationsgeometrie ist nicht implementiert (zentraler Mangel)
`manifold.py:117-152` — `riemannian_distance(...)` gibt für **alle** Metriken (`hellinger`, `kl`, `euclidean`) denselben euklidischen `‖θ1-θ2‖` zurück. Die Hellinger-Distanz wird explizit „approximiert" durch die euklidische (`manifold.py:140-144`).
`manifold.py:172-192` — Geodäte = Gerade (TODO Christoffel). `manifold.py:194-212` — Exp-Map = `θ+t·v`. `manifold.py:214-233` — Paralleltransport = Identität. `manifold.py:154-170` — Tangentialprojektion = Identität.
→ Die gesamte Riemann-Struktur kollabiert zu flachem ℝⁿ. **Die Metrik fließt nirgends in die Quantisierung** (`grep` über `igqk/compression/` und `measurement.py` nach `fisher`/`riemannian_distance` ⇒ **leer**). Damit ist die Projektion `_project_ternary` (`projection.py:249-259`) reines euklidisches Schwellwertverfahren = Ternary Weight Networks.

### 2.2 Das „Quanten"-Modul ist mathematisch unsauber und totes Gerüst
- `igqk.rs:299-311` — die Evolutionsgleichung lässt das `i` weg und „behält den Kommutator-Beitrag direkt". `-i[H,ρ]` für reelle, symmetrische H, ρ ist **nicht** reell-symmetrisch; das Weglassen von `i` ergibt weder unitäre Evolution noch eine valide GKSL/Lindblad-Mastergleichung. Es ist eine willkürliche reelle Störung.
- `igqk.rs:478` — die Hilbertraum-Dimension wird auf `min(d,16)` gekappt; `igqk.rs:540-542` extrahiert neue Gewichte aus dem dominanten Eigenvektor. **Für jedes Netz mit > 16 Parametern werden nur die ersten 16 Gewichte aktualisiert.** Der „Trainer" kann real kein Netz trainieren.
- `igqk.rs:244-259` — „Laplace-Beltrami" `H_ii = -1/g_ii` ist eine reine Diagonal-Inverse; kein zweiter Differentialoperator, keine Christoffel-Symbole. Die Bezeichnung ist irreführend.
- **Tote Codepfade:** `grep` nach `igqk_training_step | evolve_density_matrix | compress_with_bound | laplace_beltrami` außerhalb `igqk.rs` ⇒ **kein Treffer** in `training.rs`/`vm.rs`/`mnist.rs`. Die „IGQK Math Engine" wird ausschließlich von ihren eigenen Unit-Tests aufgerufen. Das **echte** Training (`training.rs:143-193`) ist Finite-Differenzen-SGD.

### 2.3 Numerische Umsetzung
- **Python-Fisher ist schlicht falsch:** `manifold.py:106-109` akkumuliert `torch.outer(grad, grad)`, wobei `grad` der Gradient des **Batch-Loss** ist. Das ist `(E[g])(E[g])ᵀ` (Rang 1), nicht die empirische Fisher `E[g·gᵀ]`. Zudem wird `fisher = torch.zeros(dim, dim)` dicht allokiert (`manifold.py:77`) — für reale Netze (Millionen²) nicht handhabbar.
- **Rust-Fisher ist besser, aber Spielzeug-Skala:** `fisher.rs:78-120` bildet korrekt Per-Output-Außenprodukte mit Dämpfung. Aber: Gradienten per **zentralen Finite-Differenzen** (`fisher.rs:38-68`) ⇒ O(n) Forward-Pässe je Sample; dichte n×n-Matrix ⇒ O(n²) Speicher. Nur für winzige n nutzbar. Außerdem werden rohe Outputs als Wahrscheinlichkeiten behandelt (`fisher.rs:107`, `p = output[k]`) ohne Softmax-Normierung — „log p" ist damit fragwürdig.
- **Kein Straight-Through-Estimator:** Ternarisierung wird post-hoc auf die Parameter angewandt (`projection.py:63-84`). Es gibt keinen STE, keinen Gradientendurchgriff durch die Quantisierung ⇒ **kein** quantisierungs-bewusstes Training (anders als von „rigoros" suggeriert).
- **Latenter Bug:** `projection.py:105` nutzt `np.sqrt`, aber `numpy` ist in `projection.py` nicht importiert ⇒ der Low-Rank-Pfad (`compression_ratio('lowrank')`) wirft `NameError`.
- `eval`-Inkonsistenz: ternäre Schwellen weichen zwischen Rust und Python ab (s. Phase 1c).

### 2.4 Die „Theoreme" beweisen nichts
- **Th. 5.2 zeigt in die falsche Richtung:** Es ist eine **untere** Schranke der Verzerrung. Für eine Qualitätsgarantie bräuchte man eine **obere** Schranke. `verify_compression` (`theorems.rs:173-223`) prüft nur `distortion ≥ bound`; da `bound → 0` für `σ²_min → 0` (`theorems.rs:156-161`), ist das nahezu trivial erfüllt und ohne Aussagekraft. Die Formel ist zudem im Kern eine Gauß-Rate-Distortion-Aussage, hier als eigener Satz verkauft.
- **Th. 5.1 ist zirkulär:** `min_loss` wird aus der Loss-Historie selbst genommen; „konvergiert" = geglätteter Loss innerhalb `c·ℏ` (`theorems.rs:85-143`). Das sagt nichts über das *globale* Optimum aus.
- **Th. 5.3 ist nicht „quantum":** Die „Quanten-Mutual-Information" ist `-0.5·ln(1-r²)` (`theorems.rs:248-282`), die **Gauß-MI aus dem Pearson-Korrelationskoeffizienten**. Nichts Quantenmechanisches; es ist eine umbenannte Input/Output-MI-Generalisierungsschranke (Xu & Raginsky, 2017).

### 2.5 Versteckte Annahmen
Diagonal-Approximation der Fisher (`igqk.rs:493-518`, `laplace_beltrami` `igqk.rs:244`), reeller Hilbertraum statt komplexem, Unabhängigkeits-/Linearisierungsannahmen in der „Distanz" (`manifold.py:142`), Gauß-Annahme implizit in Th. 5.2/5.3.

### 2.6 Empirische Claims sind irreführend
`IGQK_Complete_Package/VALIDATION_REPORT.md:114-116` und `IGQK_TEST_REPORT.md:141` — **beste MNIST-Genauigkeit 10,40 %** (Zufall bei 10 Klassen), gemeldet als „nach Kompression 10,40 %, Verlust 0,00 % 🎯". „0 % Verlust" ist trivial, wenn man ein Modell auf Zufallsniveau komprimiert. `VALIDATION_REPORT.md:17` („16×, 320 % Übertreffen") und `:18` („0,65 % Verlust") sind Marketing ohne reale Lernkurve.

---

## PHASE 3 — Einordnung in den Stand der Technik

| Verfahren | Was es real tut | IGQK/qland dagegen |
|---|---|---|
| **BitNet b1.58** | Trainiert ternär *from scratch* mit STE, LLM-Skala, Parität zu fp16 | Kein STE, kein Skalierungsnachweis, Modell auf Zufallsniveau |
| **GPTQ / AWQ** | Layerweise Fehlerkompensation 2. Ordnung (inverse Hessematrix / aktivierungs­bewusst) | Reine Magnitude-Schwelle, **keine** Fehlerkompensation ⇒ strikt schwächer als GPTQ |
| **QuIP#** | Inkohärenz-Vorverarbeitung + Gitter-Codebücher | Nicht ansatzweise vorhanden |
| **Natural Gradient / K-FAC (Amari)** | Reale, nützliche Krümmungsnutzung | Nur Diagonal-Approx, *nicht* ins Training eingebunden |
| **Rate-Distortion** | Untere Schranken der Quellcodierung | Th. 5.2 ist eine schwächere Umformulierung davon |

**Was ist wirklich neu?** Demonstrierbar: nichts. Der einzige potenziell neue Beitrag — *die information­geometrische Begründung der Quantisierungsentscheidung* (z. B. ternäre Zuweisung, die `(W-Q)ᵀ G (W-Q)` minimiert statt `‖W-Q‖²`) — ist exakt der **nicht implementierte** Teil. Es bleibt eine Reimplementierung von TWN plus nicht-funktionalem Gerüst. Der mögliche Vorsprung läge in der *Begründung*, nicht in der Implementierung — und die Begründung ist im Code abwesend.

---

## PHASE 4 — Ökonomische Bewertung (AI Factory / Tokens-pro-Watt)

**Speicherbandbreite vs. Rechenkosten:** Ternär adressiert *prinzipiell* beides — Gewichte 32 bit → ~1,58 bit (~16–20× kleiner, also **Bandbreite**) und Mult→Add (**Rechenkosten**), *sofern* ein ternärer Kernel existiert.

**In dieser Implementierung jedoch:** Ternäre Gewichte werden als `Vec<f32>` gespeichert (`igqk.rs:603`, `projection.py:255`). Es gibt **keinen** gepackten Low-Bit-Speicher und **keinen** add-basierten Matmul-Kernel. ⇒ **Null** reale Speicher- oder Recheneinsparung. Die „16×" sind eine reine Bit-Zähl-Theorie (`projection.py:96-99`). Als Implementierung adressiert die Methode damit **weder** den Bandbreiten- **noch** den Compute-Engpass.

**Software-Multiplikator oder Hardware-Konkurrenz?** Low-Bit-Inferenz *ist* konzeptionell ein wertvoller, hardware-übergreifender Software-Multiplikator (BitNet zeigt das). Aber genau dieser Wert wird bereits von BitNet/GPTQ/AWQ + existierenden Kerneln (z. B. `bitnet.cpp`) eingefahren. IGQK fügt nichts hinzu und liefert keine Kernel. Gegenüber dem, was kommende Hardware (Vera Rubin etc.) plus etablierte Low-Bit-Stacks ohnehin liefern, gibt es hier keinen Differenzierungspunkt.

**Realistischer Hebel:** Aus diesem Repo heute: **keiner**. Der einzige Pfad zu Wert ist die generische Ternär-Inferenz, die anderswo bereits existiert.

---

## Fazit

### 3 stärkste Punkte
1. **Solide Software-Substanz (Rust):** `qlang` (~31k LoC) ist sauber strukturiert, mit korrektem Jacobi-Eigensolver (`igqk.rs:362`), Gauß-Jordan-Inversion (`igqk.rs:75`) und Hunderten Unit-Tests — gutes Gerüst.
2. **Brauchbarer Baustein:** `fisher.rs:78` ist eine im Kern korrekte empirische Fisher mit Dämpfung — als Komponente wiederverwendbar.
3. **Legitime Forschungsrichtung:** Quantisierung als Projektion auf eine Untermannigfaltigkeit unter einer Informationsmetrik zu fassen, ist eine interessante, prinzipiell tragfähige Idee — *falls* sie tatsächlich implementiert würde.

### 3 größte Risiken
1. **Theorie–Code-Lücke ist nahezu total:** Die zentrale information­geometrische Behauptung steht nicht im Code; das Quantenmodul ist tot und mathematisch invalide (`i` weggelassen, dim ≤ 16). Wer auf Basis des README investiert/baut, wird in die Irre geführt.
2. **Irreführende Empirie:** „0 % Verlust, 16×, perfekt" bei einem Modell auf Zufallsniveau (10,40 % MNIST). Das ist das schwerwiegendste Integritätsproblem.
3. **Keine Skalierbarkeit, keine Kernel:** Dichte Fisher + Finite-Differenzen-Gradienten + f32-gespeicherte Ternärgewichte ⇒ weder auf realen Modellen lauffähig noch mit realer Einsparung.

### 3 wichtigste nächste Schritte (um einen echten ökonomischen Hebel zu erzeugen)
1. **Geometrie tatsächlich entscheiden lassen:** Fisher-gewichtete Quantisierung implementieren — ternäre Zuweisung, die `(W-Q)ᵀ G (W-Q)` (Diagonal-Fisher / K-FAC) minimiert — und **direkt gegen GPTQ/AWQ** an einem realen Modell messen. Das ist der einzige neue, testbare Claim.
2. **Echtes QAT mit STE in Skala:** Ternär from-scratch mit Straight-Through trainieren, BitNet-b1.58-Parität auf einem kleinen LLM reproduzieren, **reale** Genauigkeit berichten (nicht Zufall).
3. **Ternär-Inferenz-Kernel liefern:** Gepackter 1,58-bit-Speicher + add-basierter Matmul (CPU/GPU), dann **gemessene** Tokens/Watt und Bandbreite vs. fp16/INT8 — sonst bleibt „16×" Fiktion.

---
*Erstellt durch automatisiertes Code-Review (Quellcode gegen behauptete Theorie). Alle Aussagen sind über Datei + Zeile nachprüfbar.*
