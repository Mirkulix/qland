# 🚀 IGQK - Information Geometric Quantum Compression

> **Research prototype: Quantum Gradient Flow for neural-network compression**

[![Tests](https://img.shields.io/badge/tests-21%20%2F%2011%20%2F%2012-brightgreen)]()
[![Compression](https://img.shields.io/badge/ternary-15.6x%20file%20%2F%2016.0x%20payload-blue)]()
[![Accuracy](https://img.shields.io/badge/accuracy%20loss-3.8--7.9%20pp-yellow)]()
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)]()

---

## 🎯 Was ist IGQK?

IGQK ist ein **Forschungsprototyp** (research prototype) für Neural Network Compression, der:

- ✅ **Ternäre 2-Bit-Kompression** erreicht: 15,6× auf der gemessenen Checkpoint-Datei (1,53 MB → 0,099 MB); 16,0× auf den reinen ternären Nutzdaten (1 Mio. Parameter: 3,81 MB → 0,24 MB)
- ⚠️ **Genauigkeitsverlust ist modellabhängig**: −3,77 Pp. (97,45 % → 93,68 %, MLP 3072-128-64-10, MNIST, 3 Epochen, Adam) bzw. −7,85 Pp. (94,10 % → 86,25 %, MNISTNet, anderes Netz und andere Trainingsmenge). **Nicht 0,65 %.**
- ✅ **Quantenmechanik als Optimierungsansatz** (Quantum Gradient Flow)
- ✅ **Mathematisch formuliert** (Konvergenz- und Rate-Distortion-Theoreme)
- ✅ **Tests**: 21 (igqk_v4) + 11 (igqk) + 12 (SaaS-Backend) bestanden; es gibt keine Coverage-Messung, daher nicht „100 % getestet“

---

## ⚡ Schnellstart

### 🎨 **Web-UI starten** (Empfohlen!)

```bash
cd igqk
START_UI.bat         # Windows
# oder
python ui_dashboard.py
```

Öffnet automatisch im Browser: http://localhost:7860

### 🧪 **Alle Tests ausführen**

```bash
cd igqk
START_ALL.bat        # Alle Tests automatisch
# oder
python START_SYSTEM.py  # Interaktives Menü
```

### 💻 **In eigenem Code nutzen**

```python
from igqk import IGQKOptimizer

# Ersetze Adam/SGD durch IGQK
optimizer = IGQKOptimizer(
    model.parameters(),
    lr=0.01,
    hbar=0.1,  # Quantum uncertainty
    gamma=0.01  # Damping
)

# Training wie gewohnt
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Komprimiere Modell (15,6× Dateigröße, 16,0× ternäre Nutzdaten)
optimizer.compress(model)
```

---

## 📦 Was ist enthalten?

### 1. **Kernbibliothek** (`igqk/`)
- `core/` - Quantum States, QGF
- `manifolds/` - Fisher-Metrik, Statistische Mannigfaltigkeiten
- `compression/` - Ternary, Binary, Sparse, Low-Rank
- `optimizers/` - IGQKOptimizer, Scheduler

### 2. **Web-UI** (`ui_dashboard.py`) ⭐ NEU!
- Interaktive Parameter-Kontrolle
- Live-Visualisierungen (Loss, Accuracy, Entropy, Purity)
- Kompressions-Charts
- Einfache Bedienung

### 3. **Test-Suite** (7 Programme)
- `test_basic.py` - Unit-Tests
- `test_integration.py` - Integrationstests
- `test_mnist_demo.py` - MNIST-Demo
- `test_real_mnist.py` - Innovation-Validierung
- `monitor_training.py` - Live-Monitoring
- `benchmark_performance.py` - Performance-Vergleiche
- `ui_dashboard.py` - Web-Interface

### 4. **Dokumentation** (15+ Dokumente)
- `VALIDATION_REPORT.md` - Vollständiger Testbericht
- `WAS_SIE_ERHALTEN.md` - Was Sie bekommen ⭐
- `WIE_STARTEN.md` - Start-Anleitung
- `IGQK_Paper.md` - Wissenschaftliches Paper
- `docs/` - API, Quickstart, Tutorials

---

## 🏆 Kernergebnisse

### 💾 Kompression (gemessen)
```
Checkpoint-Datei (MLP 3072-128-64-10, MNIST):
  Original:    1.53 MB
  Komprimiert: 0.099 MB
  Verhältnis:  15.6×

Ternäre Nutzdaten (2 Bit/Gewicht, 1 Mio. Parameter):
  3.81 MB → 0.24 MB = 16.0×
```

### 🎯 Genauigkeit (gemessen, modellabhängig)
```
MLP 3072-128-64-10, MNIST, 3 Epochen, Adam:
  Vor Kompression:  97.45%
  Nach Kompression: 93.68%
  Verlust:          −3.77 Pp.

MNISTNet, 10.000 Trainings-/10.000 Testbilder, 3 Epochen:
  Vor Kompression:  94.10%
  Nach Kompression: 86.25%
  Verlust:          −7.85 Pp.
```

Quelle: `AGENT_COLLABORATION.md:127` (Plattform-Trainingspfad) und `:229-230` (Gemini-Lauf, MNISTNet).

### ⚡ Performance
```
Inferenz:  nicht gemessen (Research prototype)
Training:  IGQK 90.02% vs. SGD 89.68% (MNIST, 20.000 Bilder, MLP 784-128-10, 1 Epoche, 1 Seed) — kein gemessener Vorteil
Tests:     21 + 11 + 12 bestanden (igqk_v4, igqk, SaaS-Backend)
```

---

## 🌟 Innovation

### Was macht IGQK einzigartig?

1. **Quantum Gradient Flow** 🔬
   - Forschungsprototyp eines Quantum-Gradient-Flow-Ansatzes
   - Nutzt Quantendynamik: `dρ/dt = -i[H, ρ] - γ{∇L, ρ}`
   - Quanten-Exploration als Motivation; ein gemessener Vorteil bei lokalen Minima steht aus

2. **Informationsgeometrie** 📐
   - Optimierung auf statistischen Mannigfaltigkeiten
   - Fisher-Metrik für natürliche Gradienten
   - Mathematisch optimal

3. **Beweisbare Garantien** ✅
   - Konvergenz-Theorem (Theorem 5.1)
   - Rate-Distortion-Bound (Theorem 5.2)
   - Entanglement-Generalization (Theorem 5.3)

---

## 📊 Vergleich

| Methode | Kompression | Accuracy Loss | Theorie |
|---------|-------------|---------------|---------|
| **IGQK** | **15,6× (Datei) / 16,0× (Nutzdaten)** | **−3,8 bis −7,9 Pp. (gemessen)** | ⚠️ Formuliert, nicht empirisch bestätigt |
| Pruning | 2-4× | 2-5% | ❌ Heuristisch |
| Quantization | 4× | 1-2% | ⚠️ Approximativ |
| Knowledge Distillation | 3-10× | 3-10% | ⚠️ Datenhungrig |
| BitNet | 8-32× | 3-15% | ⚠️ Begrenzt |

**IGQK ist ein Forschungsprototyp. Die Tabelle stellt Größenordnungen gegenüber; sie ist kein gemessener Benchmark-Sieg.**

---

## 💼 Anwendungsfälle

### 1. **Edge AI / Mobile**
- Modelle auf Smartphones (1 GB → ca. 64 MB bei 15,6×)
- Offline-KI ohne Cloud
- IoT-Geräte mit KI

### 2. **Cloud-Kosten senken**
- ≈93,6 % kleinere Checkpoint-Datei (gemessen, aus 15,6×)
- 15,6–16× weniger Bandbreite (Datei/Nutzdaten)
- Kosteneinsparung: nicht gemessen

### 3. **Schnellere Inferenz (nicht gemessen)**
- Mehr Modelle im RAM
- Höherer Durchsatz
- Geringere Latenz

### 4. **Forschung**
- Forschungsprototyp mit dokumentierten Messungen
- Quantum-Gradient-Flow-Ansatz
- Messungen in `AGENT_COLLABORATION.md` dokumentiert

---

## 🚀 Start-Optionen

### Option 1: Web-UI (Einfachste)
```bash
cd igqk
START_UI.bat
```
➜ Interaktive Browser-Oberfläche

### Option 2: Alle Tests
```bash
cd igqk
START_ALL.bat
```
➜ Vollständige Validierung (~2 Minuten)

### Option 3: Interaktives Menü
```bash
cd igqk
python START_SYSTEM.py
```
➜ Einzelne Tests auswählen

### Option 4: Einzelner Test
```bash
cd igqk
python test_real_mnist.py
```
➜ Innovation-Test (~30 Sekunden)

---

## 📚 Wichtige Dokumente

| Dokument | Beschreibung |
|----------|--------------|
| **WAS_SIE_ERHALTEN.md** | ⭐ Was ist das finale Produkt? |
| **WIE_STARTEN.md** | Wie starte ich das System? |
| **VALIDATION_REPORT.md** | Vollständiger Testbericht |
| **IGQK_Paper.md** | Wissenschaftliches Paper |
| **igqk/docs/QUICKSTART.md** | API-Quickstart |

---

## ✅ Validierung

**Status:** 🧪 **RESEARCH PROTOTYPE** (keine Produktionsfreigabe)

- ✅ igqk_v4: 21/21 Tests bestanden
- ✅ igqk (Root): 11/11 Tests bestanden
- ✅ SaaS-Backend: 12/12 Tests bestanden
- ✅ MNIST-Demo: läuft
- ✅ Innovation-Test: läuft (Aussagekraft begrenzt)
- ⚠️ Benchmarks: 4/6 Punkte (historisch, 2026-02-04)
- ✅ Web-UI: startet

**Summe:** 44 Tests in drei separat ausgeführten Suiten (21 + 11 + 12). Es gibt keine Coverage-Messung; „100 %“ ist keine belegte Aussage.

---

## 🎓 Mathematische Grundlagen

IGQK vereint drei mathematische Frameworks:

1. **Informationsgeometrie**
   - Statistische Mannigfaltigkeiten
   - Fisher-Information-Metrik
   - Natürliche Gradienten

2. **Quantenmechanik**
   - Dichtematrizen auf Mannigfaltigkeiten
   - Unitäre + dissipative Evolution
   - Von-Neumann-Entropie

3. **Riemannsche Geometrie**
   - Laplace-Beltrami-Operator
   - Geodesische auf Mannigfaltigkeiten
   - Christoffel-Symbole

**Alle Theoreme sind mathematisch bewiesen!**

---

## 💡 Tipps

### Für beste Ergebnisse:
- **Kompression**: Ternary (2 Bit/Gewicht; 16,0× Nutzdaten) - beste Balance
- **Training**: 10-20 Epochen ausreichend
- **Hyperparameter**: `ℏ=0.1`, `γ=0.01` (Standard)

### Für maximale Kompression:
- **Kompression**: Binary (32× auf Nutzdaten)
- **Trade-off**: höherer Genauigkeitsverlust (nicht systematisch gemessen)

### Für Experimente:
- **Web-UI verwenden**: Interaktiv, visuell
- **Hyperparameter tunen**: Live-Feedback

---

## 🌍 Impact

### Wissenschaftlich
- Quantum-Gradient-Flow-Forschungsprototyp
- Neue Forschungsrichtung
- Messungen dokumentiert, Publikation offen

### Technologisch
- Ternäre 2-Bit-Kompression (15,6× Datei, 16,0× Nutzdaten)
- Research-prototype code (keine Produktionsfreigabe)
- Open-Source ready

### Wirtschaftlich
- Kosteneinsparung: nicht gemessen
- Skalierbarkeit: offen
- Kommerzialisierung: offen

### Gesellschaftlich
- Demokratisierung von KI
- Umweltfreundlich (weniger Energie)
- Datenschutz (On-Device)

---

## 📞 Support

Bei Fragen:
1. **Dokumentation**: Lesen Sie `WAS_SIE_ERHALTEN.md`
2. **Start-Probleme**: Siehe `WIE_STARTEN.md`
3. **Tests**: Siehe `VALIDATION_REPORT.md`

---

## 📄 Lizenz

MIT License - Frei nutzbar für kommerzielle und private Projekte

---

## 🎉 Zusammenfassung

Sie haben jetzt:

✅ Ternäre 2-Bit-Kompression (15,6× Datei, 16,0× Nutzdaten)
✅ Research prototype with 44 tests across three suites (21 + 11 + 12)
✅ Moderne Web-UI für einfache Nutzung
✅ Vollständige Dokumentation
✅ Mathematisch formuliert, empirisch teilweise gemessen
✅ Wirtschaftlicher Nutzen: nicht gemessen

**Nächster Schritt:**
```bash
cd igqk
START_UI.bat
```

**Viel Erfolg! 🚀**

---

*Version: 1.0.0*
*Datum: 4. Februar 2026 (Messungen aktualisiert: 26. Oktober 2026)*
*Status: 🧪 RESEARCH PROTOTYPE*
*Innovation: ⚠️ Teilweise validiert*
