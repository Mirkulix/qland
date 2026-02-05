# 🚀 IGQK v4.0 - RALPH-LOOP MASTER GUIDE

**Vollständiger Plan zur Fertigstellung von IGQK v4.0 mit Ralph-loop**

---

## 📊 ÜBERSICHT

```
IGQK v4.0 Fertigstellung in 3 PHASEN:

Phase 1: MODELS          [50 Iterations, ~1-2h]
├─ QuantumGPT
├─ QuantumBERT
└─ QuantumViT

Phase 2: MULTI-MODAL     [40 Iterations, ~1-2h]
├─ Vision Encoder
├─ Language Encoder
├─ Quantum Fusion
└─ MultiModalModel

Phase 3: INTEGRATION     [30 Iterations, ~1h]
├─ Complete Trainer
├─ HLWT/TLGT/FCHL Integration
├─ Example Scripts
└─ Tests

Total: ~3-5 Stunden für v4.0 ALPHA!
```

---

## 🔧 SETUP (EINMALIG)

### Schritt 1: Ralph-loop Plugin installieren

```bash
cd C:\Users\a.b\Workspace\IGQK

# Kopiere Plugin ins Projekt
mkdir -p .claude/plugins
cp -r /tmp/claude-plugins-official/plugins/ralph-loop .claude/plugins/

# Verifiziere
ls .claude/plugins/ralph-loop/
```

### Schritt 2: Arbeitsverzeichnis vorbereiten

```bash
cd C:\Users\a.b\Workspace\IGQK\IGQK_Complete_Package\igqk_v4

# Erstelle Output-Ordner für Logs
mkdir -p .ralph-logs
```

---

## 🚀 PHASE 1: MODELS (START HERE!)

### Starten

```bash
# 1. Beende aktuelle Claude Code Session
exit

# 2. Wechsle ins igqk_v4 Verzeichnis
cd C:\Users\a.b\Workspace\IGQK\IGQK_Complete_Package\igqk_v4

# 3. Starte Claude Code neu
npx @anthropic-ai/claude-code
```

### Im neuen Chat: Phase 1 Ralph-loop starten

```bash
/ralph-loop "Implement missing model classes for IGQK v4.0.

GOAL: Implement QuantumGPT, QuantumBERT, QuantumViT in models/

REQUIREMENTS:
1. Create models/ directory with __init__.py
2. Implement QuantumGPT (GPT-2 style decoder)
   - Multi-head attention with causal mask
   - Feed-forward networks
   - Optional ternary weights via TernaryLinear
   - generate() method
3. Implement QuantumBERT (bidirectional encoder)
   - Same as GPT but NO causal mask
   - [CLS] token support
4. Implement QuantumViT (Vision Transformer)
   - Patch embedding (Conv2d)
   - [CLS] token
   - Works with (batch, 3, 224, 224) images

KEY INTEGRATION:
- Import TernaryLinear from theory.tlgt.ternary_lie_group
- Use TernaryLinear when config.train_compressed=True
- Use nn.Linear when config.train_compressed=False
- Compatible with QuantumTrainingConfig

SUCCESS CRITERIA:
✅ models/__init__.py exports all models
✅ QuantumGPT forward & generation work
✅ QuantumBERT forward works
✅ QuantumViT forward works (images → features)
✅ Ternary mode works
✅ Standard mode works
✅ No import errors
✅ Test script passes

REFERENCE:
- See IGQK_V4_MASTER_PLAN.md for complete code
- See RALPH_PHASE1_MODELS.md for full spec
- theory/tlgt/ternary_lie_group.py has TernaryLinear

OUTPUT when complete: <promise>MODELS_PHASE_COMPLETE</promise>

WORKING DIRECTORY: igqk_v4/" --completion-promise "MODELS_PHASE_COMPLETE" --max-iterations 50
```

### Was passiert

Ralph-loop wird:
1. Analysiere Projekt-Struktur
2. Erstelle models/ Ordner
3. Implementiere QuantumGPT Schritt für Schritt
4. Teste QuantumGPT
5. Fixe Errors
6. Implementiere QuantumBERT
7. Teste QuantumBERT
8. Implementiere QuantumViT
9. Teste QuantumViT
10. Integration Tests
11. Output: `<promise>MODELS_PHASE_COMPLETE</promise>`

**Geschätzte Zeit:** 15-25 Iterationen, 30-90 Minuten

### Nach Phase 1

Wenn du siehst:
```
<promise>MODELS_PHASE_COMPLETE</promise>
```

**→ PHASE 1 FERTIG! Weiter zu Phase 2!**

---

## 🎨 PHASE 2: MULTI-MODAL

### Starten

```bash
# Wenn Phase 1 fertig, im gleichen Chat:
/cancel-ralph  # Falls noch läuft

# Dann starte Phase 2:
/ralph-loop "Implement Multi-Modal AI components for IGQK v4.0.

GOAL: Implement Vision + Language + Fusion

REQUIREMENTS:
1. multimodal/vision/vision_encoder.py
   - QuantumVisionEncoder (ViT-based)
   - Input: (batch, 3, 224, 224)
   - Output: (batch, d_model)

2. multimodal/language/language_encoder.py
   - QuantumLanguageEncoder (BERT-based)
   - Input: (batch, seq_len)
   - Output: (batch, d_model)

3. multimodal/fusion/quantum_fusion.py
   - QuantumMultiModalFusion
   - Cross-modal attention
   - Quantum gates (unitary transforms)
   - Fusion projection

4. multimodal/models/multimodal_model.py
   - MultiModalModel combining all
   - forward(images, text)
   - contrastive_loss() CLIP-style

INTEGRATION:
- Can reuse QuantumViT from models/vit.py for vision
- Can reuse QuantumBERT from models/bert.py for language
- Fusion uses quantum entanglement concept

SUCCESS CRITERIA:
✅ All multimodal directories created
✅ QuantumVisionEncoder works
✅ QuantumLanguageEncoder works
✅ QuantumMultiModalFusion works
✅ MultiModalModel integrates all
✅ Forward pass works
✅ Contrastive loss works
✅ Tests pass

REFERENCE:
- See IGQK_V4_MASTER_PLAN.md
- See RALPH_PHASE2_MULTIMODAL.md
- models/vit.py and models/bert.py exist from Phase 1

OUTPUT when complete: <promise>MULTIMODAL_PHASE_COMPLETE</promise>

WORKING DIRECTORY: igqk_v4/" --completion-promise "MULTIMODAL_PHASE_COMPLETE" --max-iterations 40
```

**Geschätzte Zeit:** 15-30 Iterationen, 30-90 Minuten

---

## 🔗 PHASE 3: INTEGRATION

### Starten

```bash
# Wenn Phase 2 fertig:
/cancel-ralph

# Phase 3:
/ralph-loop "Complete QuantumLLMTrainer and verify end-to-end training.

GOAL: Finish Trainer implementation and test everything

REQUIREMENTS:
1. Complete QuantumLLMTrainer methods:
   - train_epoch() - Main training loop
   - validate() - Validation loop
   - fit() - Multi-epoch training
   - save_checkpoint() / load_checkpoint()

2. Integrate Theory Modules:
   - HLWT: Adaptive learning rate in train_epoch()
   - TLGT: Project to ternary after optimizer.step()
   - FCHL: Update memory after updates

3. Example Scripts:
   - examples/training/train_mnist_gpt.py
   - examples/training/train_multimodal_clip.py

4. Tests:
   - tests/test_training.py
   - Test all training functions
   - Test HLWT/TLGT/FCHL integration

SUCCESS CRITERIA:
✅ QuantumLLMTrainer 100% complete
✅ train_epoch(), validate(), fit() work
✅ HLWT adaptive LR works
✅ TLGT ternary projection works
✅ FCHL memory updates work
✅ Example scripts run successfully
✅ All tests pass
✅ Can train QuantumGPT end-to-end
✅ Can train MultiModalModel end-to-end
✅ Loss decreases over epochs

REFERENCE:
- See RALPH_PHASE3_INTEGRATION.md
- quantum_training/trainers/quantum_llm_trainer.py
- theory/hlwt/, theory/tlgt/, theory/fchl/

OUTPUT when complete: <promise>IGQK_V4_ALPHA_COMPLETE</promise>

WORKING DIRECTORY: igqk_v4/" --completion-promise "IGQK_V4_ALPHA_COMPLETE" --max-iterations 30
```

**Geschätzte Zeit:** 10-20 Iterationen, 20-60 Minuten

---

## ✅ FERTIG!

Wenn du siehst:
```
<promise>IGQK_V4_ALPHA_COMPLETE</promise>
```

**🎉 IGQK v4.0 ALPHA IST FERTIG! 🎉**

### Was jetzt funktioniert:

```
✅ QuantumGPT, QuantumBERT, QuantumViT
✅ Multi-Modal AI (Vision + Language + Fusion)
✅ Quantum Training from Scratch
✅ Direct Ternary Compression
✅ HLWT Adaptive Learning Rates
✅ TLGT Geodesic Optimization
✅ FCHL Fractional Memory
✅ Example Training Scripts
✅ Tests

Status: 50% of v4.0 complete!
```

---

## 📊 FORTSCHRITT TRACKING

### Während Ralph-loop läuft

Ralph gibt regelmäßig Progress-Updates:
```
[RALPH-PROGRESS]
Phase: 1.1 - QuantumGPT
Status: In Progress
Completed:
- Created models/ directory
- Implemented QuantumMultiHeadAttention
Current Task: Implementing QuantumFeedForward
Next: QuantumTransformerBlock
[/RALPH-PROGRESS]
```

### Abbruch bei Problemen

Wenn etwas schief geht:
```bash
/cancel-ralph

# Check was passiert ist:
git status
git log

# Fix das Problem manuell
# Dann ralph neu starten
```

---

## ⚠️ TROUBLESHOOTING

### Ralph läuft ewig

```bash
# Check max-iterations
# Sollte nicht über 50 sein pro Phase

# Abbrechen:
/cancel-ralph
```

### Import Errors

Ralph wird diese fixen, aber falls persistent:
```bash
# Check ob alle Ordner existieren:
ls -la igqk_v4/models/
ls -la igqk_v4/multimodal/

# Check __init__.py files
find igqk_v4 -name "__init__.py"
```

### API Kosten

Ralph kann viele API Calls machen:
- Phase 1: ~15-25 Calls
- Phase 2: ~15-30 Calls
- Phase 3: ~10-20 Calls
- **Total: ~40-75 Calls**

Bei langen Konversationen → höhere Kosten!

---

## 🎯 QUICK START (TL;DR)

```bash
# 1. Setup (einmalig)
cd C:\Users\a.b\Workspace\IGQK
cp -r /tmp/claude-plugins-official/plugins/ralph-loop .claude/plugins/

# 2. Phase 1 starten
cd IGQK_Complete_Package/igqk_v4
npx @anthropic-ai/claude-code

# 3. Im Chat - Phase 1:
/ralph-loop "<paste from RALPH_PHASE1_MODELS.md>" --completion-promise "MODELS_PHASE_COMPLETE" --max-iterations 50

# 4. Warte auf: <promise>MODELS_PHASE_COMPLETE</promise>

# 5. Phase 2:
/ralph-loop "<paste from RALPH_PHASE2_MULTIMODAL.md>" --completion-promise "MULTIMODAL_PHASE_COMPLETE" --max-iterations 40

# 6. Phase 3:
/ralph-loop "<paste from RALPH_PHASE3_INTEGRATION.md>" --completion-promise "IGQK_V4_ALPHA_COMPLETE" --max-iterations 30

# 7. FERTIG! 🎉
```

---

## 📚 ALLE DOKUMENTE

```
📄 RALPH_MASTER_GUIDE.md (diese Datei)
   → Übersicht & Anleitung

📄 RALPH_PHASE1_MODELS.md
   → Detaillierter Prompt für Phase 1

📄 RALPH_PHASE2_MULTIMODAL.md
   → Detaillierter Prompt für Phase 2

📄 RALPH_PHASE3_INTEGRATION.md
   → Detaillierter Prompt für Phase 3

📄 IGQK_V4_MASTER_PLAN.md
   → Kompletter Code für alle Features

📄 IGQK_VOLLSTAENDIGE_ANALYSE_V4.md
   → Vollständige Projekt-Analyse
```

---

## 🚀 NÄCHSTE SCHRITTE NACH ALPHA

Wenn v4.0 Alpha fertig:

1. **Phase 4: Distributed Training** (DDP/FSDP)
2. **Phase 5: AutoML** (Hyperparameter Tuning)
3. **Phase 6: Hardware** (CUDA Kernels)
4. **Phase 7: Deployment** (Edge/Cloud)
5. **Phase 8: Tests & Docs**

---

**READY? Lass Ralph die Arbeit machen! 🤖**

**STARTE JETZT:** Führe Schritt 1-3 aus dem Quick Start aus!
