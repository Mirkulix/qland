# 🚀 RALPH-LOOP PHASE 1: IGQK v4.0 MODELS

**Ziel:** Implementiere die fehlenden Model-Klassen für IGQK v4.0

---

## 📋 TASK SPECIFICATION

### Primary Goal
Implement the missing model classes for IGQK v4.0 so that `QuantumLLMTrainer` can create and train models.

### Current State
- ✅ Theory Layer (HLWT, TLGT, FCHL) implemented
- ✅ QuantumTrainingConfig complete
- ⚠️ QuantumLLMTrainer exists but can't create models (imports fail)
- ❌ models/ directory doesn't exist
- ❌ GPT, BERT, ViT models missing

### Target State
- ✅ models/ directory created with __init__.py
- ✅ QuantumGPT fully implemented and tested
- ✅ QuantumBERT fully implemented and tested
- ✅ QuantumViT fully implemented and tested
- ✅ All models integrate with TernaryLinear from theory/tlgt/
- ✅ All models work with QuantumTrainingConfig
- ✅ Test scripts pass

---

## 🎯 REQUIREMENTS

### 1. Directory Structure
```
igqk_v4/
├── models/
│   ├── __init__.py          # Export all models
│   ├── gpt.py               # QuantumGPT implementation
│   ├── bert.py              # QuantumBERT implementation
│   └── vit.py               # QuantumViT implementation
└── tests/
    └── test_models.py       # Basic model tests
```

### 2. QuantumGPT (models/gpt.py)

**Requirements:**
- Standard GPT architecture (transformer decoder)
- Multi-head self-attention with causal masking
- Feed-forward networks
- Layer normalization (Pre-LN style)
- Token + positional embeddings
- Language model head (tied weights with token embedding)
- **Optional ternary weights** via TernaryLinear when config.train_compressed=True
- Compatible with QuantumTrainingConfig
- `generate()` method for auto-regressive generation
- Proper weight initialization (GPT-2 style)

**Key Features:**
- Uses `TernaryLinear` from `theory.tlgt.ternary_lie_group` when ternary mode enabled
- Falls back to standard `nn.Linear` when ternary mode disabled
- Causal mask for autoregressive generation
- Support for attention mask

**Class Structure:**
```python
class QuantumMultiHeadAttention(nn.Module):
    # Multi-head attention with optional ternary weights

class QuantumFeedForward(nn.Module):
    # FFN with optional ternary weights

class QuantumTransformerBlock(nn.Module):
    # Complete transformer block

class QuantumGPT(nn.Module):
    # Main GPT model
    def __init__(self, config: QuantumTrainingConfig)
    def forward(self, input_ids, attention_mask=None)
    def generate(self, input_ids, max_new_tokens, temperature)
```

### 3. QuantumBERT (models/bert.py)

**Requirements:**
- Bidirectional transformer encoder
- Multi-head self-attention WITHOUT causal masking
- Same architecture as GPT but bidirectional
- [CLS] token support
- Masked language model head
- Optional ternary weights

**Key Differences from GPT:**
- NO causal mask (bidirectional)
- [CLS] token at start
- Different use case (encoding vs generation)

### 4. QuantumViT (models/vit.py)

**Requirements:**
- Vision Transformer architecture
- Patch embedding (Conv2d for patch extraction)
- [CLS] token for classification
- Positional embeddings for patches
- Transformer blocks (reuse from GPT)
- Classification head
- Optional ternary weights
- Support for 224x224 images (default)
- 16x16 patches (default)

**Key Features:**
```python
class PatchEmbedding(nn.Module):
    # Convert images to patches

class QuantumViT(nn.Module):
    # Main ViT model
    def __init__(self, config: QuantumTrainingConfig)
    def forward(self, images)  # (batch, 3, 224, 224)
```

### 5. Integration

**models/__init__.py:**
```python
from .gpt import QuantumGPT
from .bert import QuantumBERT
from .vit import QuantumViT

__all__ = ['QuantumGPT', 'QuantumBERT', 'QuantumViT']
```

### 6. Testing

**tests/test_models.py:**
- Test QuantumGPT forward pass
- Test QuantumGPT generation
- Test QuantumBERT forward pass
- Test QuantumViT forward pass
- Test ternary mode (verify TernaryLinear is used)
- Test standard mode (verify nn.Linear is used)
- Test compatibility with QuantumTrainingConfig

---

## 📐 IMPLEMENTATION GUIDELINES

### Import Structure
```python
# In models/gpt.py, bert.py, vit.py:
import torch
import torch.nn as nn
from typing import Optional
import math

from ..quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from ..theory.tlgt.ternary_lie_group import TernaryLinear
```

### Ternary Mode Logic
```python
# Use this pattern in all models:
if config.train_compressed:
    self.layer = TernaryLinear(in_features, out_features)
else:
    self.layer = nn.Linear(in_features, out_features)
```

### Weight Initialization
```python
# GPT-2 style initialization:
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

### Code Reuse
- QuantumTransformerBlock can be shared between GPT, BERT, ViT
- Only difference is masking strategy
- Create in gpt.py, import in bert.py and vit.py

---

## ✅ SUCCESS CRITERIA

### Phase 1.1: QuantumGPT (Complete First!)
1. [ ] models/ directory exists
2. [ ] models/__init__.py exists and exports QuantumGPT
3. [ ] models/gpt.py exists with complete implementation
4. [ ] QuantumGPT can be imported: `from models.gpt import QuantumGPT`
5. [ ] QuantumGPT.__init__ works with QuantumTrainingConfig
6. [ ] Forward pass works: input (batch, seq_len) → output (batch, seq_len, vocab_size)
7. [ ] Generation works: generate(start_tokens, max_new_tokens) returns tokens
8. [ ] Ternary mode works: when train_compressed=True, uses TernaryLinear
9. [ ] Standard mode works: when train_compressed=False, uses nn.Linear
10. [ ] No import errors, no runtime errors
11. [ ] Test script runs successfully

### Phase 1.2: QuantumBERT (After GPT works!)
1. [ ] models/bert.py exists
2. [ ] QuantumBERT implemented (bidirectional, no causal mask)
3. [ ] Forward pass works
4. [ ] Ternary mode works
5. [ ] Test script runs

### Phase 1.3: QuantumViT (After BERT works!)
1. [ ] models/vit.py exists
2. [ ] QuantumViT implemented with PatchEmbedding
3. [ ] Forward pass works for images (batch, 3, 224, 224)
4. [ ] Ternary mode works
5. [ ] Test script runs

### Phase 1.4: Integration (Final!)
1. [ ] QuantumLLMTrainer can create all three models without errors
2. [ ] All imports work: `from models import QuantumGPT, QuantumBERT, QuantumViT`
3. [ ] Test suite passes
4. [ ] No errors in any model

---

## 🔍 VERIFICATION STEPS

After each model implementation, run:

```python
# Test QuantumGPT
import torch
import sys
sys.path.append('igqk_v4')

from quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from models.gpt import QuantumGPT

config = QuantumTrainingConfig(
    model_type='GPT',
    n_layers=6,
    n_heads=8,
    d_model=512,
    vocab_size=10000,
    train_compressed=True,
)

model = QuantumGPT(config)
batch = torch.randint(0, 10000, (2, 10))
output = model(batch)
print(f"✅ QuantumGPT forward: {batch.shape} → {output.shape}")

generated = model.generate(batch[:1, :5], max_new_tokens=10)
print(f"✅ QuantumGPT generate: {generated.shape}")
```

---

## 📊 OUTPUT FORMAT

As you work, output progress in this format:

```
[RALPH-PROGRESS]
Phase: 1.1 - QuantumGPT
Status: In Progress
Completed:
- Created models/ directory
- Created models/__init__.py
- Implemented QuantumMultiHeadAttention
Current Task: Implementing QuantumFeedForward
Next: QuantumTransformerBlock
[/RALPH-PROGRESS]
```

---

## 🎯 COMPLETION SIGNAL

When **ALL** models are implemented and tested, output:

```
<promise>MODELS_PHASE_COMPLETE</promise>

Summary:
✅ QuantumGPT: Implemented and tested
✅ QuantumBERT: Implemented and tested
✅ QuantumViT: Implemented and tested
✅ All imports working
✅ All tests passing
✅ QuantumLLMTrainer integration verified

Ready for Phase 2: Multi-Modal!
```

---

## ⚠️ IMPORTANT NOTES

1. **Work incrementally**: Complete QuantumGPT FIRST before moving to BERT
2. **Test after each step**: Run verification after each model
3. **Use existing code**: TernaryLinear already exists in theory/tlgt/
4. **Follow patterns**: Look at quantum_training_config.py for style
5. **Fix all errors**: Don't move to next model until current one works
6. **Read context**: Check IGQK_V4_MASTER_PLAN.md for complete code examples

---

## 📚 REFERENCE FILES

Key files to reference:
- `igqk_v4/theory/tlgt/ternary_lie_group.py` - TernaryLinear implementation
- `igqk_v4/quantum_training/trainers/quantum_training_config.py` - Config structure
- `igqk_v4/quantum_training/trainers/quantum_llm_trainer.py` - Integration point
- `IGQK_V4_MASTER_PLAN.md` - Complete code examples for all models
- `IGQK_V4_QUICK_START.md` - Quick reference

---

## 🎯 ITERATION STRATEGY

Each iteration should:
1. Read previous iteration's work (files, git history)
2. Identify what's missing or broken
3. Implement one component
4. Test the component
5. Commit changes with clear message
6. Output progress update
7. Continue until completion signal

---

**WORKING DIRECTORY:** `C:\Users\a.b\Workspace\IGQK\IGQK_Complete_Package\igqk_v4`

**MAX ITERATIONS:** 50 (should complete in 10-20)

**COMPLETION PROMISE:** `MODELS_PHASE_COMPLETE`
