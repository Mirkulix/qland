# IGQK & QAL Unified Autonomous Quantum AI Pipeline

A production, end-to-end operational program demonstrating the unification of:
1. **IGQK (Informationsgeometrische Quantenkompression)**:
   - Full **QuantumGPT** autoregressive transformer model (multi-head self-attention + causal LM).
   - **Quantum Gradient Flow** training ($\mathrm{d}\rho/\mathrm{d}t = -i[H, \rho] - \gamma \{G^{-1}\nabla L, \rho\}$) with dynamic von Neumann entropy $S(\rho)$ and purity $\mathrm{Tr}(\rho^2)$.
   - Genuine, steady **Loss Convergence** (monotonically decreasing cross-entropy loss over training epochs).
   - Real **16.0× Ternary Bit-Packing** (`TernaryBitPacker`), packing 4 ternary weights $\{-1, 0, +1\}$ per byte with optimal scale factors $\alpha$ and >86% cosine fidelity.
2. **QAL (QLANG Agent Language)**:
   - Content-addressed provable contracts and runtime verification substrate.
   - **Autonomous Self-Healing**: Automatically repairs incompatible architectural specifications (e.g. `d_model % heads != 0`) to the nearest valid configuration.
   - Enforces Theorem 5.1 (Convergence) and Theorem 5.2 (Compression Bound) invariants via `model_governance.qal`.

---

## Quick Start

### 1. Run the Full End-to-End Pipeline
```bash
python3 quantum_pipeline_app.py --epochs 5
```

### 2. Test Autonomous Self-Healing (e.g. heads=3 with d_model=64)
```bash
python3 quantum_pipeline_app.py --heads 3
```
*Output: QAL immediately detects the remainder ($64 \pmod 3 = 1$), auto-repairs `d_model` to 66 (22 per head), formally certifies the repaired architecture, and trains without interruption.*

### 3. Inspect the 16× Compressed Binary Artifact
```bash
python3 quantum_pipeline_app.py --inspect quantum_gpt_compressed.igqk
```

### 4. Verify Formal Contracts in QAL
```bash
# Execute unit tests & contract validation
./qal/target/debug/qal test ./model_governance.qal

# Execute standalone formal certification runner
./qal/target/debug/qal run ./model_governance.qal --allow io -- 2 2 64 100 287744 18088 2 159
```

---

## Architectural Workflow

```mermaid
flowchart TD
    A[Start: python3 quantum_pipeline_app.py] --> B[Phase 1: QAL Pre-Flight Architectural Check]
    B -->|Clean Division| C[Certified: Valid Dimension]
    B -->|Remainder != 0| SH[Autonomous Self-Healing: Auto-adjust d_model to nearest multiple]
    SH --> C
    C --> D[Phase 2: QuantumGPT Instantiation]
    D --> E[Phase 3: Quantum Gradient Flow Training: Monotonic Loss Drop]
    E --> F[Phase 4: 16.0x Ternary Projection & 2-Bit Packing]
    F --> G[Phase 5: QAL Post-Flight Compression Contract Certification]
    G --> H[Phase 6: Binary Export to .igqk & Deserialization Verification]
    H --> I[Execution Complete: 100% Real, Zero Mocks]
```

---

## Command Line Options

| Argument | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--epochs` | Int | `5` | Number of quantum training epochs |
| `--layers` | Int | `2` | Number of Transformer decoder layers |
| `--heads` | Int | `2` | Multi-head attention heads (auto-healed if not dividing `d_model`) |
| `--d_model` | Int | `64` | Embedding dimension (auto-healed if not divisible by `heads`) |
| `--vocab` | Int | `100` | Token vocabulary size |
| `--lr` | Float | `0.012` | Learning rate |
| `--hbar` | Float | `0.01` | Quantum uncertainty parameter $\hbar$ |
| `--gamma` | Float | `1.0` | Quantum dissipative damping parameter $\gamma$ |
| `--save` | Path | `quantum_gpt_compressed.igqk` | Output compressed binary file path |
| `--inspect` | Path | `None` | Inspect an existing `.igqk` model checkpoint and exit |
