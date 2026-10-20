# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**IGQK (Informationsgeometrische Quantenkompression)** - Information-Geometric Quantum Compression

This is a theoretical mathematical framework for neural network compression that combines:
- **Information Geometry**: Statistical manifolds with Fisher-Information metric
- **Quantum Mechanics Principles**: Superposition and entanglement for weight optimization
- **Compression Theory**: Projection onto low-dimensional submanifolds

The theory unifies three mathematical approaches:
1. **HLWT** (Hybrid Laplace-Wavelet Transformation)
2. **TLGT** (Ternary Lie Group Theory)
3. **FCHL** (Fractional Calculus for Hebbian Learning)

## Current State

This repository contains mathematical documentation only (no implementation yet):
- `Entwicklung_der_IGQK-Theorie_Mathematische_Details.pdf` - Complete mathematical formalization
- `KI-Analyse_Welche_mathematischen_Wege_machen_Sinn.pdf` - Meta-analysis and evaluation

## Build Requirements

**Windows Visual Studio C++ is required** for future implementations involving numerical libraries (as noted in user configuration).

When implementing the theory, you will likely need:
- C++ compiler with Visual Studio
- Python 3.8+ (for ML framework integration)
- NumPy/SciPy (for numerical computations)
- PyTorch or TensorFlow (for neural network integration)
- Libraries for Riemannian geometry computations

## Core Mathematical Concepts

### 1. Statistical Manifold
Neural network weights θ form a Riemannian manifold M with Fisher-Information metric:
```
g_ij(θ) = E_θ[∂_i log p(x; θ) · ∂_j log p(x; θ)]
```

### 2. Quantum State on Manifold
Weights are represented as density matrices ρ: M → ℂ^{d×d} with:
- ρ(θ) ≥ 0 (positive semidefinite)
- Tr(ρ(θ)) = 1 (normalized)
- ρ is smooth mapping

### 3. Quantum Gradient Flow
Evolution equation:
```
dρ/dt = -i[H, ρ] - γ{G^{-1}∇L, ρ}
```
Where:
- H = -Δ_M (Laplace-Beltrami operator)
- [H, ρ] = unitary evolution (quantum exploration)
- {G^{-1}∇L, ρ} = dissipative evolution (gradient descent)
- γ > 0 (damping parameter)

### 4. Compression as Projection
Compression is projection Π: M → N onto low-dimensional submanifold N:
- N = {θ : W_θ ternary} (ternary weights {-1, 0, +1})
- N = {θ : rank(W_θ) ≤ r} (low-rank)
- N = {θ : ||W_θ||_0 ≤ s} (sparse)

### 5. Measurement and Collapse
Quantum measurement operators {M_w : w ∈ W} collapse continuous quantum state to discrete weights following Born rule:
```
P(w | ρ) = Tr(ρ M_w)
```

## Key Theorems

### Theorem 5.1 (Convergence)
Quantum gradient flow converges to stationary state ρ* with:
```
E_ρ*[L] ≤ min_{θ ∈ M} L(θ) + O(ℏ)
```
where ℏ is quantum uncertainty.

### Theorem 5.2 (Compression Bound)
Minimum distortion D for compression to k-dimensional N:
```
D ≥ (n-k)/(2β) · log(1 + β·σ²_min)
```
For ternary compression (n → n/16):
```
D ≥ (15n/16)/(2β) · log(1 + β·σ²_min)
```

### Theorem 5.3 (Entanglement & Generalization)
Entangled quantum states across layers improve generalization:
```
E_gen ≤ E_train + O(√(I(A:B)/n))
```
where I(A:B) is quantum mutual information.

## IGQK Training Algorithm (Algorithm 1)

```
Input: Training data D, architecture f_θ, compression submanifold N
Output: Compressed weights w* ∈ N

1. Initialize:
   - ρ_0 = |θ_init⟩⟨θ_init|
   - ℏ = 0.1 (quantum uncertainty)
   - γ = 0.01 (damping)

2. Quantum Training (T steps):
   For t = 1 to T:
     a. Compute loss: L_t = E_{(x,y)~D}[loss(f_θ(x), y)]
     b. Compute gradient: ∇L_t
     c. Compute Fisher metric: G_t
     d. Update quantum state:
        ρ_{t+1} = ρ_t - dt·(i[H, ρ_t] + γ{G_t^{-1}∇L_t, ρ_t})
     e. Renormalize: ρ_{t+1} ← ρ_{t+1}/Tr(ρ_{t+1})

3. Compression:
   - Compute optimal projection: θ* = Π_N(E[ρ_T])
   - Or: Sample from ρ_T and project

4. Measurement:
   - Construct measurement operators {M_w : w ∈ N}
   - Measure: w* ~ P(w|ρ_T) = Tr(ρ_T M_w)

5. Return w*
```

**Complexity**: O(T · n² · d²) with low-rank approximation (d = O(log n)): O(T · n² · log² n)

## Implementation Architecture

When implementing IGQK, organize code as follows:

```
igqk/
├── core/
│   ├── manifold.py          # Statistical manifold with Fisher metric
│   ├── quantum_state.py     # Density matrix operations
│   ├── evolution.py         # Quantum gradient flow
│   └── measurement.py       # Measurement operators
├── compression/
│   ├── projection.py        # Projection to submanifolds
│   ├── ternary.py          # Ternary weight compression
│   ├── lowrank.py          # Low-rank compression
│   └── sparse.py           # Sparse compression
├── integration/
│   ├── pytorch.py          # PyTorch integration
│   ├── tensorflow.py       # TensorFlow integration
│   └── optimizer.py        # Custom optimizer interface
├── geometry/
│   ├── fisher.py           # Fisher information metric
│   ├── laplacian.py        # Laplace-Beltrami operator
│   └── geodesic.py         # Geodesic computation
├── theory/
│   ├── hlwt.py            # Hybrid Laplace-Wavelet Transform
│   ├── tlgt.py            # Ternary Lie Group Theory
│   └── fchl.py            # Fractional Calculus Hebbian Learning
└── experiments/
    ├── benchmarks/         # Standard ML benchmarks
    ├── analysis/           # Theoretical analysis scripts
    └── visualization/      # Result visualization
```

## Development Guidelines

### Mathematical Rigor
- All implementations must preserve mathematical properties from the theory
- Density matrices must remain positive semidefinite and normalized
- Fisher metric must remain positive definite
- Trace and normalization must be preserved at each step

### Numerical Stability
- Use matrix exponentials carefully (small time steps dt)
- Implement proper renormalization after each update
- Consider low-rank approximations for large models
- Use stable implementations of matrix logarithm for Fisher metric

### Testing Strategy
1. **Unit Tests**: Test each mathematical component in isolation
   - Density matrix properties
   - Fisher metric computation
   - Projection operators
   - Measurement operators

2. **Integration Tests**: Test quantum training on toy models
   - 2D toy problems (visualizable)
   - Small neural networks (MNIST)
   - Compare to classical gradient descent

3. **Theory Verification**: Verify theoretical bounds
   - Convergence guarantees
   - Compression bounds
   - Generalization bounds

### Debugging Quantum States
When debugging, monitor:
- `Tr(ρ)` should always be 1.0
- Eigenvalues of ρ should be in [0, 1]
- Loss should decrease (on average)
- Quantum uncertainty: `S(ρ) = -Tr(ρ log ρ)` (von Neumann entropy)

## Performance Considerations

### Computational Bottlenecks
1. **Fisher Metric Computation**: O(n² · batch_size)
   - Consider mini-batch approximations
   - Use empirical Fisher as approximation

2. **Quantum State Update**: O(n² · d²)
   - Use low-rank density matrices (d << n)
   - Kraus operators for efficient updates

3. **Matrix Inverse**: G^{-1}
   - Use conjugate gradient instead of direct inverse
   - Preconditioned methods

### Scalability to Large Models
- For models with 1B+ parameters:
  - Apply IGQK layer-wise
  - Use block-diagonal approximations for Fisher metric
  - Parallel quantum updates across layers
  - Consider hierarchical compression

## Open Research Questions

From the theoretical documentation:

1. **Optimal ℏ**: How to choose quantum uncertainty optimally?
2. **Entanglement Structure**: Which entanglement patterns are optimal?
3. **Quantum Advantage**: Is there genuine speedup over classical methods?
4. **Hardware Implementation**: Can IGQK run on quantum computers?
5. **Universality**: Does IGQK work for all architectures (CNNs, Transformers, etc.)?

## Related Theory Connections

### HLWT as Special Case (Proposition 6.1)
HLWT is Fourier transform of quantum gradient flow in local coordinates.

### TLGT as Special Case (Proposition 6.2)
Ternary Lie Group G₃ is discrete subgroup of quantum symmetry group on M.

### FCHL as Special Case (Proposition 6.3)
Fractional Hebbian Learning uses fractional Laplace-Beltrami operator H = -(-Δ_M)^α.

## Common Issues

### "Always getting errors"
As noted in user configuration, expect errors during setup. Common issues:
- Missing Visual Studio C++ build tools
- Numerical instability in quantum state updates
- Memory issues with large density matrices

### Where to see the process?
When implementing:
- Add logging for quantum state properties (trace, entropy)
- Visualize density matrix eigenvalues
- Track convergence metrics
- Monitor Fisher metric condition number

## Starting the System

Once implementation exists:

```bash
# Setup (requires VS C++)
python setup.py install

# Basic training example
python examples/train_mnist.py --method igqk --compression ternary

# Monitor quantum state
python tools/monitor_quantum_state.py --checkpoint model.pt

# Evaluate compression
python tools/evaluate_compression.py --model compressed_model.pt
```

## References

See the mathematical PDFs in this repository for complete theoretical foundations and proofs.
