"""
Unit tests for quantum state module.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from igqk.core.quantum_state import QuantumState, QuantumGradientFlow


class TestQuantumState:
    """Tests for QuantumState class."""
    
    def test_initialization(self):
        """Test quantum state initialization."""
        n_params = 10
        rank = 3
        
        eigenvectors = torch.randn(n_params, rank)
        eigenvalues = torch.rand(rank)
        
        state = QuantumState(eigenvectors, eigenvalues)
        
        assert state.n_params == n_params
        assert state.rank == rank
        assert torch.allclose(state.eigenvalues.sum(), torch.tensor(1.0))
    
    def test_from_classical(self):
        """Test creating quantum state from classical parameters."""
        params = torch.randn(10)
        state = QuantumState.from_classical(params, hbar=0.1)
        
        # Should be centered at params
        classical = state.to_classical()
        assert torch.allclose(classical, params, atol=0.2)
    
    def test_expectation(self):
        """Test expectation value computation."""
        params = torch.randn(10)
        state = QuantumState.from_classical(params, hbar=0.01)
        
        observable = torch.ones(10)
        expectation = state.expectation(observable)
        
        # Should be close to sum of params
        assert torch.isclose(expectation, params.sum(), atol=0.1)
    
    def test_von_neumann_entropy(self):
        """Test von Neumann entropy."""
        params = torch.randn(10)
        state = QuantumState.from_classical(params, hbar=0.1)
        
        entropy = state.von_neumann_entropy()
        
        # Entropy should be non-negative
        assert entropy >= 0
    
    def test_purity(self):
        """Test purity computation."""
        params = torch.randn(10)
        
        # Pure state (small hbar)
        pure_state = QuantumState.from_classical(params, hbar=0.001)
        purity_pure = pure_state.purity()
        
        # Mixed state (large hbar)
        mixed_state = QuantumState.from_classical(params, hbar=1.0)
        purity_mixed = mixed_state.purity()
        
        # Pure state should have higher purity
        assert purity_pure > purity_mixed
        assert purity_pure <= 1.0
    
    def test_sampling(self):
        """Test sampling from quantum state."""
        params = torch.randn(10)
        state = QuantumState.from_classical(params, hbar=0.1)
        
        samples = state.sample(n_samples=100)
        
        assert samples.shape == (100, 10)
        
        # Mean of samples should be close to classical state
        mean_sample = samples.mean(dim=0)
        classical = state.to_classical()
        assert torch.allclose(mean_sample, classical, atol=0.5)


class TestQuantumGradientFlow:
    """Tests for QuantumGradientFlow class."""
    
    def test_initialization(self):
        """Test QGF initialization."""
        qgf = QuantumGradientFlow(hbar=0.1, gamma=0.01)
        
        assert qgf.hbar == 0.1
        assert qgf.gamma == 0.01
    
    def test_step(self):
        """Test quantum gradient flow step."""
        params = torch.randn(10)
        state = QuantumState.from_classical(params, hbar=0.1)
        
        gradient = torch.randn(10)
        
        qgf = QuantumGradientFlow(hbar=0.1, gamma=0.01)
        new_state = qgf.step(state, gradient, dt=0.01)
        
        # State should have evolved
        assert new_state.n_params == state.n_params
        assert new_state.rank == state.rank
        
        # Classical state should have moved in direction of -gradient
        old_classical = state.to_classical()
        new_classical = new_state.to_classical()
        
        # Direction check (approximately)
        direction = new_classical - old_classical
        expected_direction = -gradient
        
        # Should be negatively correlated with gradient
        correlation = torch.dot(direction, expected_direction)
        assert correlation > 0  # Moving toward -gradient


class TestIGQKOptimizerLearns:
    """The optimizer must learn: it stayed at chance on MNIST (9-16%) while SGD reached 88.7%."""

    def _task(self):
        torch.manual_seed(0)
        x = torch.randn(2000, 20)
        y = (x @ torch.randn(20, 4)).argmax(dim=1)  # 4 classes, learnable by a linear model
        return x[:1500], y[:1500], x[1500:], y[1500:]

    def _accuracy(self, **kw):
        import torch.nn as nn
        import torch.nn.functional as F
        from igqk.optimizers.igqk_optimizer import IGQKOptimizer
        xtr, ytr, xte, yte = self._task()
        model = nn.Linear(20, 4)
        opt = IGQKOptimizer(model.parameters(), lr=0.1, **kw)
        for _ in range(5):
            for i in range(0, 1500, 50):
                opt.zero_grad()
                F.cross_entropy(model(xtr[i:i + 50]), ytr[i:i + 50]).backward()
                opt.step()
        with torch.no_grad():
            return (model(xte).argmax(dim=1) == yte).float().mean().item()

    def test_quantum_branch_learns(self):
        assert self._accuracy(use_quantum=True) > 0.85

    def test_classical_branch_learns(self):
        assert self._accuracy(use_quantum=False, n_data=1500) > 0.85

    def test_to_classical_is_the_mean(self):
        params = torch.randn(100)
        state = QuantumState.from_classical(params, hbar=0.1)
        assert torch.equal(state.to_classical(), params)
        assert torch.allclose(state.sample(4000).mean(dim=0), params, atol=0.05)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
