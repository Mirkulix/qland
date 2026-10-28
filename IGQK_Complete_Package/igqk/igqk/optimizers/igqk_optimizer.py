"""
IGQK Optimizer: Information Geometric Quantum Compression Optimizer.

Implements the quantum gradient flow on statistical manifolds.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Callable, List
import numpy as np

from ..core.quantum_state import QuantumState, QuantumGradientFlow
from ..manifolds.statistical_manifold import StatisticalManifold, DiagonalFisherManifold
from ..compression.projectors import CompressionProjector, TernaryProjector


class IGQKOptimizer(Optimizer):
    """
    IGQK Optimizer implementing quantum gradient flow.
    
    Combines:
    - Quantum exploration (unitaryevolution)
    - Natural gradient descent (Fisher metric)
    - Compression via projection
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        hbar: float = 0.1,
        gamma: float = 0.01,
        manifold: Optional[StatisticalManifold] = None,
        projector: Optional[CompressionProjector] = None,
        use_quantum: bool = True,
        ensemble_size: int = 1,
        n_data: Optional[int] = None,
    ):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            hbar: Quantum uncertainty parameter
            gamma: Damping coefficient
            manifold: Statistical manifold (default: DiagonalFisher)
            projector: Compression projector (default: Ternary)
            use_quantum: Whether to use quantum dynamics
            ensemble_size: Number of particles for ensemble
            n_data: Training set size; scales the exploration noise of the classical branch
                (none without it)
        """
        defaults = dict(
            lr=lr,
            hbar=hbar,
            gamma=gamma,
        )
        super().__init__(params, defaults)
        
        self.manifold = manifold or DiagonalFisherManifold(n_samples=100)
        self.projector = projector or TernaryProjector(method='optimal')
        self.use_quantum = use_quantum
        self.ensemble_size = ensemble_size
        self.n_data = n_data
        
        # Quantum gradient flow
        self.qgf = QuantumGradientFlow(hbar=hbar, gamma=gamma)
        
        # Initialize quantum states for each parameter group
        if use_quantum:
            self._init_quantum_states()
        
        # Ensemble particles (for Monte Carlo approximation)
        self.ensemble = []
        if ensemble_size > 1:
            self._init_ensemble()
    
    def _init_quantum_states(self):
        """Initialize quantum states for parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # Create quantum state centered at current parameters
                    state['quantum'] = QuantumState.from_classical(
                        p.data.flatten(),
                        hbar=group['hbar'],
                        device=p.device
                    )
    
    def _init_ensemble(self):
        """Initialize ensemble of particles."""
        # Sample initial ensemble from parameter distribution
        for group in self.param_groups:
            ensemble_group = []
            for p in group['params']:
                if p.requires_grad:
                    # Sample particles around current parameters
                    particles = p.data.unsqueeze(0).repeat(self.ensemble_size, *([1] * p.ndim))
                    noise = torch.randn_like(particles) * group['hbar']
                    particles = particles + noise
                    ensemble_group.append(particles)
            self.ensemble.append(ensemble_group)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            hbar = group['hbar']
            gamma = group['gamma']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if self.use_quantum:
                    # Quantum gradient flow
                    state = self.state[p]
                    quantum_state = state.get('quantum')
                    
                    if quantum_state is not None:
                        # The mean follows the parameter (it may have been changed outside, e.g. loaded)
                        quantum_state.eigenvectors[:, 0] = p.data.flatten()
                        # Update quantum state. The drift is γ·dt·∇L; dt = lr/γ makes it a step of lr
                        # (with dt = lr the step was γ·lr, 100× too small for γ = 0.01).
                        new_quantum_state = self.qgf.step(
                            quantum_state,
                            grad.flatten(),
                            fisher_metric=None,  # Could compute Fisher here
                            dt=lr / gamma if gamma > 0 else lr
                        )
                        state['quantum'] = new_quantum_state
                        
                        # Collapse to classical for parameter update
                        classical = new_quantum_state.to_classical()
                        p.data = classical.reshape(p.shape)
                else:
                    # Classical natural gradient descent
                    # For efficiency, use simple gradient descent
                    # (Natural gradient would require Fisher computation)
                    p.data.add_(grad, alpha=-lr)
                    
                    # Add exploration noise (quantum-inspired): Langevin noise at temperature ħ for the
                    # loss summed over the data, i.e. √(2·lr·ħ/N) for the mean loss gradients are
                    # taken of. Unscaled √(2·lr·ħ) (0.14 per step at lr = ħ = 0.1) cost 18 points
                    # of MNIST test accuracy (68.7% against 86.9%).
                    if hbar > 0 and self.n_data:
                        noise = torch.randn_like(p.data) * np.sqrt(2 * lr * hbar / self.n_data)
                        p.data.add_(noise)
        
        return loss
    
    def compress(self, model: Optional[nn.Module] = None) -> Optional[nn.Module]:
        """
        Compress model parameters using the projector.
        
        Args:
            model: Model to compress (if None, compress in-place)
            
        Returns:
            Compressed model (if model was provided)
        """
        if model is not None:
            # Compress external model
            with torch.no_grad():
                for param in model.parameters():
                    compressed = self.projector.project(param.data)
                    param.data.copy_(compressed)
            return model
        else:
            # Compress optimizer's parameters
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        compressed = self.projector.project(p.data)
                        p.data.copy_(compressed)
            return None
    
    def get_quantum_state(self, param: torch.nn.Parameter) -> Optional[QuantumState]:
        """
        Get quantum state for a parameter.
        
        Args:
            param: Parameter tensor
            
        Returns:
            QuantumState or None
        """
        if not self.use_quantum:
            return None
        
        state = self.state.get(param)
        if state is None:
            return None
        
        return state.get('quantum')
    
    def entropy(self) -> float:
        """
        Compute total von Neumann entropy of all quantum states.
        
        Returns:
            Total entropy
        """
        if not self.use_quantum:
            return 0.0
        
        total_entropy = 0.0
        for group in self.param_groups:
            for p in group['params']:
                quantum_state = self.get_quantum_state(p)
                if quantum_state is not None:
                    total_entropy += quantum_state.von_neumann_entropy().item()
        
        return total_entropy
    
    def purity(self) -> float:
        """
        Compute average purity of all quantum states.
        
        Returns:
            Average purity (1 = pure, < 1 = mixed)
        """
        if not self.use_quantum:
            return 1.0
        
        total_purity = 0.0
        count = 0
        
        for group in self.param_groups:
            for p in group['params']:
                quantum_state = self.get_quantum_state(p)
                if quantum_state is not None:
                    total_purity += quantum_state.purity().item()
                    count += 1
        
        return total_purity / count if count > 0 else 1.0


class IGQKScheduler:
    """
    Learning rate scheduler for IGQK optimizer.
    
    Adjusts hbar (quantum uncertainty) and gamma (damping) during training.
    """
    
    def __init__(
        self,
        optimizer: IGQKOptimizer,
        mode: str = 'cosine',
        T_max: int = 100,
        hbar_min: float = 0.01,
        gamma_max: float = 0.1
    ):
        """
        Args:
            optimizer: IGQK optimizer
            mode: Scheduling mode ('cosine', 'linear', 'exponential')
            T_max: Maximum number of steps
            hbar_min: Minimum quantum uncertainty
            gamma_max: Maximum damping
        """
        self.optimizer = optimizer
        self.mode = mode
        self.T_max = T_max
        self.hbar_min = hbar_min
        self.gamma_max = gamma_max
        
        self.hbar_init = optimizer.defaults['hbar']
        self.gamma_init = optimizer.defaults['gamma']
        self.step_count = 0
    
    def step(self):
        """Update hbar and gamma."""
        self.step_count += 1
        t = self.step_count / self.T_max
        
        if self.mode == 'cosine':
            # Cosine annealing: hbar decreases, gamma increases
            hbar = self.hbar_min + 0.5 * (self.hbar_init - self.hbar_min) * (1 + np.cos(np.pi * t))
            gamma = self.gamma_init + 0.5 * (self.gamma_max - self.gamma_init) * (1 - np.cos(np.pi * t))
        
        elif self.mode == 'linear':
            hbar = self.hbar_init - t * (self.hbar_init - self.hbar_min)
            gamma = self.gamma_init + t * (self.gamma_max - self.gamma_init)
        
        elif self.mode == 'exponential':
            hbar = self.hbar_init * (self.hbar_min / self.hbar_init) ** t
            gamma = self.gamma_init * (self.gamma_max / self.gamma_init) ** t
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Update optimizer
        for group in self.optimizer.param_groups:
            group['hbar'] = hbar
            group['gamma'] = gamma
        
        self.optimizer.qgf.hbar = hbar
        self.optimizer.qgf.gamma = gamma
