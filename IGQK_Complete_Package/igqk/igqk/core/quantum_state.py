"""
Quantum state representation for IGQK.

Implements density matrices and quantum operations on statistical manifolds.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np


class QuantumState:
    """
    Represents a quantum state (density matrix) on a statistical manifold.
    
    For efficiency, we use a low-rank representation: ρ = Σᵢ λᵢ |ψᵢ⟩⟨ψᵢ|

    Column 0 of `eigenvectors` is the mean parameter vector θ; columns 1.. are unit-norm fluctuation
    modes around it (directions with zero mean), weighted by their eigenvalues.
    """
    
    def __init__(
        self,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            eigenvectors: (n_params, rank) tensor of eigenvectors
            eigenvalues: (rank,) tensor of eigenvalues (must sum to 1)
            device: torch device
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eigenvectors = eigenvectors.to(self.device)
        self.eigenvalues = eigenvalues.to(self.device)
        
        # Normalize eigenvalues
        self.eigenvalues = self.eigenvalues / self.eigenvalues.sum()
        
        assert self.eigenvalues.shape[0] == self.eigenvectors.shape[1]
        assert torch.all(self.eigenvalues >= 0), "Eigenvalues must be non-negative"
        
    @property
    def rank(self) -> int:
        """Rank of the density matrix."""
        return self.eigenvalues.shape[0]
    
    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return self.eigenvectors.shape[0]
    
    def expectation(self, observable: torch.Tensor) -> torch.Tensor:
        """
        Compute expectation value: Tr(ρ O)
        
        Args:
            observable: (n_params,) tensor
            
        Returns:
            Scalar expectation value
        """
        # For a linear observable the fluctuation modes (zero mean) cancel: E[O] = Oᵀθ
        return torch.sum(observable * self.to_classical())
    
    def von_neumann_entropy(self) -> torch.Tensor:
        """
        Compute von Neumann entropy: S(ρ) = -Tr(ρ log ρ) = -Σᵢ λᵢ log λᵢ
        """
        # Avoid log(0)
        eps = 1e-10
        log_eigs = torch.log(self.eigenvalues + eps)
        return -torch.sum(self.eigenvalues * log_eigs)
    
    def purity(self) -> torch.Tensor:
        """
        Compute purity: Tr(ρ²) = Σᵢ λᵢ²
        
        Returns 1 for pure states, < 1 for mixed states.
        """
        return torch.sum(self.eigenvalues ** 2)
    
    def to_classical(self) -> torch.Tensor:
        """
        Collapse to classical state (expectation value).

        The fluctuation modes have zero mean, so the expectation is the mean θ (column 0). Mixing
        the unit-norm modes into it (Σᵢ λᵢ ψᵢ) shrank θ by λ₀ and added random directions.

        Returns:
            (n_params,) tensor of parameter values
        """
        return self.eigenvectors[:, 0]
    
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample classical states from the quantum distribution.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            (n_samples, n_params) tensor
        """
        # θ + Σₖ √λₖ ξₖ ψₖ with ξₖ ~ N(0, 1): the mean plus Gaussian spread along the fluctuation modes
        mean = self.to_classical()
        if self.rank == 1:
            return mean.unsqueeze(0).repeat(n_samples, 1)
        xi = torch.randn(n_samples, self.rank - 1, device=self.device)
        spread = (xi * self.eigenvalues[1:].sqrt()) @ self.eigenvectors[:, 1:].T
        return mean.unsqueeze(0) + spread  # (n_samples, n_params)
    
    @classmethod
    def from_classical(
        cls,
        params: torch.Tensor,
        hbar: float = 0.1,
        device: Optional[torch.device] = None
    ) -> 'QuantumState':
        """
        Create quantum state from classical parameters with uncertainty.
        
        Args:
            params: (n_params,) classical parameter vector
            hbar: Quantum uncertainty (spread)
            device: torch device
            
        Returns:
            QuantumState centered at params
        """
        device = device or params.device
        n_params = params.shape[0]
        
        # Create low-rank approximation with Gaussian spread
        # Main eigenstate: the classical state itself
        eigenvectors = params.unsqueeze(1)  # (n_params, 1)
        eigenvalues = torch.tensor([1.0], device=device)
        
        # Add small perturbations (quantum fluctuations)
        if hbar > 0:
            n_fluctuations = min(10, n_params // 10)  # Low-rank
            fluctuations = torch.randn(n_params, n_fluctuations, device=device) * hbar
            fluctuations = fluctuations / torch.norm(fluctuations, dim=0, keepdim=True)
            
            eigenvectors = torch.cat([eigenvectors, fluctuations], dim=1)
            
            # Eigenvalues decay exponentially
            fluc_eigenvalues = torch.exp(-torch.arange(n_fluctuations, device=device, dtype=torch.float32))
            fluc_eigenvalues = fluc_eigenvalues * hbar
            eigenvalues = torch.cat([eigenvalues, fluc_eigenvalues])
            
        return cls(eigenvectors, eigenvalues, device)


class QuantumGradientFlow:
    """
    Implements the quantum gradient flow dynamics:
    dρ/dt = -i[H, ρ] - γ{∇L, ρ}
    """
    
    def __init__(
        self,
        hbar: float = 0.1,
        gamma: float = 0.01,
        mass: float = 1.0
    ):
        """
        Args:
            hbar: Quantum uncertainty parameter
            gamma: Damping coefficient
            mass: Mass parameter for kinetic energy
        """
        self.hbar = hbar
        self.gamma = gamma
        self.mass = mass
        
    def hamiltonian(
        self,
        state: QuantumState,
        fisher_metric: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Hamiltonian (kinetic energy): H = -ℏ²/(2m) Δ_M
        
        For low-rank states, approximate as spread of eigenvectors.
        
        Args:
            state: Current quantum state
            fisher_metric: Optional Fisher information matrix
            
        Returns:
            Hamiltonian operator (approximated)
        """
        # Kinetic energy ≈ variance of the state
        classical = state.to_classical()
        variance = torch.var(state.eigenvectors, dim=1)
        
        return -(self.hbar ** 2) / (2 * self.mass) * variance
    
    def step(
        self,
        state: QuantumState,
        gradient: torch.Tensor,
        fisher_metric: Optional[torch.Tensor] = None,
        dt: float = 0.01
    ) -> QuantumState:
        """
        Perform one step of quantum gradient flow.
        
        Args:
            state: Current quantum state
            gradient: Gradient of loss ∇L
            fisher_metric: Fisher information matrix (optional)
            dt: Time step
            
        Returns:
            Updated quantum state
        """
        # Natural gradient
        if fisher_metric is not None:
            nat_gradient = torch.linalg.solve(fisher_metric, gradient)
        else:
            nat_gradient = gradient
        
        new_eigenvectors = state.eigenvectors.clone()

        # Dissipative evolution (exploitation): the mean moves toward -gradient
        new_eigenvectors[:, 0] -= self.gamma * dt * nat_gradient

        # Unitary evolution (exploration): approximate [H, ρ] by rotating the fluctuation modes.
        # The mean gets no noise: noise of ħ·dt per step and weight is a random walk that, over a
        # few thousand steps, is as large as the weights themselves (training stayed at chance).
        if new_eigenvectors.shape[1] > 1:
            new_eigenvectors[:, 1:] += torch.randn_like(new_eigenvectors[:, 1:]) * np.sqrt(self.hbar * dt)
        
        # Normalize fluctuation eigenvectors only (col 1:), preserving classical parameter scale (col 0)
        if new_eigenvectors.shape[1] > 1:
            fluc_norms = torch.norm(new_eigenvectors[:, 1:], dim=0, keepdim=True)
            new_eigenvectors[:, 1:] = new_eigenvectors[:, 1:] / (fluc_norms + 1e-8)
        
        # Eigenvalues decay slightly (entropy increase)
        entropy_factor = 1.0 + dt * 0.01  # Slow entropy increase
        new_eigenvalues = state.eigenvalues / entropy_factor
        new_eigenvalues = new_eigenvalues / new_eigenvalues.sum()
        
        return QuantumState(new_eigenvectors, new_eigenvalues, state.device)
