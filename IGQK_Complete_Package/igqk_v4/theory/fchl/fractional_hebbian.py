"""
FCHL - Fractional Calculus for Hebbian Learning

Theoretical Framework from Roadmap Phase 2.

This module implements fractional derivatives for long-term memory in learning.
"""

import torch
import numpy as np
from typing import List, Iterator
from collections import deque


class FractionalHebbian:
    """
    Fractional Calculus for Hebbian Learning.

    Mathematical Formula (from Roadmap):
        D^α w = η · x · y    (0 < α < 1)

    Where:
    - D^α: Fractional derivative of order α
    - w: Weights
    - x: Input activation
    - y: Output activation
    - η: Learning rate

    Key Property:
    Power-law memory instead of exponential decay.
    This is more biologically plausible and better for long-term dependencies.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        memory_length: int = 100,
    ):
        """
        Initialize FCHL module.

        Args:
            alpha: Fractional order (0 < α < 1)
            memory_length: Number of past steps to remember
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.memory_length = memory_length

        # Memory buffer for parameter history
        self.param_history = {}

        # Precompute fractional weights (Grünwald-Letnikov coefficients)
        self.weights = self._compute_gl_weights(memory_length, alpha)

        print(f" FCHL initialized: α={alpha}, memory={memory_length}")

    def _compute_gl_weights(self, L: int, alpha: float) -> np.ndarray:
        """
        Compute Grünwald-Letnikov weights for fractional derivative.

        Formula:
            w_k = (-1)^k * C(alpha, k)

        where C(alpha, k) is binomial coefficient.

        Args:
            L: Memory length
            alpha: Fractional order

        Returns:
            Array of weights
        """
        weights = np.zeros(L)
        weights[0] = 1.0

        for k in range(1, L):
            weights[k] = weights[k-1] * (k - 1 - alpha) / k

        return weights

    def update_memory(self, parameters: Iterator[torch.nn.Parameter]):
        """
        Update memory buffer with current parameter values.

        Args:
            parameters: Model parameters iterator
        """
        for param in parameters:
            param_id = id(param)

            # Initialize history if needed
            if param_id not in self.param_history:
                self.param_history[param_id] = deque(maxlen=self.memory_length)

            # Store current parameter value
            self.param_history[param_id].append(param.data.clone().cpu())

    def compute_fractional_gradient(
        self,
        param: torch.nn.Parameter,
        current_grad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute fractional gradient incorporating memory.

        The fractional derivative is a weighted sum of past gradients,
        with weights decaying as a power law.

        Args:
            param: Parameter
            current_grad: Current gradient

        Returns:
            Fractional gradient
        """
        param_id = id(param)

        # If no history, return current gradient
        if param_id not in self.param_history or len(self.param_history[param_id]) < 2:
            return current_grad

        # Get parameter history
        history = list(self.param_history[param_id])
        L = len(history)

        # Compute fractional derivative using Grünwald-Letnikov formula
        fractional_grad = torch.zeros_like(current_grad)

        for k in range(min(L, self.memory_length)):
            if k < len(history):
                past_param = history[-(k+1)].to(current_grad.device)
                weight = self.weights[k]
                fractional_grad += weight * (current_grad - past_param)

        return fractional_grad

    def get_memory_stats(self) -> dict:
        """
        Get statistics about memory usage.

        Returns:
            Dictionary with memory statistics
        """
        total_params = len(self.param_history)
        avg_memory_length = np.mean([len(h) for h in self.param_history.values()]) if total_params > 0 else 0

        stats = {
            'tracked_parameters': total_params,
            'average_memory_length': avg_memory_length,
            'max_memory_length': self.memory_length,
            'alpha': self.alpha,
        }

        return stats

    def reset_memory(self):
        """Reset all memory buffers."""
        self.param_history = {}

    def visualize_memory_kernel(self):
        """Visualize the memory kernel (fractional weights)."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(self.weights, 'b-', linewidth=2)
            plt.xlabel('Time Steps Back', fontsize=12)
            plt.ylabel('Weight', fontsize=12)
            plt.title(f'Fractional Memory Kernel (α={self.alpha})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('fchl_memory_kernel.png', dpi=150)
            plt.close()

            print("📊 Memory kernel visualization saved to 'fchl_memory_kernel.png'")

        except ImportError:
            print("⚠️  matplotlib not available, skipping visualization")


class FractionalOptimizer(torch.optim.Optimizer):
    """
    Optimizer wrapper that applies fractional calculus.

    This wraps any standard optimizer (like Adam) and adds
    fractional memory effects.
    """

    def __init__(
        self,
        params,
        base_optimizer: torch.optim.Optimizer,
        fchl: FractionalHebbian,
    ):
        """
        Initialize fractional optimizer.

        Args:
            params: Model parameters
            base_optimizer: Base optimizer (e.g., Adam)
            fchl: FCHL module
        """
        self.base_optimizer = base_optimizer
        self.fchl = fchl

        defaults = dict()
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform optimization step with fractional gradients.

        Args:
            closure: Optional closure

        Returns:
            Loss (if closure provided)
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Update memory
        self.fchl.update_memory(self.param_groups[0]['params'])

        # Modify gradients with fractional memory
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # Compute fractional gradient
                    frac_grad = self.fchl.compute_fractional_gradient(param, param.grad)

                    # Replace gradient
                    param.grad = frac_grad

        # Call base optimizer step
        self.base_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


if __name__ == "__main__":
    print("🧪 Testing FCHL Module...\n")

    # Test 1: Initialize FCHL
    print("Test 1: Initialize FCHL")
    fchl = FractionalHebbian(alpha=0.7, memory_length=100)
    print(f"  Weights shape: {fchl.weights.shape}")
    print(f"  First 10 weights: {fchl.weights[:10]}")

    # Test 2: Memory update
    print("\nTest 2: Memory Update")
    model = torch.nn.Linear(10, 5)
    for i in range(20):
        fchl.update_memory(model.parameters())

    stats = fchl.get_memory_stats()
    print(f"  Memory stats: {stats}")

    # Test 3: Fractional gradient
    print("\nTest 3: Fractional Gradient")
    param = list(model.parameters())[0]
    grad = torch.randn_like(param)
    frac_grad = fchl.compute_fractional_gradient(param, grad)
    print(f"  Original grad norm: {torch.norm(grad):.4f}")
    print(f"  Fractional grad norm: {torch.norm(frac_grad):.4f}")

    # Test 4: Visualize memory kernel
    print("\nTest 4: Visualize Memory Kernel")
    fchl.visualize_memory_kernel()

    # Test 5: Fractional Optimizer
    print("\nTest 5: Fractional Optimizer")
    base_opt = torch.optim.Adam(model.parameters(), lr=0.001)
    frac_opt = FractionalOptimizer(model.parameters(), base_opt, fchl)

    # Dummy training loop
    x = torch.randn(32, 10)
    y = torch.randn(32, 5)

    for step in range(5):
        frac_opt.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        frac_opt.step()

        print(f"  Step {step}: Loss = {loss.item():.4f}")

    print("\n FCHL test completed!")
