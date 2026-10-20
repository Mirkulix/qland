"""
IGQK v4.0 Multi-Modal AI Package.

Provides QuantumVisionEncoder, QuantumLanguageEncoder, QuantumMultiModalFusion,
and the unified MultiModalModel with quantum entanglement.
"""

from .vision import QuantumVisionEncoder
from .language import QuantumLanguageEncoder
from .fusion import QuantumMultiModalFusion
from .models import MultiModalModel

__all__ = [
    'QuantumVisionEncoder',
    'QuantumLanguageEncoder',
    'QuantumMultiModalFusion',
    'MultiModalModel',
]
