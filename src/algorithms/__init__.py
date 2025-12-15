"""
Algorithms module for Novelty AI.

This module auto-discovers and registers all available learning algorithms.
Import this module to populate the AlgorithmRegistry.
"""

from .registry import AlgorithmRegistry

# Import algorithm modules to trigger registration
from . import dqn

__all__ = [
    'AlgorithmRegistry',
]
