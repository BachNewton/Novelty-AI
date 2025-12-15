"""
DQN (Deep Q-Network) algorithm module.

This module auto-registers the DQN algorithm when imported.
"""

from ..registry import AlgorithmRegistry
from .agent import DQNAgent
from .network import DQNNetwork

# Auto-register DQN when this module is imported
AlgorithmRegistry.register(
    algorithm_id="dqn",
    agent_class=DQNAgent,
    config_class=None,  # Uses dict-based config
    description="Deep Q-Network with Double DQN and experience replay"
)

__all__ = [
    'DQNAgent',
    'DQNNetwork',
]
