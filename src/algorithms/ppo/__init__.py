"""
PPO (Proximal Policy Optimization) algorithm module.

This module auto-registers the PPO algorithm when imported.
"""

from ..registry import AlgorithmRegistry
from .agent import PPOAgent, ActorCriticNetwork
from .buffer import RolloutBuffer, RolloutBufferSamples

# Auto-register PPO when this module is imported
AlgorithmRegistry.register(
    algorithm_id="ppo",
    agent_class=PPOAgent,
    config_class=None,  # Uses dict-based config
    description="Proximal Policy Optimization with GAE and clipped objective"
)

__all__ = [
    'PPOAgent',
    'ActorCriticNetwork',
    'RolloutBuffer',
    'RolloutBufferSamples',
]
