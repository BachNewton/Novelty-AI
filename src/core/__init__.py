"""
Core abstractions for Novelty AI.

Provides abstract interfaces that all games, environments, agents, and renderers must implement.
"""

from .game_interface import GameInterface, GameMetadata
from .env_interface import EnvInterface
from .agent_interface import AgentInterface
from .renderer_interface import RendererInterface

__all__ = [
    'GameInterface',
    'GameMetadata',
    'EnvInterface',
    'AgentInterface',
    'RendererInterface',
]
