"""
Snake game module for Novelty AI.

This module auto-registers the Snake game when imported.
"""

from ..registry import GameRegistry
from .game import SnakeGame, Direction, Point
from .env import SnakeEnv
from .renderer import SnakeRenderer
from .config import SnakeConfig

# Auto-register Snake game when this module is imported
GameRegistry.register(
    game_class=SnakeGame,
    env_class=SnakeEnv,
    renderer_class=SnakeRenderer,
    config_class=SnakeConfig
)

__all__ = [
    'SnakeGame',
    'SnakeEnv',
    'SnakeRenderer',
    'SnakeConfig',
    'Direction',
    'Point',
]
