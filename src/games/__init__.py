"""
Games module for Novelty AI.

This module auto-discovers and registers all available games.
Import this module to populate the GameRegistry.
"""

from .registry import GameRegistry, register_game
from ..core.game_interface import GameMetadata

# Register placeholder games for "Coming Soon" display
_placeholders = [
    GameMetadata(
        name="Tetris",
        id="tetris",
        description="Clear lines by arranging falling pieces",
        supports_human=True,
        recommended_algorithms=["dqn", "ppo"]
    ),
    GameMetadata(
        name="Pong",
        id="pong",
        description="Classic paddle and ball arcade game",
        supports_human=True,
        recommended_algorithms=["dqn", "ppo"]
    ),
]

for placeholder in _placeholders:
    GameRegistry.register_placeholder(placeholder)

# Import game modules to trigger registration
# Each game's __init__.py calls GameRegistry.register()
from . import snake
from . import space_invaders

__all__ = [
    'GameRegistry',
    'register_game',
]
