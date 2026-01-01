"""
Space Invaders game module for Novelty AI.

This module auto-registers the Space Invaders game when imported.
"""

from ..registry import GameRegistry
from .game import SpaceInvadersGame, InvaderType, ProjectileOwner, Action
from .env import SpaceInvadersEnv
from .renderer import SpaceInvadersRenderer
from .config import SpaceInvadersConfig

# Auto-register Space Invaders game when this module is imported
GameRegistry.register(
    game_class=SpaceInvadersGame,
    env_class=SpaceInvadersEnv,
    renderer_class=SpaceInvadersRenderer,
    config_class=SpaceInvadersConfig,
)

__all__ = [
    "SpaceInvadersGame",
    "SpaceInvadersEnv",
    "SpaceInvadersRenderer",
    "SpaceInvadersConfig",
    "InvaderType",
    "ProjectileOwner",
    "Action",
]
