"""
Tetris game module for Novelty AI.

This module auto-registers the Tetris game when imported.
"""

from ..registry import GameRegistry
from .game import TetrisGame, PieceType, Action, Piece
from .env import TetrisEnv
from .renderer import TetrisRenderer
from .config import TetrisConfig

# Auto-register Tetris game when this module is imported
GameRegistry.register(
    game_class=TetrisGame,
    env_class=TetrisEnv,
    renderer_class=TetrisRenderer,
    config_class=TetrisConfig
)

__all__ = [
    'TetrisGame',
    'TetrisEnv',
    'TetrisRenderer',
    'TetrisConfig',
    'PieceType',
    'Action',
    'Piece',
]
