"""
Tetris game configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TetrisConfig:
    """Configuration for Tetris game."""

    # Board dimensions (standard Tetris)
    board_width: int = 10
    board_height: int = 20

    # Preview and hold
    preview_count: int = 5  # Number of next pieces shown

    # Rewards
    reward_single: float = 100.0      # 1 line clear
    reward_double: float = 300.0      # 2 lines clear
    reward_triple: float = 500.0      # 3 lines clear
    reward_tetris: float = 800.0      # 4 lines clear (Tetris!)
    reward_game_over: float = -100.0  # Game over penalty
    reward_step: float = 0.01         # Small reward for surviving
    reward_height_penalty: float = -0.1  # Penalty per max height

    def get_reward_config(self) -> Dict[str, float]:
        """Get reward configuration dictionary."""
        return {
            "single": self.reward_single,
            "double": self.reward_double,
            "triple": self.reward_triple,
            "tetris": self.reward_tetris,
            "game_over": self.reward_game_over,
            "step": self.reward_step,
            "height_penalty": self.reward_height_penalty,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "board_width": self.board_width,
            "board_height": self.board_height,
            "preview_count": self.preview_count,
            "rewards": self.get_reward_config(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TetrisConfig":
        """Create config from dictionary."""
        rewards = data.get("rewards", {})
        return cls(
            board_width=data.get("board_width", 10),
            board_height=data.get("board_height", 20),
            preview_count=data.get("preview_count", 5),
            reward_single=rewards.get("single", 100.0),
            reward_double=rewards.get("double", 300.0),
            reward_triple=rewards.get("triple", 500.0),
            reward_tetris=rewards.get("tetris", 800.0),
            reward_game_over=rewards.get("game_over", -100.0),
            reward_step=rewards.get("step", 0.01),
            reward_height_penalty=rewards.get("height_penalty", -0.1),
        )
