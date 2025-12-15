"""
Snake game configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SnakeConfig:
    """Configuration for Snake game."""

    # Grid dimensions
    grid_width: int = 20
    grid_height: int = 20

    # Rewards
    reward_food: float = 10.0
    reward_death: float = -10.0
    reward_step_penalty: float = -0.01
    reward_approach_food: float = 0.0
    reward_retreat_food: float = 0.0
    reward_length_bonus_factor: float = 0.0

    def get_reward_config(self) -> Dict[str, float]:
        """Get reward configuration dictionary."""
        return {
            "food": self.reward_food,
            "death": self.reward_death,
            "step_penalty": self.reward_step_penalty,
            "approach_food": self.reward_approach_food,
            "retreat_food": self.reward_retreat_food,
            "length_bonus_factor": self.reward_length_bonus_factor,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "rewards": self.get_reward_config(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnakeConfig":
        """Create config from dictionary."""
        rewards = data.get("rewards", {})
        return cls(
            grid_width=data.get("grid_width", 20),
            grid_height=data.get("grid_height", 20),
            reward_food=rewards.get("food", 10.0),
            reward_death=rewards.get("death", -10.0),
            reward_step_penalty=rewards.get("step_penalty", -0.01),
            reward_approach_food=rewards.get("approach_food", 0.0),
            reward_retreat_food=rewards.get("retreat_food", 0.0),
            reward_length_bonus_factor=rewards.get("length_bonus_factor", 0.0),
        )
