"""
Space Invaders game configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SpaceInvadersConfig:
    """Configuration for Space Invaders game."""

    # Screen dimensions (classic arcade resolution)
    width: int = 224
    height: int = 256

    # Player settings (classic arcade speeds)
    player_speed: float = 1.0  # Classic: 1px/frame
    player_bullet_speed: float = 4.0  # Classic: 4px/frame
    player_start_lives: int = 3

    # Invader settings (one alien moves per frame)
    invader_rows: int = 5
    invader_cols: int = 11
    march_step: int = 2  # Classic: 2px per alien step
    invader_bullet_speed: float = 1.0  # Classic: 1px/frame
    invader_drop_distance: float = 8.0  # Classic: 8px per drop
    base_fire_cooldown: int = 60  # Frames between invader shots
    max_invader_projectiles: int = 3

    # Wave progression
    wave_start_offset: float = 16.0  # Each wave starts this many pixels lower

    # Mystery ship
    mystery_ship_speed: float = 2.0
    mystery_ship_spawn_interval: int = 600  # Frames between spawn chances

    # Rewards
    reward_kill_bottom: float = 1.0  # 10-point invaders
    reward_kill_middle: float = 2.0  # 20-point invaders
    reward_kill_top: float = 3.0  # 30-point invaders
    reward_mystery_ship: float = 5.0
    reward_wave_clear: float = 20.0
    reward_death: float = -10.0
    reward_game_over: float = -50.0
    reward_step_penalty: float = -0.001
    reward_survival_bonus: float = 0.01
    reward_edge_priority: float = 0.2

    def get_reward_config(self) -> Dict[str, float]:
        """Get reward configuration dictionary."""
        return {
            "kill_bottom": self.reward_kill_bottom,
            "kill_middle": self.reward_kill_middle,
            "kill_top": self.reward_kill_top,
            "mystery_ship": self.reward_mystery_ship,
            "wave_clear": self.reward_wave_clear,
            "death": self.reward_death,
            "game_over": self.reward_game_over,
            "step_penalty": self.reward_step_penalty,
            "survival_bonus": self.reward_survival_bonus,
            "edge_priority": self.reward_edge_priority,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "player_speed": self.player_speed,
            "player_bullet_speed": self.player_bullet_speed,
            "player_start_lives": self.player_start_lives,
            "invader_rows": self.invader_rows,
            "invader_cols": self.invader_cols,
            "march_step": self.march_step,
            "invader_bullet_speed": self.invader_bullet_speed,
            "invader_drop_distance": self.invader_drop_distance,
            "base_fire_cooldown": self.base_fire_cooldown,
            "max_invader_projectiles": self.max_invader_projectiles,
            "wave_start_offset": self.wave_start_offset,
            "mystery_ship_speed": self.mystery_ship_speed,
            "mystery_ship_spawn_interval": self.mystery_ship_spawn_interval,
            "rewards": self.get_reward_config(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpaceInvadersConfig":
        """Create config from dictionary."""
        rewards = data.get("rewards", {})
        return cls(
            width=data.get("width", 224),
            height=data.get("height", 256),
            player_speed=data.get("player_speed", 1.0),
            player_bullet_speed=data.get("player_bullet_speed", 4.0),
            player_start_lives=data.get("player_start_lives", 3),
            invader_rows=data.get("invader_rows", 5),
            invader_cols=data.get("invader_cols", 11),
            march_step=data.get("march_step", 2),
            invader_bullet_speed=data.get("invader_bullet_speed", 1.0),
            invader_drop_distance=data.get("invader_drop_distance", 8.0),
            base_fire_cooldown=data.get("base_fire_cooldown", 60),
            max_invader_projectiles=data.get("max_invader_projectiles", 3),
            wave_start_offset=data.get("wave_start_offset", 16.0),
            mystery_ship_speed=data.get("mystery_ship_speed", 2.0),
            mystery_ship_spawn_interval=data.get("mystery_ship_spawn_interval", 600),
            reward_kill_bottom=rewards.get("kill_bottom", 1.0),
            reward_kill_middle=rewards.get("kill_middle", 2.0),
            reward_kill_top=rewards.get("kill_top", 3.0),
            reward_mystery_ship=rewards.get("mystery_ship", 5.0),
            reward_wave_clear=rewards.get("wave_clear", 20.0),
            reward_death=rewards.get("death", -10.0),
            reward_game_over=rewards.get("game_over", -50.0),
            reward_step_penalty=rewards.get("step_penalty", -0.001),
            reward_survival_bonus=rewards.get("survival_bonus", 0.01),
            reward_edge_priority=rewards.get("edge_priority", 0.2),
        )
