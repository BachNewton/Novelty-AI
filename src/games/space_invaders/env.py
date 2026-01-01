"""
Space Invaders Environment - Gym-like wrapper implementing EnvInterface.
Provides 32-dimensional state encoding suitable for neural network input.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional

from ...core.env_interface import EnvInterface
from .game import SpaceInvadersGame, InvaderType
from .config import SpaceInvadersConfig


class SpaceInvadersEnv(EnvInterface):
    """
    Gym-like environment wrapper for the Space Invaders game implementing EnvInterface.

    Provides a clean interface for reinforcement learning:
    - reset() -> initial state
    - step(action) -> (next_state, reward, done, info)

    State is a 32-dimensional feature vector encoding player position, danger zones,
    invader positions, bunker health, and strategic features.
    """

    def __init__(
        self,
        width: int = 224,
        height: int = 256,
        reward_config: Optional[Dict[str, float]] = None,
        config: Optional[SpaceInvadersConfig] = None,
    ):
        """
        Initialize the environment.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels
            reward_config: Optional reward configuration dictionary
            config: Optional SpaceInvadersConfig object
        """
        game_config = {}
        if config:
            game_config = config.to_dict()
            # Remove rewards from game_config since we pass it separately
            game_config.pop("rewards", None)
            if reward_config is None:
                reward_config = config.get_reward_config()

        self.game = SpaceInvadersGame(
            width=width,
            height=height,
            reward_config=reward_config,
            config=game_config,
        )
        self.width = width
        self.height = height
        self._reward_config = reward_config
        self.frames_since_shot = 0

    @property
    def state_size(self) -> int:
        """Get the state size (32 features)."""
        return 32

    @property
    def action_size(self) -> int:
        """Get the action size (6 actions: movement x firing combinations)."""
        return 6

    def reset(self, record: bool = False) -> np.ndarray:
        """
        Reset environment and return initial state.

        Args:
            record: If True, start recording for replay

        Returns:
            Initial state as numpy array
        """
        self.game.reset()
        self.frames_since_shot = 0
        if record:
            self.game.start_recording()
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return results.

        Args:
            action: 0-5 representing movement and firing combinations

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Track firing for feature
        fire = action in [1, 3, 5]  # STAY_FIRE, LEFT_FIRE, RIGHT_FIRE
        if fire and self.game.player_projectile is None:
            self.frames_since_shot = 0
        else:
            self.frames_since_shot += 1

        _, reward, done, info = self.game.step(action)
        state = self._get_state()
        return state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """
        Get 32-dimensional state vector.

        Features:
        [0]:     Player X position (normalized 0-1)
        [1]:     Can fire (no active projectile) (0/1)
        [2-4]:   Danger from projectiles (left/center/right zones) (0/1)
        [5-7]:   Nearest projectile distance per zone (0-1)
        [8-10]:  Nearest invader X offset per zone (-1 to 1)
        [11]:    Invaders alive ratio (0-1)
        [12]:    Formation Y position / threat level (0-1)
        [13]:    Formation direction (0=left, 1=right)
        [14]:    Formation speed normalized (0-1)
        [15]:    Mystery ship active (0/1)
        [16]:    Mystery ship X position (0-1)
        [17-20]: Bunker health ratios (0-1)
        [21-23]: Nearest invader Y distance per zone (0-1)
        [24]:    Time since last shot normalized (0-1)
        [25]:    Player projectile active (0/1)
        [26]:    Player projectile Y position (0-1)
        [27]:    Shooting gap score (0-1)
        [28]:    Fire threat score (0-1)
        [29]:    Leftmost invader X (0-1)
        [30]:    Rightmost invader X (0-1)
        [31]:    Player-bunker proximity (0-1)

        Returns:
            State as numpy array of shape (32,)
        """
        game = self.game
        zone_width = game.width / 3

        # [0] Player X position normalized
        player_x_norm = game.player.x / game.width

        # [1] Can fire (no active projectile)
        can_fire = 1.0 if game.player_projectile is None else 0.0

        # [2-4] Danger from projectiles per zone
        danger_left = self._get_projectile_danger(0, zone_width)
        danger_center = self._get_projectile_danger(zone_width, 2 * zone_width)
        danger_right = self._get_projectile_danger(2 * zone_width, game.width)

        # [5-7] Nearest projectile distance per zone
        proj_dist_left = self._get_nearest_projectile_distance(0, zone_width)
        proj_dist_center = self._get_nearest_projectile_distance(
            zone_width, 2 * zone_width
        )
        proj_dist_right = self._get_nearest_projectile_distance(
            2 * zone_width, game.width
        )

        # [8-10] Nearest invader X offset per zone
        inv_x_left = self._get_nearest_invader_x_offset(0, zone_width)
        inv_x_center = self._get_nearest_invader_x_offset(zone_width, 2 * zone_width)
        inv_x_right = self._get_nearest_invader_x_offset(2 * zone_width, game.width)

        # [11] Invaders alive ratio
        total = game.invader_rows * game.invader_cols
        alive_ratio = game._get_alive_count() / total if total > 0 else 0.0

        # [12] Formation Y position (threat level)
        formation_y = game._get_formation_bottom_y() / game.height

        # [13] Formation direction (0=left, 1=right)
        formation_dir = 1.0 if game.invader_direction > 0 else 0.0

        # [14] Formation speed normalized (inversely proportional to alive count)
        # With 55 aliens = slow (0.0), with 1 alien = fast (1.0)
        alive_count = game._get_alive_count()
        total = game.invader_rows * game.invader_cols
        speed_norm = 1.0 - (alive_count / total) if total > 0 else 0.0

        # [15] Mystery ship active
        mystery_active = 1.0 if game.mystery_ship.active else 0.0

        # [16] Mystery ship X position
        mystery_x = (
            game.mystery_ship.x / game.width if game.mystery_ship.active else 0.5
        )
        mystery_x = max(0.0, min(1.0, mystery_x))

        # [17-20] Bunker health ratios
        bunker_health = [self._get_bunker_health_ratio(b) for b in game.bunkers]
        # Pad if fewer than 4 bunkers
        while len(bunker_health) < 4:
            bunker_health.append(0.0)

        # [21-23] Nearest invader Y distance per zone
        inv_y_left = self._get_nearest_invader_y_distance(0, zone_width)
        inv_y_center = self._get_nearest_invader_y_distance(zone_width, 2 * zone_width)
        inv_y_right = self._get_nearest_invader_y_distance(2 * zone_width, game.width)

        # [24] Time since last shot normalized
        time_since_shot = min(1.0, self.frames_since_shot / 30.0)

        # [25] Player projectile active
        proj_active = 1.0 if game.player_projectile else 0.0

        # [26] Player projectile Y position
        proj_y = (
            game.player_projectile.y / game.height if game.player_projectile else 1.0
        )

        # [27] Shooting gap score
        gap_score = self._find_shooting_gap()

        # [28] Fire threat score
        fire_threat = self._calculate_fire_threat()

        # [29-30] Leftmost and rightmost invader X
        left_edge, right_edge = game._get_formation_edges()
        leftmost_x = left_edge / game.width
        rightmost_x = right_edge / game.width

        # [31] Player-bunker proximity (how close player is to nearest bunker)
        bunker_proximity = self._get_player_bunker_proximity()

        state = np.array(
            [
                player_x_norm,  # 0
                can_fire,  # 1
                danger_left,  # 2
                danger_center,  # 3
                danger_right,  # 4
                proj_dist_left,  # 5
                proj_dist_center,  # 6
                proj_dist_right,  # 7
                inv_x_left,  # 8
                inv_x_center,  # 9
                inv_x_right,  # 10
                alive_ratio,  # 11
                formation_y,  # 12
                formation_dir,  # 13
                speed_norm,  # 14
                mystery_active,  # 15
                mystery_x,  # 16
                bunker_health[0],  # 17
                bunker_health[1],  # 18
                bunker_health[2],  # 19
                bunker_health[3],  # 20
                inv_y_left,  # 21
                inv_y_center,  # 22
                inv_y_right,  # 23
                time_since_shot,  # 24
                proj_active,  # 25
                proj_y,  # 26
                gap_score,  # 27
                fire_threat,  # 28
                leftmost_x,  # 29
                rightmost_x,  # 30
                bunker_proximity,  # 31
            ],
            dtype=np.float32,
        )

        return state

    def _get_projectile_danger(self, x_min: float, x_max: float) -> float:
        """Returns 1.0 if any invader projectile is in danger zone near player."""
        game = self.game
        danger_y_threshold = game.height * 0.3  # Bottom 30% is danger zone

        for proj in game.invader_projectiles:
            if (
                x_min <= proj.x <= x_max
                and proj.y >= (game.height - danger_y_threshold)
            ):
                return 1.0
        return 0.0

    def _get_nearest_projectile_distance(self, x_min: float, x_max: float) -> float:
        """Normalized distance to nearest invader projectile in zone."""
        game = self.game
        min_dist = 1.0

        for proj in game.invader_projectiles:
            if x_min <= proj.x <= x_max:
                # Distance from bottom of screen
                dist = (game.height - proj.y) / game.height
                min_dist = min(min_dist, dist)

        return min_dist

    def _get_nearest_invader_x_offset(self, x_min: float, x_max: float) -> float:
        """X offset of nearest invader in zone, relative to player, normalized."""
        game = self.game
        player_x = game.player.x
        nearest_offset = 0.0
        min_y_dist = float("inf")

        for row in game.invaders:
            for inv in row:
                if inv.alive and x_min <= inv.x <= x_max:
                    # Find the closest invader vertically
                    y_dist = inv.y
                    if y_dist < min_y_dist:
                        min_y_dist = y_dist
                        # Normalize offset to [-1, 1]
                        nearest_offset = (inv.x - player_x) / game.width

        return max(-1.0, min(1.0, nearest_offset))

    def _get_nearest_invader_y_distance(self, x_min: float, x_max: float) -> float:
        """Normalized Y distance to nearest invader in zone."""
        game = self.game
        max_y = 0.0

        for row in game.invaders:
            for inv in row:
                if inv.alive and x_min <= inv.x <= x_max:
                    # Track the lowest (closest to player) invader
                    max_y = max(max_y, inv.y)

        # Normalize: 0 = at top, 1 = at player level
        return max_y / game.height if max_y > 0 else 0.0

    def _get_bunker_health_ratio(self, bunker: Any) -> float:
        """Calculate remaining health as ratio of maximum."""
        if not bunker.cells:
            return 0.0

        max_health = 0
        current_health = 0

        for row in bunker.cells:
            for cell in row:
                if cell is not None:
                    max_health += 4  # Max health per cell
                    current_health += cell.health

        return current_health / max_health if max_health > 0 else 0.0

    def _find_shooting_gap(self) -> float:
        """Find if there's a good shooting opportunity (gap above player)."""
        game = self.game
        player_x = game.player.x

        # Check if there's a clear path to an invader
        has_target = False
        for row in game.invaders:
            for inv in row:
                if inv.alive and abs(inv.x - player_x) < 20:
                    has_target = True
                    break
            if has_target:
                break

        return 1.0 if has_target else 0.0

    def _calculate_fire_threat(self) -> float:
        """Estimate probability of being hit if staying still."""
        game = self.game
        threat = 0.0
        player_x = game.player.x

        for proj in game.invader_projectiles:
            # Check if projectile is aligned with player
            if abs(proj.x - player_x) < game.player.width:
                # Closer = more threat
                dist_ratio = (game.height - proj.y) / game.height
                threat = max(threat, 1.0 - dist_ratio)

        return threat

    def _get_player_bunker_proximity(self) -> float:
        """Get how close the player is to the nearest bunker (for cover)."""
        game = self.game
        player_x = game.player.x
        min_dist = float("inf")

        for bunker in game.bunkers:
            # Bunker center X
            bunker_width = len(bunker.cells[0]) * 4 if bunker.cells else 32
            bunker_center_x = bunker.x + bunker_width / 2
            dist = abs(player_x - bunker_center_x)
            min_dist = min(min_dist, dist)

        # Normalize: 0 = far from bunker, 1 = at bunker
        max_dist = game.width / 2
        return 1.0 - min(1.0, min_dist / max_dist)

    def get_game_state(self) -> Dict[str, Any]:
        """
        Get raw game state for visualization.

        Returns:
            Dictionary containing full game state
        """
        return self.game.get_state()

    def get_replay(self) -> List[Dict[str, Any]]:
        """
        Get the recorded game history.

        Returns:
            List of game state dictionaries
        """
        return self.game.stop_recording()

    def get_score(self) -> int:
        """Get current game score."""
        return self.game.score

    def is_recording(self) -> bool:
        """Check if game is being recorded."""
        return self.game.recording

    def render(self) -> None:
        """Render current state (no-op, use renderer instead)."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        if seed is not None:
            import random

            random.seed(seed)
