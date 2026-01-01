"""
Space Invaders Game Core - Pure game logic implementing GameInterface.
Designed for high-speed training with optional visualization.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import IntEnum
import random

from ...core.game_interface import GameInterface, GameMetadata


class InvaderType(IntEnum):
    """Types of invaders with different point values."""
    TOP = 0      # 30 points - small invader (1 row)
    MIDDLE = 1   # 20 points - medium invader (2 rows)
    BOTTOM = 2   # 10 points - large invader (2 rows)


class ProjectileOwner(IntEnum):
    """Identifies who owns a projectile."""
    PLAYER = 0
    INVADER = 1


class Action(IntEnum):
    """Actions combining movement and firing."""
    STAY_NO_FIRE = 0
    STAY_FIRE = 1
    LEFT_NO_FIRE = 2
    LEFT_FIRE = 3
    RIGHT_NO_FIRE = 4
    RIGHT_FIRE = 5


@dataclass
class Player:
    """The player's laser cannon."""
    x: float
    y: float
    width: int = 13  # Classic: 13×8
    height: int = 8
    lives: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "lives": self.lives,
        }


@dataclass
class Invader:
    """An individual invader alien.

    Classic dimensions by type:
    - Squid (TOP): 8×8
    - Crab (MIDDLE): 11×8
    - Octopus (BOTTOM): 12×8
    """
    x: float
    y: float
    invader_type: InvaderType
    alive: bool = True
    width: int = 12  # Default, overridden per type
    height: int = 8  # Classic: all 8px tall

    def get_points(self) -> int:
        """Return point value for this invader type."""
        return {
            InvaderType.TOP: 30,
            InvaderType.MIDDLE: 20,
            InvaderType.BOTTOM: 10,
        }[self.invader_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "type": int(self.invader_type),
            "alive": self.alive,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Projectile:
    """A bullet fired by player or invader."""
    x: float
    y: float
    owner: ProjectileOwner
    speed: float
    width: int = 1   # Classic: 1×4
    height: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "owner": int(self.owner),
            "width": self.width,
            "height": self.height,
        }


@dataclass
class BunkerCell:
    """A single destructible cell of a bunker."""
    x: int
    y: int
    health: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "health": self.health}


@dataclass
class Bunker:
    """A defensive bunker made of destructible cells."""
    x: float
    y: float
    cells: List[List[Optional[BunkerCell]]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "cells": [
                [cell.to_dict() if cell and cell.health > 0 else None for cell in row]
                for row in self.cells
            ],
        }


@dataclass
class MysteryShip:
    """Bonus UFO that crosses the screen."""
    x: float
    y: float
    active: bool = False
    direction: int = 1  # 1 = right, -1 = left
    speed: float = 2.0
    width: int = 16  # Classic: 16×8
    height: int = 8

    def get_points(self) -> int:
        """Mystery ship gives 50, 100, 150, or 300 points randomly."""
        return random.choice([50, 100, 150, 300])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "active": self.active,
            "direction": self.direction,
            "width": self.width,
            "height": self.height,
        }


class SpaceInvadersGame(GameInterface):
    """
    Core Space Invaders game logic implementing GameInterface.

    The player controls a laser cannon at the bottom of the screen, shooting
    at waves of descending alien invaders. Invaders move as a formation,
    dropping down when they reach screen edges.
    """

    @classmethod
    def get_metadata(cls) -> GameMetadata:
        """Return metadata about Space Invaders game."""
        return GameMetadata(
            name="Space Invaders",
            id="space_invaders",
            description="Classic arcade shooter - destroy alien invaders before they reach Earth",
            version="1.0.0",
            min_players=1,
            max_players=1,
            supports_human=True,
            recommended_algorithms=["dqn"],
        )

    def __init__(
        self,
        width: int = 224,
        height: int = 256,
        reward_config: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the game.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels
            reward_config: Optional reward configuration dictionary
            config: Optional full configuration dictionary
        """
        self.width = width
        self.height = height

        # Game settings from config (classic arcade values)
        config = config or {}
        self.player_speed = config.get("player_speed", 1.0)  # Classic: 1px/frame
        self.player_bullet_speed = config.get("player_bullet_speed", 4.0)  # Classic: 4px/frame
        self.player_start_lives = config.get("player_start_lives", 3)
        self.invader_rows = config.get("invader_rows", 5)
        self.invader_cols = config.get("invader_cols", 11)
        self.march_step = config.get("march_step", 2)  # Classic: 2px per alien step
        self.invader_bullet_speed = config.get("invader_bullet_speed", 1.0)  # Classic: 1px/frame
        self.invader_drop_distance = config.get("invader_drop_distance", 8.0)  # Classic: 8 pixels
        self.base_fire_cooldown = config.get("base_fire_cooldown", 60)
        self.max_invader_projectiles = config.get("max_invader_projectiles", 3)
        self.wave_start_offset = config.get("wave_start_offset", 16.0)
        self.mystery_ship_speed = config.get("mystery_ship_speed", 2.0)
        self.mystery_ship_spawn_interval = config.get("mystery_ship_spawn_interval", 600)

        # Reward configuration with defaults
        reward_config = reward_config or {}
        self.reward_kill_bottom = reward_config.get("kill_bottom", 1.0)
        self.reward_kill_middle = reward_config.get("kill_middle", 2.0)
        self.reward_kill_top = reward_config.get("kill_top", 3.0)
        self.reward_mystery_ship = reward_config.get("mystery_ship", 5.0)
        self.reward_wave_clear = reward_config.get("wave_clear", 20.0)
        self.reward_death = reward_config.get("death", -10.0)
        self.reward_game_over = reward_config.get("game_over", -50.0)
        self.reward_step_penalty = reward_config.get("step_penalty", -0.001)
        self.reward_survival_bonus = reward_config.get("survival_bonus", 0.01)
        self.reward_edge_priority = reward_config.get("edge_priority", 0.2)

        # Layout constants (classic arcade values for 224x256 resolution)
        self.edge_margin = 8  # Boundaries at X=8 and X=216
        self.invader_spacing_x = 16
        self.invader_spacing_y = 16
        self.bunker_y = 152  # Classic: bunkers at Y=152
        self.player_y = 216  # Classic: player at Y=216
        self.ufo_y = 32      # Classic: UFO at Y=32
        self.game_over_y = 216  # Aliens reaching this Y = game over
        # Classic bunker X positions
        self.bunker_positions = [32, 72, 112, 152]

        # Game state (initialized in reset)
        self.player: Player = Player(0, 0)
        self.invaders: List[List[Invader]] = []
        self.invader_direction: int = 1
        self.march_index: int = 0  # Which alien is being moved this frame
        self.player_projectile: Optional[Projectile] = None
        self.invader_projectiles: List[Projectile] = []
        self.bunkers: List[Bunker] = []
        self.mystery_ship: MysteryShip = MysteryShip(0, 0)
        self.score: int = 0
        self.wave: int = 1
        self.frame_count: int = 0
        self.game_over: bool = False
        self.invader_fire_cooldown: int = 0
        self.mystery_ship_timer: int = 0
        self.aimed_shot_fired: bool = False

        # For replay recording
        self.history: List[Dict[str, Any]] = []
        self.recording: bool = False

        self.reset()

    @property
    def action_space_size(self) -> int:
        """Number of possible actions (6 combinations of movement and firing)."""
        return 6

    @property
    def action_names(self) -> List[str]:
        """Human-readable action names."""
        return [
            "Stay",
            "Stay+Fire",
            "Left",
            "Left+Fire",
            "Right",
            "Right+Fire",
        ]

    def reset(self) -> Dict[str, Any]:
        """
        Reset game state and return initial state.

        Returns:
            Dictionary containing the initial game state
        """
        # Reset player
        self.player = Player(
            x=self.width / 2,
            y=self.player_y,
            lives=self.player_start_lives,
        )

        # Reset wave
        self.wave = 1
        self.score = 0
        self.frame_count = 0
        self.game_over = False

        # Create invader formation
        self._create_invaders(wave_offset=0)

        # Reset projectiles
        self.player_projectile = None
        self.invader_projectiles = []
        self.invader_fire_cooldown = self.base_fire_cooldown
        self.aimed_shot_fired = False

        # Create bunkers
        self._create_bunkers()

        # Reset mystery ship
        self.mystery_ship = MysteryShip(
            x=-50,
            y=self.ufo_y,  # Classic: Y=32
            active=False,
            speed=self.mystery_ship_speed,
        )
        self.mystery_ship_timer = self.mystery_ship_spawn_interval

        # Reset history if recording
        self.history = []
        if self.recording:
            self._record_frame()

        return self.get_state()

    def _create_invaders(self, wave_offset: float = 0) -> None:
        """Create the invader formation."""
        self.invaders = []
        # Classic arcade: leftmost column at X=24, 16px spacing
        start_x = 24 + self.invader_spacing_x / 2  # Center of first alien
        # Classic arcade: top row at Y=72 for wave 1
        start_y = 72 + wave_offset

        # Classic sprite widths by type
        type_widths = {
            InvaderType.TOP: 8,      # Squid: 8×8
            InvaderType.MIDDLE: 11,  # Crab: 11×8
            InvaderType.BOTTOM: 12,  # Octopus: 12×8
        }

        for row in range(self.invader_rows):
            invader_row: List[Invader] = []
            # Determine invader type based on row
            if row == 0:
                inv_type = InvaderType.TOP
            elif row <= 2:
                inv_type = InvaderType.MIDDLE
            else:
                inv_type = InvaderType.BOTTOM

            inv_width = type_widths[inv_type]

            for col in range(self.invader_cols):
                x = start_x + col * self.invader_spacing_x
                y = start_y + row * self.invader_spacing_y
                invader_row.append(Invader(x=x, y=y, invader_type=inv_type, width=inv_width))

            self.invaders.append(invader_row)

        self.invader_direction = 1
        self.march_index = 0  # Reset march for new formation

    def _create_bunkers(self) -> None:
        """Create the four defensive bunkers at classic positions."""
        self.bunkers = []
        bunker_width = 22   # Classic: 22×16
        bunker_height = 16
        cell_size = 2       # 2px cells for 11×8 grid

        for bunker_x in self.bunker_positions:
            bunker = Bunker(x=bunker_x, y=self.bunker_y)

            # Create cell grid with arch shape
            rows = bunker_height // cell_size  # 8 rows
            cols = bunker_width // cell_size   # 11 cols
            cells: List[List[Optional[BunkerCell]]] = []

            for row in range(rows):
                cell_row: List[Optional[BunkerCell]] = []
                for col in range(cols):
                    # Create arch shape - remove bottom center cells
                    is_arch = (
                        row >= rows - 2
                        and col >= cols // 2 - 1
                        and col <= cols // 2 + 1
                    )
                    if is_arch:
                        cell_row.append(None)
                    else:
                        cell_row.append(BunkerCell(x=col, y=row))
                cells.append(cell_row)

            bunker.cells = cells
            self.bunkers.append(bunker)

    def _get_movement_from_action(self, action: int) -> int:
        """Get movement direction from action."""
        if action in [Action.LEFT_NO_FIRE, Action.LEFT_FIRE]:
            return -1
        elif action in [Action.RIGHT_NO_FIRE, Action.RIGHT_FIRE]:
            return 1
        return 0

    def _get_fire_from_action(self, action: int) -> bool:
        """Get fire decision from action."""
        return action in [Action.STAY_FIRE, Action.LEFT_FIRE, Action.RIGHT_FIRE]

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one game step.

        Args:
            action: 0-5 representing movement and firing combinations

        Returns:
            Tuple of (state, reward, done, info)
        """
        self.frame_count += 1
        reward = self.reward_step_penalty + self.reward_survival_bonus

        # Parse action
        move = self._get_movement_from_action(action)
        fire = self._get_fire_from_action(action)

        # Move player (clamp to screen bounds)
        self.player.x += move * self.player_speed
        self.player.x = max(
            self.player.width / 2,
            min(self.width - self.player.width / 2, self.player.x),
        )

        # Player fires (if no projectile active)
        if fire and self.player_projectile is None:
            self.player_projectile = Projectile(
                x=self.player.x,
                y=self.player.y - self.player.height / 2,
                owner=ProjectileOwner.PLAYER,
                speed=-self.player_bullet_speed,
            )

        # Move player projectile
        if self.player_projectile:
            self.player_projectile.y += self.player_projectile.speed
            if self.player_projectile.y < 0:
                self.player_projectile = None

        # Check player projectile collisions
        reward += self._check_player_projectile_collisions()

        # Move invader formation
        self._move_invaders()

        # Check if invaders are destroying bunkers (deletion mask behavior)
        self._check_invader_bunker_collision()

        # Invaders fire
        self._invaders_fire()

        # Move invader projectiles
        for proj in self.invader_projectiles[:]:
            proj.y += proj.speed
            if proj.y > self.height:
                self.invader_projectiles.remove(proj)

        # Check bullet vs bullet collision
        self._check_bullet_collision()

        # Check invader projectile collisions
        hit_player = self._check_invader_projectile_collisions()
        if hit_player:
            reward += self.reward_death

        # Update mystery ship
        reward += self._update_mystery_ship()

        # Check game over conditions
        if self.player.lives <= 0:
            self.game_over = True
            reward += self.reward_game_over

        # Check if invaders reached bottom (classic: Y=216 triggers game over)
        bottom_y = self._get_formation_bottom_y()
        if bottom_y >= self.game_over_y:
            self.game_over = True
            reward += self.reward_game_over

        # Check if all invaders destroyed
        if self._get_alive_count() == 0:
            self._start_next_wave()
            reward += self.reward_wave_clear

        if self.recording:
            self._record_frame()

        return self.get_state(), reward, self.game_over, {"score": self.score}

    def _move_invaders(self) -> None:
        """Move ONE invader per frame (classic arcade behavior).

        In the original arcade, the CPU updates one alien per frame.
        With 55 aliens, it takes 55 frames to move the whole grid once.
        As aliens die, the grid moves faster because fewer need updating.
        """
        # Get list of alive invaders
        alive_invaders = [inv for row in self.invaders for inv in row if inv.alive]

        if not alive_invaders:
            return

        # Wrap march_index to valid range
        self.march_index = self.march_index % len(alive_invaders)

        # Check if we need to drop (only at start of new march cycle)
        if self.march_index == 0:
            left_edge, right_edge = self._get_formation_edges()
            hit_edge = (
                (self.invader_direction > 0 and right_edge >= self.width - self.edge_margin) or
                (self.invader_direction < 0 and left_edge <= self.edge_margin)
            )

            if hit_edge:
                # Drop ALL invaders and reverse direction
                for inv in alive_invaders:
                    inv.y += self.invader_drop_distance
                self.invader_direction *= -1

        # Move current alien horizontally
        current_inv = alive_invaders[self.march_index]
        current_inv.x += self.invader_direction * self.march_step
        self.march_index += 1

    def _get_formation_edges(self) -> Tuple[float, float]:
        """Get left and right edges of living invaders."""
        left_edge: float = float(self.width)
        right_edge: float = 0.0

        for row in self.invaders:
            for inv in row:
                if inv.alive:
                    left_edge = min(left_edge, inv.x - inv.width / 2)
                    right_edge = max(right_edge, inv.x + inv.width / 2)

        return left_edge, right_edge

    def _get_formation_bottom_y(self) -> float:
        """Get the Y coordinate of the bottom-most living invader."""
        bottom: float = 0.0
        for row in self.invaders:
            for inv in row:
                if inv.alive:
                    bottom = max(bottom, inv.y + inv.height / 2)
        return bottom

    def _get_alive_count(self) -> int:
        """Count living invaders."""
        return sum(1 for row in self.invaders for inv in row if inv.alive)

    def _get_bottom_invaders_per_column(self) -> List[Optional[Invader]]:
        """Get the bottom-most alive invader in each column."""
        result: List[Optional[Invader]] = []
        for col in range(self.invader_cols):
            bottom: Optional[Invader] = None
            for row in range(self.invader_rows - 1, -1, -1):
                if self.invaders[row][col].alive:
                    bottom = self.invaders[row][col]
                    break
            result.append(bottom)
        return result

    def _invaders_fire(self) -> None:
        """Invaders fire following the specified rules."""
        # Max projectiles on screen
        if len(self.invader_projectiles) >= self.max_invader_projectiles:
            return

        # Decrease cooldown
        self.invader_fire_cooldown -= 1
        if self.invader_fire_cooldown > 0:
            return

        # Reset cooldown (faster as fewer invaders remain)
        alive_count = self._get_alive_count()
        total = self.invader_rows * self.invader_cols
        if total > 0:
            cooldown_factor = max(0.3, alive_count / total)
            self.invader_fire_cooldown = int(self.base_fire_cooldown * cooldown_factor)

        # Get bottom-most invaders per column
        shooters = [s for s in self._get_bottom_invaders_per_column() if s is not None]
        if not shooters:
            return

        # Rule: First shot should aim at player's X position
        if not self.aimed_shot_fired and len(self.invader_projectiles) == 0:
            # Find invader closest to player's X position
            closest = min(shooters, key=lambda inv: abs(inv.x - self.player.x))
            self._fire_from_invader(closest)
            self.aimed_shot_fired = True
        else:
            # Random invader fires
            shooter = random.choice(shooters)
            self._fire_from_invader(shooter)

        # Reset aimed shot flag when all projectiles clear
        if len(self.invader_projectiles) == 0:
            self.aimed_shot_fired = False

    def _fire_from_invader(self, invader: Invader) -> None:
        """Create a projectile from an invader."""
        self.invader_projectiles.append(
            Projectile(
                x=invader.x,
                y=invader.y + invader.height / 2,
                owner=ProjectileOwner.INVADER,
                speed=self.invader_bullet_speed,
            )
        )

    def _rect_collision(
        self,
        x1: float,
        y1: float,
        w1: float,
        h1: float,
        x2: float,
        y2: float,
        w2: float,
        h2: float,
    ) -> bool:
        """AABB collision detection (center-based coordinates)."""
        return abs(x1 - x2) < (w1 + w2) / 2 and abs(y1 - y2) < (h1 + h2) / 2

    def _check_player_projectile_collisions(self) -> float:
        """Check player bullet against invaders, bunkers, mystery ship."""
        reward = 0.0
        if not self.player_projectile:
            return reward

        proj = self.player_projectile

        # Check invaders
        for row in self.invaders:
            for inv in row:
                if inv.alive and self._rect_collision(
                    proj.x,
                    proj.y,
                    proj.width,
                    proj.height,
                    inv.x,
                    inv.y,
                    inv.width,
                    inv.height,
                ):
                    inv.alive = False
                    self.player_projectile = None
                    points = inv.get_points()
                    self.score += points
                    reward = self._calculate_kill_reward(inv)
                    return reward

        # Check mystery ship
        if self.mystery_ship.active and self._rect_collision(
            proj.x,
            proj.y,
            proj.width,
            proj.height,
            self.mystery_ship.x,
            self.mystery_ship.y,
            self.mystery_ship.width,
            self.mystery_ship.height,
        ):
            points = self.mystery_ship.get_points()
            self.score += points
            self.mystery_ship.active = False
            self.player_projectile = None
            reward = self.reward_mystery_ship
            return reward

        # Check bunkers
        for bunker in self.bunkers:
            if self._projectile_hits_bunker(proj, bunker, from_above=False):
                self.player_projectile = None
                return reward

        return reward

    def _check_bullet_collision(self) -> bool:
        """Check if player bullet hits any invader bullet. Both destroyed on collision."""
        if not self.player_projectile:
            return False

        for proj in self.invader_projectiles[:]:
            if self._rect_collision(
                self.player_projectile.x, self.player_projectile.y,
                self.player_projectile.width, self.player_projectile.height,
                proj.x, proj.y, proj.width, proj.height
            ):
                self.player_projectile = None
                self.invader_projectiles.remove(proj)
                return True
        return False

    def _check_invader_projectile_collisions(self) -> bool:
        """Check invader bullets against player, bunkers. Returns True if player hit."""
        hit_player = False

        for proj in self.invader_projectiles[:]:
            # Check player
            if self._rect_collision(
                proj.x,
                proj.y,
                proj.width,
                proj.height,
                self.player.x,
                self.player.y,
                self.player.width,
                self.player.height,
            ):
                self.player.lives -= 1
                self.invader_projectiles.remove(proj)
                hit_player = True
                continue

            # Check bunkers
            for bunker in self.bunkers:
                if self._projectile_hits_bunker(proj, bunker, from_above=True):
                    if proj in self.invader_projectiles:
                        self.invader_projectiles.remove(proj)
                    break

        return hit_player

    def _check_invader_bunker_collision(self) -> None:
        """Check if any invaders overlap bunkers and destroy overlapping cells.

        Classic arcade behavior: aliens act as a deletion mask when they
        march through bunkers. Any bunker pixels overlapping with an alien's
        bounding box are instantly removed (not gradual like bullet damage).
        """
        cell_size = 2  # Classic: 2px cells

        for bunker in self.bunkers:
            bunker_width = len(bunker.cells[0]) * cell_size if bunker.cells else 0
            bunker_height = len(bunker.cells) * cell_size if bunker.cells else 0

            for row in self.invaders:
                for inv in row:
                    if not inv.alive:
                        continue

                    # Check if invader overlaps bunker's bounding box
                    if not self._rect_collision(
                        inv.x, inv.y, inv.width, inv.height,
                        bunker.x + bunker_width / 2,
                        bunker.y + bunker_height / 2,
                        bunker_width, bunker_height
                    ):
                        continue

                    # Calculate invader's bounding box in bunker-local coords
                    inv_left = inv.x - inv.width / 2 - bunker.x
                    inv_right = inv.x + inv.width / 2 - bunker.x
                    inv_top = inv.y - inv.height / 2 - bunker.y
                    inv_bottom = inv.y + inv.height / 2 - bunker.y

                    # Find all cells within the invader's bounding box
                    col_start = max(0, int(inv_left / cell_size))
                    col_end = min(len(bunker.cells[0]), int(inv_right / cell_size) + 1)
                    row_start = max(0, int(inv_top / cell_size))
                    row_end = min(len(bunker.cells), int(inv_bottom / cell_size) + 1)

                    # Destroy all overlapping cells (instant deletion, not gradual)
                    for r in range(row_start, row_end):
                        for c in range(col_start, col_end):
                            cell = bunker.cells[r][c]
                            if cell and cell.health > 0:
                                cell.health = 0  # Instant destruction

    def _projectile_hits_bunker(
        self, proj: Projectile, bunker: Bunker, from_above: bool
    ) -> bool:
        """Check if projectile hits bunker and apply chunk damage.

        Classic arcade behavior:
        - Player bullets (from below) erode the BOTTOM of bunkers
        - Enemy bullets (from above) erode the TOP of bunkers
        - Each hit removes a 2x2 cell chunk (4x4 pixels) centered on impact
        - This allows players to create "sniper holes" through their own cover
        """
        cell_size = 2  # Classic: 2px cells for 22×16 bunker
        damage_radius = 1  # 2x2 chunk (radius of 1 = 3x3 area, but we use 2x2)

        # Check if projectile is in bunker's general area
        bunker_width = len(bunker.cells[0]) * cell_size if bunker.cells else 0
        bunker_height = len(bunker.cells) * cell_size if bunker.cells else 0
        num_rows = len(bunker.cells)
        num_cols = len(bunker.cells[0]) if bunker.cells else 0

        if not self._rect_collision(
            proj.x,
            proj.y,
            proj.width,
            proj.height,
            bunker.x + bunker_width / 2,
            bunker.y + bunker_height / 2,
            bunker_width,
            bunker_height,
        ):
            return False

        # Find column where projectile is
        local_x = proj.x - bunker.x
        col = int(local_x / cell_size)

        if not (0 <= col < num_cols):
            return False

        # Scan for first solid cell in projectile's path (direction-dependent)
        impact_row = -1
        if from_above:
            # Enemy bullet: scan from top down to find surface
            for row in range(num_rows):
                cell = bunker.cells[row][col]
                if cell and cell.health > 0:
                    impact_row = row
                    break
        else:
            # Player bullet: scan from bottom up to find surface
            for row in range(num_rows - 1, -1, -1):
                cell = bunker.cells[row][col]
                if cell and cell.health > 0:
                    impact_row = row
                    break

        if impact_row < 0:
            return False

        # Apply 2x2 damage mask centered on impact point
        # For player bullets, damage extends upward from impact
        # For enemy bullets, damage extends downward from impact
        hit_something = False
        for dr in range(-damage_radius, damage_radius + 1):
            for dc in range(-damage_radius, damage_radius + 1):
                r = impact_row + dr
                c = col + dc

                if 0 <= r < num_rows and 0 <= c < num_cols:
                    cell = bunker.cells[r][c]
                    if cell and cell.health > 0:
                        cell.health = 0  # Instant destruction for chunk
                        hit_something = True

        return hit_something

    def _calculate_kill_reward(self, invader: Invader) -> float:
        """Calculate reward for killing an invader."""
        base_rewards = {
            InvaderType.BOTTOM: self.reward_kill_bottom,
            InvaderType.MIDDLE: self.reward_kill_middle,
            InvaderType.TOP: self.reward_kill_top,
        }
        reward = base_rewards[invader.invader_type]

        # Bonus for edge invaders (they determine when formation drops)
        if self._is_edge_invader(invader):
            reward += self.reward_edge_priority

        return reward

    def _is_edge_invader(self, invader: Invader) -> bool:
        """Check if invader is at the horizontal edge of formation."""
        left_edge, right_edge = self._get_formation_edges()

        # Check if this invader is at the edge
        inv_left = invader.x - invader.width / 2
        inv_right = invader.x + invader.width / 2

        return abs(inv_left - left_edge) < 1 or abs(inv_right - right_edge) < 1

    def _update_mystery_ship(self) -> float:
        """Update mystery ship position and spawning."""
        reward = 0.0

        if self.mystery_ship.active:
            # Move mystery ship
            self.mystery_ship.x += self.mystery_ship.direction * self.mystery_ship.speed

            # Check if off screen
            if self.mystery_ship.direction > 0 and self.mystery_ship.x > self.width + 50:
                self.mystery_ship.active = False
            elif self.mystery_ship.direction < 0 and self.mystery_ship.x < -50:
                self.mystery_ship.active = False
        else:
            # Check if we should spawn
            self.mystery_ship_timer -= 1
            if self.mystery_ship_timer <= 0:
                self.mystery_ship_timer = self.mystery_ship_spawn_interval
                # Random chance to spawn
                if random.random() < 0.3:
                    self.mystery_ship.active = True
                    self.mystery_ship.direction = random.choice([-1, 1])
                    if self.mystery_ship.direction > 0:
                        self.mystery_ship.x = -30
                    else:
                        self.mystery_ship.x = self.width + 30

        return reward

    def _start_next_wave(self) -> None:
        """Start the next wave of invaders."""
        self.wave += 1

        # Classic arcade: Wave 2 at Y=80, Wave 3 at Y=96, max Y=120
        # Offset from base Y=72: Wave 2 = 8, Wave 3 = 24, etc.
        wave_offset = min((self.wave - 1) * 8, 48)  # Max offset = 48 (Y=120)

        # Create new invader formation
        self._create_invaders(wave_offset=wave_offset)

        # Clear projectiles
        self.player_projectile = None
        self.invader_projectiles = []
        self.aimed_shot_fired = False

    def start_recording(self) -> None:
        """Start recording game history for replay."""
        self.recording = True
        self.history = []
        self._record_frame()

    def stop_recording(self) -> List[Dict[str, Any]]:
        """Stop recording and return the history."""
        self.recording = False
        return self.history

    def _record_frame(self) -> None:
        """Record the current frame to history."""
        self.history.append(self.get_state())

    def is_valid_action(self, action: int) -> bool:
        """Check if an action is valid."""
        return 0 <= action < 6

    def get_state(self) -> Dict[str, Any]:
        """Get current game state for rendering or AI."""
        return {
            "player": self.player.to_dict(),
            "invaders": [[inv.to_dict() for inv in row] for row in self.invaders],
            "invader_direction": self.invader_direction,
            "march_step": self.march_step,
            "player_projectile": (
                self.player_projectile.to_dict() if self.player_projectile else None
            ),
            "invader_projectiles": [p.to_dict() for p in self.invader_projectiles],
            "bunkers": [b.to_dict() for b in self.bunkers],
            "mystery_ship": self.mystery_ship.to_dict(),
            "score": self.score,
            "lives": self.player.lives,
            "wave": self.wave,
            "game_over": self.game_over,
            "frame": self.frame_count,
            "width": self.width,
            "height": self.height,
            "invaders_alive": self._get_alive_count(),
            "total_invaders": self.invader_rows * self.invader_cols,
        }

    def get_score(self) -> int:
        """Get current game score."""
        return self.score
