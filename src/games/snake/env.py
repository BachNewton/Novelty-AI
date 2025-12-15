"""
Snake Environment - Gym-like wrapper implementing EnvInterface.
Provides 20-dimensional state encoding suitable for neural network input.
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, List, Optional

from ...core.env_interface import EnvInterface
from .game import SnakeGame, Direction, Point


class SnakeEnv(EnvInterface):
    """
    Gym-like environment wrapper for the Snake game implementing EnvInterface.

    Provides a clean interface for reinforcement learning:
    - reset() -> initial state
    - step(action) -> (next_state, reward, done, info)

    State is a 20-dimensional feature vector encoding danger, direction, food,
    multi-step danger look-ahead, and reachable cells (flood fill).
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        reward_config: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the environment.

        Args:
            width: Grid width in cells
            height: Grid height in cells
            reward_config: Optional reward configuration dictionary
        """
        self.game = SnakeGame(width, height, reward_config=reward_config)
        self.width = width
        self.height = height
        self._reward_config = reward_config

    @property
    def state_size(self) -> int:
        """Get the state size (20 features)."""
        return 20

    @property
    def action_size(self) -> int:
        """Get the action size (3 actions: straight, right, left)."""
        return 3

    def reset(self, record: bool = False) -> np.ndarray:
        """
        Reset environment and return initial state.

        Args:
            record: If True, start recording for replay

        Returns:
            Initial state as numpy array
        """
        self.game.reset()
        if record:
            self.game.start_recording()
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return results.

        Args:
            action: 0 = straight, 1 = right turn, 2 = left turn

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        _, reward, done, info = self.game.step(action)
        state = self._get_state()
        return state, reward, done, info

    def _turn_right(self, direction: Direction) -> Direction:
        """Get direction after turning right (clockwise)."""
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(direction)
        return clock_wise[(idx + 1) % 4]

    def _turn_left(self, direction: Direction) -> Direction:
        """Get direction after turning left (counter-clockwise)."""
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(direction)
        return clock_wise[(idx - 1) % 4]

    def _get_next_point(self, direction: Direction, start: Optional[Point] = None) -> Point:
        """Get the next point in the given direction from start (or head)."""
        if start is None:
            start = self.game.head
        x, y = start.x, start.y
        if direction == Direction.RIGHT:
            x += 1
        elif direction == Direction.LEFT:
            x -= 1
        elif direction == Direction.DOWN:
            y += 1
        elif direction == Direction.UP:
            y -= 1
        return Point(x, y)

    def _get_point_n_steps(self, direction: Direction, n: int) -> Point:
        """Get the point N steps in the given direction from head."""
        x, y = self.game.head.x, self.game.head.y
        if direction == Direction.RIGHT:
            x += n
        elif direction == Direction.LEFT:
            x -= n
        elif direction == Direction.DOWN:
            y += n
        elif direction == Direction.UP:
            y -= n
        return Point(x, y)

    def _count_reachable_cells(self, start_point: Point, max_depth: int = 20) -> int:
        """
        Count reachable cells from a starting point using BFS.
        This detects traps - if few cells are reachable, the snake would be trapped.

        Args:
            start_point: Starting position to check from
            max_depth: Maximum search depth (limits computation)

        Returns:
            Number of reachable cells
        """
        if self.game.is_danger(start_point):
            return 0

        # Use set of tuples for O(1) lookup
        snake_set = {(p.x, p.y) for p in self.game.snake}
        visited = {(start_point.x, start_point.y)}
        queue = deque([(start_point.x, start_point.y, 0)])
        count = 1

        while queue:
            x, y, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Check all 4 neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if (nx, ny) in visited:
                    continue

                # Check bounds and snake body
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in snake_set:
                    visited.add((nx, ny))
                    queue.append((nx, ny, depth + 1))
                    count += 1

        return count

    def _get_state(self) -> np.ndarray:
        """
        Get 20-dimensional state vector.

        Returns:
            State as numpy array of shape (20,)
        """
        return self._get_state_vector()

    def _get_state_vector(self) -> np.ndarray:
        """
        Convert game state to 20-dimensional feature vector.

        Features:
        [0-2]: Danger straight, right, left (1 step)
        [3-6]: Direction (one-hot: left, right, up, down)
        [7-10]: Food location relative to head (left, right, up, down)
        [11-13]: Danger 2 steps ahead (straight, right, left)
        [14-16]: Danger 3 steps ahead (straight, right, left)
        [17-19]: Reachable cells (flood fill) for each move direction

        Returns:
            State as numpy array of shape (20,)
        """
        head = self.game.head
        direction = self.game.direction

        dir_r = direction == Direction.RIGHT
        dir_l = direction == Direction.LEFT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        # Points for danger detection (1 step)
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        # Get directions for relative movement
        dir_straight = direction
        dir_right = self._turn_right(direction)
        dir_left = self._turn_left(direction)

        # Points for multi-step danger detection
        point_straight_2 = self._get_point_n_steps(dir_straight, 2)
        point_straight_3 = self._get_point_n_steps(dir_straight, 3)
        point_right_2 = self._get_point_n_steps(dir_right, 2)
        point_right_3 = self._get_point_n_steps(dir_right, 3)
        point_left_2 = self._get_point_n_steps(dir_left, 2)
        point_left_3 = self._get_point_n_steps(dir_left, 3)

        # Points for flood fill (1 step ahead in each direction)
        point_straight_1 = self._get_next_point(dir_straight)
        point_right_1 = self._get_next_point(dir_right)
        point_left_1 = self._get_next_point(dir_left)

        # Flood fill normalization factor
        max_cells = (self.width * self.height) / 2.0

        state = [
            # Danger straight (1 step)
            (dir_r and self.game.is_danger(point_r)) or
            (dir_l and self.game.is_danger(point_l)) or
            (dir_u and self.game.is_danger(point_u)) or
            (dir_d and self.game.is_danger(point_d)),

            # Danger right (clockwise from current direction, 1 step)
            (dir_r and self.game.is_danger(point_d)) or
            (dir_l and self.game.is_danger(point_u)) or
            (dir_u and self.game.is_danger(point_r)) or
            (dir_d and self.game.is_danger(point_l)),

            # Danger left (counter-clockwise from current direction, 1 step)
            (dir_r and self.game.is_danger(point_u)) or
            (dir_l and self.game.is_danger(point_d)) or
            (dir_u and self.game.is_danger(point_l)) or
            (dir_d and self.game.is_danger(point_r)),

            # Direction (one-hot encoded)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location relative to head
            self.game.food.x < head.x,  # Food is to the left
            self.game.food.x > head.x,  # Food is to the right
            self.game.food.y < head.y,  # Food is above (y=0 is top)
            self.game.food.y > head.y,  # Food is below

            # Multi-step danger (2 steps ahead)
            self.game.is_danger(point_straight_2),
            self.game.is_danger(point_right_2),
            self.game.is_danger(point_left_2),

            # Multi-step danger (3 steps ahead)
            self.game.is_danger(point_straight_3),
            self.game.is_danger(point_right_3),
            self.game.is_danger(point_left_3),

            # Reachable cells (flood fill) - normalized
            self._count_reachable_cells(point_straight_1) / max_cells,
            self._count_reachable_cells(point_right_1) / max_cells,
            self._count_reachable_cells(point_left_1) / max_cells,
        ]

        return np.array(state, dtype=np.float32)

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
