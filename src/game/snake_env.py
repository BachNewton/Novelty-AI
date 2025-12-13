"""
Snake Environment - Gym-like wrapper for RL training.
Provides state encoding suitable for neural network input.
"""
import numpy as np
from typing import Tuple, Dict, Any, List
from .snake_game import SnakeGame, Direction, Point


class SnakeEnv:
    """
    Gym-like environment wrapper for the Snake game.

    Provides a clean interface for reinforcement learning:
    - reset() -> initial state
    - step(action) -> (next_state, reward, done, info)

    State is encoded as an 11-dimensional feature vector suitable
    for neural network input.
    """

    def __init__(self, width: int = 20, height: int = 20):
        """
        Initialize the environment.

        Args:
            width: Grid width in cells
            height: Grid height in cells
        """
        self.game = SnakeGame(width, height)
        self.width = width
        self.height = height

        # State space: 11 features (danger + direction + food location)
        self.state_size = 11
        # Action space: 3 actions (straight, right, left)
        self.action_size = 3

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
        return self._get_state_vector()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return results.

        Args:
            action: 0 = straight, 1 = right turn, 2 = left turn

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        _, reward, done, info = self.game.step(action)
        state = self._get_state_vector()
        return state, reward, done, info

    def _get_state_vector(self) -> np.ndarray:
        """
        Convert game state to 11-dimensional feature vector.

        Features:
        [0-2]: Danger straight, right, left
        [3-6]: Direction (one-hot: left, right, up, down)
        [7-10]: Food location relative to head (left, right, up, down)

        Returns:
            State as numpy array of shape (11,)
        """
        head = self.game.head
        direction = self.game.direction

        dir_r = direction == Direction.RIGHT
        dir_l = direction == Direction.LEFT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        # Points for danger detection
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        state = [
            # Danger straight
            (dir_r and self.game.is_danger(point_r)) or
            (dir_l and self.game.is_danger(point_l)) or
            (dir_u and self.game.is_danger(point_u)) or
            (dir_d and self.game.is_danger(point_d)),

            # Danger right (clockwise from current direction)
            (dir_r and self.game.is_danger(point_d)) or
            (dir_l and self.game.is_danger(point_u)) or
            (dir_u and self.game.is_danger(point_r)) or
            (dir_d and self.game.is_danger(point_l)),

            # Danger left (counter-clockwise from current direction)
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
