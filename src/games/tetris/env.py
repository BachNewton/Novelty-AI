"""
Tetris Environment - Gym-like wrapper implementing EnvInterface.
Provides engineered state features suitable for neural network input.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional

from ...core.env_interface import EnvInterface
from .game import TetrisGame, PieceType, Action


class TetrisEnv(EnvInterface):
    """
    Gym-like environment wrapper for Tetris implementing EnvInterface.

    State is an 86-dimensional feature vector encoding:
    - Board structure (heights, holes, bumpiness)
    - Current piece information
    - Preview queue
    - Held piece
    - Game progress

    This engineered representation is much more efficient than raw pixels.
    """

    # State size breakdown:
    # - Column heights: 10
    # - Height differences: 9
    # - Holes per column: 10
    # - Current piece one-hot: 7
    # - Next pieces one-hot: 5 * 7 = 35
    # - Held piece one-hot (+ empty): 8
    # - Can hold flag: 1
    # - Piece x position (normalized): 1
    # - Piece y position (normalized): 1
    # - Piece rotation (normalized): 1
    # - Lines cleared (normalized): 1
    # - Level (normalized): 1
    # - Max height (normalized): 1
    # Total: 86

    STATE_SIZE = 86

    def __init__(
        self,
        width: int = 10,
        height: int = 20,
        preview_count: int = 5,
        reward_config: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the environment.

        Args:
            width: Board width (standard: 10)
            height: Board height (standard: 20)
            preview_count: Number of next pieces to show
            reward_config: Optional reward configuration
        """
        self.game = TetrisGame(
            width=width,
            height=height,
            preview_count=preview_count,
            reward_config=reward_config
        )
        self.width = width
        self.height = height
        self.preview_count = preview_count
        self._reward_config = reward_config

    @property
    def state_size(self) -> int:
        """Get state size (86 features)."""
        return self.STATE_SIZE

    @property
    def action_size(self) -> int:
        """Get action size (number of possible actions)."""
        return self.game.action_space_size

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
            action: Action index (0-6, see Action enum)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        _, reward, done, info = self.game.step(action)
        state = self._get_state()
        return state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """
        Get 86-dimensional state vector.

        Returns:
            State as numpy array
        """
        features = []

        # --- Board features ---

        # Column heights (10 values, normalized 0-1)
        heights = self.game.get_column_heights()
        max_height = self.game.total_height
        for h in heights:
            features.append(h / max_height)

        # Height differences between adjacent columns (9 values, normalized)
        for i in range(len(heights) - 1):
            diff = (heights[i + 1] - heights[i]) / max_height
            features.append((diff + 1) / 2)  # Normalize to 0-1

        # Holes per column (10 values, normalized)
        for x in range(self.width):
            holes = self._count_holes_in_column(x)
            features.append(min(holes / 10, 1.0))  # Cap at 10 holes

        # --- Current piece features ---

        # Current piece type (7 one-hot)
        piece_one_hot = [0.0] * 7
        if self.game.current_piece is not None:
            piece_one_hot[self.game.current_piece.piece_type] = 1.0
        features.extend(piece_one_hot)

        # --- Preview queue features ---

        # Next pieces (preview_count * 7 one-hot)
        for i in range(self.preview_count):
            next_one_hot = [0.0] * 7
            if i < len(self.game.next_pieces):
                next_one_hot[self.game.next_pieces[i]] = 1.0
            features.extend(next_one_hot)

        # --- Hold piece features ---

        # Held piece (7 one-hot + 1 for empty = 8)
        held_one_hot = [0.0] * 8
        if self.game.held_piece is None:
            held_one_hot[7] = 1.0  # Empty slot
        else:
            held_one_hot[self.game.held_piece] = 1.0
        features.extend(held_one_hot)

        # Can hold flag (1 value)
        features.append(1.0 if self.game.can_hold else 0.0)

        # --- Piece position features ---

        # Current piece position (normalized)
        if self.game.current_piece is not None:
            features.append(self.game.current_piece.x / self.width)
            features.append(self.game.current_piece.y / self.game.total_height)
            features.append(self.game.current_piece.rotation / 3.0)
        else:
            features.extend([0.5, 0.0, 0.0])

        # --- Game progress features ---

        # Lines cleared (normalized, cap at 200)
        features.append(min(self.game.lines_cleared / 200, 1.0))

        # Level (normalized, cap at 20)
        features.append(min(self.game.level / 20, 1.0))

        # Max height (normalized)
        current_max_height = max(heights) if heights else 0
        features.append(current_max_height / max_height)

        return np.array(features, dtype=np.float32)

    def _count_holes_in_column(self, x: int) -> int:
        """Count holes in a specific column."""
        holes = 0
        found_block = False
        for y in range(self.game.total_height):
            if self.game.board[y][x] is not None:
                found_block = True
            elif found_block:
                holes += 1
        return holes

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
