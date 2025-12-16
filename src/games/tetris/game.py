"""
Tetris Game Core - Modern Tetris with SRS rotation, hold, and preview.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import IntEnum
import random
import copy

from ...core.game_interface import GameInterface, GameMetadata


class PieceType(IntEnum):
    """The 7 standard Tetris pieces."""
    I = 0
    O = 1
    T = 2
    S = 3
    Z = 4
    J = 5
    L = 6


class Action(IntEnum):
    """Tetris actions."""
    NOOP = 0       # No operation - just let gravity tick
    LEFT = 1
    RIGHT = 2
    ROTATE_CW = 3
    ROTATE_CCW = 4
    SOFT_DROP = 5
    HARD_DROP = 6
    HOLD = 7


# Tetromino shapes: list of (x, y) offsets for each rotation state
# Origin is the rotation center
TETROMINOES: Dict[PieceType, List[List[Tuple[int, int]]]] = {
    PieceType.I: [
        [(0, 1), (1, 1), (2, 1), (3, 1)],    # 0
        [(2, 0), (2, 1), (2, 2), (2, 3)],    # R
        [(0, 2), (1, 2), (2, 2), (3, 2)],    # 2
        [(1, 0), (1, 1), (1, 2), (1, 3)],    # L
    ],
    PieceType.O: [
        [(1, 0), (2, 0), (1, 1), (2, 1)],    # All rotations same
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
    ],
    PieceType.T: [
        [(1, 0), (0, 1), (1, 1), (2, 1)],    # 0
        [(1, 0), (1, 1), (2, 1), (1, 2)],    # R
        [(0, 1), (1, 1), (2, 1), (1, 2)],    # 2
        [(1, 0), (0, 1), (1, 1), (1, 2)],    # L
    ],
    PieceType.S: [
        [(1, 0), (2, 0), (0, 1), (1, 1)],    # 0
        [(1, 0), (1, 1), (2, 1), (2, 2)],    # R
        [(1, 1), (2, 1), (0, 2), (1, 2)],    # 2
        [(0, 0), (0, 1), (1, 1), (1, 2)],    # L
    ],
    PieceType.Z: [
        [(0, 0), (1, 0), (1, 1), (2, 1)],    # 0
        [(2, 0), (1, 1), (2, 1), (1, 2)],    # R
        [(0, 1), (1, 1), (1, 2), (2, 2)],    # 2
        [(1, 0), (0, 1), (1, 1), (0, 2)],    # L
    ],
    PieceType.J: [
        [(0, 0), (0, 1), (1, 1), (2, 1)],    # 0
        [(1, 0), (2, 0), (1, 1), (1, 2)],    # R
        [(0, 1), (1, 1), (2, 1), (2, 2)],    # 2
        [(1, 0), (1, 1), (0, 2), (1, 2)],    # L
    ],
    PieceType.L: [
        [(2, 0), (0, 1), (1, 1), (2, 1)],    # 0
        [(1, 0), (1, 1), (1, 2), (2, 2)],    # R
        [(0, 1), (1, 1), (2, 1), (0, 2)],    # 2
        [(0, 0), (1, 0), (1, 1), (1, 2)],    # L
    ],
}

# SRS wall kick data: (from_rotation, to_rotation) -> list of (dx, dy) offsets to try
# For J, L, S, T, Z pieces
WALL_KICKS_JLSTZ: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
    (0, 1): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (1, 0): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (1, 2): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (2, 1): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (2, 3): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (3, 2): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (3, 0): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (0, 3): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
}

# For I piece (different wall kicks)
WALL_KICKS_I: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
}

# Standard Tetris colors for each piece type
PIECE_COLORS: Dict[PieceType, str] = {
    PieceType.I: "cyan",
    PieceType.O: "yellow",
    PieceType.T: "purple",
    PieceType.S: "green",
    PieceType.Z: "red",
    PieceType.J: "blue",
    PieceType.L: "orange",
}


@dataclass
class Piece:
    """A Tetris piece with position and rotation."""
    piece_type: PieceType
    x: int  # Left edge position on board
    y: int  # Top edge position on board
    rotation: int  # 0, 1, 2, 3 (0=spawn, 1=CW, 2=180, 3=CCW)

    def get_cells(self) -> List[Tuple[int, int]]:
        """Get absolute board positions of all cells."""
        offsets = TETROMINOES[self.piece_type][self.rotation]
        return [(self.x + dx, self.y + dy) for dx, dy in offsets]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": int(self.piece_type),
            "x": self.x,
            "y": self.y,
            "rotation": self.rotation,
        }


class TetrisGame(GameInterface):
    """
    Modern Tetris with SRS rotation, hold piece, and preview queue.

    Features:
    - Standard 10x20 visible board (+ 2 hidden rows for spawning)
    - 7 tetromino shapes with SRS (Super Rotation System)
    - Wall kicks for rotation
    - Hold piece (swap once per drop)
    - Preview queue showing next pieces
    - 7-bag randomizer for fair piece distribution
    - Line clearing with scoring
    """

    @classmethod
    def get_metadata(cls) -> GameMetadata:
        """Return metadata about Tetris game."""
        return GameMetadata(
            name="Tetris",
            id="tetris",
            description="Classic block-stacking puzzle - clear lines, score points, survive!",
            version="1.0.0",
            min_players=1,
            max_players=1,
            supports_human=True,
            recommended_algorithms=["ppo", "dqn"]
        )

    def __init__(
        self,
        width: int = 10,
        height: int = 20,
        preview_count: int = 5,
        reward_config: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Tetris game.

        Args:
            width: Board width in cells (standard: 10)
            height: Visible board height (standard: 20)
            preview_count: Number of next pieces to show
            reward_config: Optional reward configuration
        """
        self.width = width
        self.height = height
        self.visible_height = height
        self.total_height = height + 4  # Extra rows for spawning
        self.preview_count = preview_count

        # Reward configuration
        reward_config = reward_config or {}
        self.reward_single = reward_config.get("single", 100.0)
        self.reward_double = reward_config.get("double", 300.0)
        self.reward_triple = reward_config.get("triple", 500.0)
        self.reward_tetris = reward_config.get("tetris", 800.0)
        self.reward_game_over = reward_config.get("game_over", -100.0)
        self.reward_step = reward_config.get("step", 0.01)
        self.reward_height_penalty = reward_config.get("height_penalty", -0.1)

        # Game state (initialized in reset)
        self.board: List[List[Optional[PieceType]]] = []
        self.current_piece: Optional[Piece] = None
        self.held_piece: Optional[PieceType] = None
        self.can_hold: bool = True
        self.bag: List[PieceType] = []
        self.next_pieces: List[PieceType] = []

        self.score: int = 0
        self.lines_cleared: int = 0
        self.level: int = 1
        self.frame_count: int = 0
        self.game_over: bool = False
        self.pieces_placed: int = 0

        # Gravity timing
        self.gravity_counter: int = 0

        # Recording
        self.history: List[Dict[str, Any]] = []
        self.recording: bool = False

        self.reset()

    @property
    def action_space_size(self) -> int:
        """Number of possible actions."""
        return 8

    @property
    def action_names(self) -> List[str]:
        """Human-readable action names."""
        return ["Noop", "Left", "Right", "Rotate CW", "Rotate CCW", "Soft Drop", "Hard Drop", "Hold"]

    def reset(self) -> Dict[str, Any]:
        """Reset game to initial state."""
        # Clear board (row 0 is top)
        self.board = [[None for _ in range(self.width)] for _ in range(self.total_height)]

        # Reset state
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.frame_count = 0
        self.game_over = False
        self.pieces_placed = 0
        self.gravity_counter = 0

        # Reset hold
        self.held_piece = None
        self.can_hold = True

        # Fill bag and next pieces
        self.bag = []
        self.next_pieces = []
        self._fill_bag()
        for _ in range(self.preview_count):
            self._add_next_piece()

        # Spawn first piece
        self._spawn_piece()

        # Reset recording
        self.history = []
        if self.recording:
            self._record_frame()

        return self.get_state()

    def _fill_bag(self) -> None:
        """Fill the bag with a shuffled set of all 7 pieces."""
        self.bag = list(PieceType)
        random.shuffle(self.bag)

    def _get_next_from_bag(self) -> PieceType:
        """Get next piece from bag, refilling if empty."""
        if not self.bag:
            self._fill_bag()
        return self.bag.pop()

    def _add_next_piece(self) -> None:
        """Add a piece to the next queue."""
        self.next_pieces.append(self._get_next_from_bag())

    def _spawn_piece(self, piece_type: Optional[PieceType] = None) -> bool:
        """
        Spawn a new piece at top of board.

        Args:
            piece_type: Optional specific piece to spawn

        Returns:
            True if spawn successful, False if blocked (game over)
        """
        if piece_type is None:
            if self.next_pieces:
                piece_type = self.next_pieces.pop(0)
                self._add_next_piece()
            else:
                piece_type = self._get_next_from_bag()

        # Spawn position (centered, at top)
        spawn_x = (self.width - 4) // 2
        spawn_y = 0  # Top of total board (above visible area)

        self.current_piece = Piece(
            piece_type=piece_type,
            x=spawn_x,
            y=spawn_y,
            rotation=0
        )

        # Check if spawn position is blocked
        if self._check_collision(self.current_piece):
            self.game_over = True
            return False

        self.can_hold = True
        self.gravity_counter = 0  # Reset gravity for new piece
        return True

    def _check_collision(self, piece: Piece) -> bool:
        """Check if piece collides with walls or placed blocks."""
        for x, y in piece.get_cells():
            # Wall collision
            if x < 0 or x >= self.width:
                return True
            if y < 0 or y >= self.total_height:
                return True
            # Block collision
            if self.board[y][x] is not None:
                return True
        return False

    def _lock_piece(self) -> int:
        """
        Lock current piece to board and check for line clears.

        Returns:
            Number of lines cleared
        """
        if self.current_piece is None:
            return 0

        # Place piece on board
        for x, y in self.current_piece.get_cells():
            if 0 <= y < self.total_height and 0 <= x < self.width:
                self.board[y][x] = self.current_piece.piece_type

        self.pieces_placed += 1

        # Check for line clears
        lines_to_clear = []
        for y in range(self.total_height):
            if all(self.board[y][x] is not None for x in range(self.width)):
                lines_to_clear.append(y)

        # Clear lines (from bottom up to avoid index issues)
        for y in sorted(lines_to_clear, reverse=True):
            del self.board[y]
            self.board.insert(0, [None for _ in range(self.width)])

        cleared = len(lines_to_clear)
        self.lines_cleared += cleared

        # Update level (every 10 lines)
        self.level = (self.lines_cleared // 10) + 1

        return cleared

    def _move(self, dx: int, dy: int) -> bool:
        """Try to move current piece. Returns True if successful."""
        if self.current_piece is None:
            return False

        test_piece = Piece(
            piece_type=self.current_piece.piece_type,
            x=self.current_piece.x + dx,
            y=self.current_piece.y + dy,
            rotation=self.current_piece.rotation
        )

        if not self._check_collision(test_piece):
            self.current_piece = test_piece
            return True
        return False

    def _rotate(self, direction: int) -> bool:
        """
        Try to rotate current piece with wall kicks.

        Args:
            direction: 1 for CW, -1 for CCW

        Returns:
            True if rotation successful
        """
        if self.current_piece is None:
            return False

        piece_type = self.current_piece.piece_type
        old_rotation = self.current_piece.rotation
        new_rotation = (old_rotation + direction) % 4

        # Get wall kick data
        if piece_type == PieceType.I:
            kicks = WALL_KICKS_I.get((old_rotation, new_rotation), [(0, 0)])
        elif piece_type == PieceType.O:
            kicks = [(0, 0)]  # O doesn't kick
        else:
            kicks = WALL_KICKS_JLSTZ.get((old_rotation, new_rotation), [(0, 0)])

        # Try each kick offset
        for dx, dy in kicks:
            test_piece = Piece(
                piece_type=piece_type,
                x=self.current_piece.x + dx,
                y=self.current_piece.y - dy,  # SRS uses inverted Y for kicks
                rotation=new_rotation
            )
            if not self._check_collision(test_piece):
                self.current_piece = test_piece
                return True

        return False

    def _hard_drop(self) -> int:
        """Drop piece to bottom instantly. Returns cells dropped."""
        if self.current_piece is None:
            return 0

        cells_dropped = 0
        while self._move(0, 1):
            cells_dropped += 1
        return cells_dropped

    def _hold(self) -> bool:
        """Swap current piece with held piece. Returns True if successful."""
        if not self.can_hold or self.current_piece is None:
            return False

        current_type = self.current_piece.piece_type

        if self.held_piece is None:
            # First hold - take from next queue
            self.held_piece = current_type
            self._spawn_piece()
        else:
            # Swap with held
            self.held_piece, swap_type = current_type, self.held_piece
            self._spawn_piece(swap_type)

        self.can_hold = False
        return True

    def _get_ghost_piece(self) -> Optional[Piece]:
        """Get position where piece would land (for ghost preview)."""
        if self.current_piece is None:
            return None

        ghost = Piece(
            piece_type=self.current_piece.piece_type,
            x=self.current_piece.x,
            y=self.current_piece.y,
            rotation=self.current_piece.rotation
        )

        # Move down until collision
        while not self._check_collision(Piece(
            piece_type=ghost.piece_type,
            x=ghost.x,
            y=ghost.y + 1,
            rotation=ghost.rotation
        )):
            ghost.y += 1

        return ghost

    def _get_max_height(self) -> int:
        """Get height of tallest column."""
        for y in range(self.total_height):
            if any(self.board[y][x] is not None for x in range(self.width)):
                return self.total_height - y
        return 0

    def _get_gravity_speed(self) -> int:
        """
        Get frames per automatic drop based on level.

        Lower values = faster falling.
        Level 1: 30 frames per drop (~1 second at 30fps)
        Level 10+: 3 frames per drop (very fast)
        """
        # Each level reduces frames by 3, minimum of 3 frames
        return max(3, 30 - (self.level - 1) * 3)

    def _calculate_reward(self, lines_cleared: int, game_over: bool) -> float:
        """Calculate reward for current step."""
        if game_over:
            return self.reward_game_over

        reward = self.reward_step

        # Line clear rewards
        if lines_cleared == 1:
            reward += self.reward_single
        elif lines_cleared == 2:
            reward += self.reward_double
        elif lines_cleared == 3:
            reward += self.reward_triple
        elif lines_cleared >= 4:
            reward += self.reward_tetris

        # Height penalty
        max_height = self._get_max_height()
        reward += self.reward_height_penalty * max_height

        return reward

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one game step.

        Args:
            action: Action to take (see Action enum)

        Returns:
            Tuple of (state, reward, done, info)
        """
        self.frame_count += 1
        lines_cleared = 0
        piece_locked = False

        if self.game_over:
            return self.get_state(), 0.0, True, {"score": self.score}

        # Apply automatic gravity
        self.gravity_counter += 1
        if self.gravity_counter >= self._get_gravity_speed():
            self.gravity_counter = 0
            if not self._move(0, 1):  # Can't move down
                # Piece hit bottom, lock it
                lines_cleared = self._lock_piece()
                piece_locked = True
                if not self.game_over:
                    self._spawn_piece()

        # Execute action (if piece wasn't just locked by gravity)
        if not piece_locked and self.current_piece is not None:
            if action == Action.LEFT:
                self._move(-1, 0)
            elif action == Action.RIGHT:
                self._move(1, 0)
            elif action == Action.ROTATE_CW:
                self._rotate(1)
            elif action == Action.ROTATE_CCW:
                self._rotate(-1)
            elif action == Action.SOFT_DROP:
                if not self._move(0, 1):
                    # Hit bottom, lock piece
                    lines_cleared = self._lock_piece()
                    piece_locked = True
                    if not self.game_over:
                        self._spawn_piece()
            elif action == Action.HARD_DROP:
                self._hard_drop()
                lines_cleared = self._lock_piece()
                piece_locked = True
                if not self.game_over:
                    self._spawn_piece()
            elif action == Action.HOLD:
                self._hold()

        # Update score based on lines cleared
        if lines_cleared > 0:
            base_score = [0, 100, 300, 500, 800][min(lines_cleared, 4)]
            self.score += base_score * self.level

        # Calculate reward
        reward = self._calculate_reward(lines_cleared, self.game_over)

        if self.recording:
            self._record_frame()

        info = {
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "level": self.level,
            "pieces_placed": self.pieces_placed,
            "lines_this_step": lines_cleared,
            "piece_locked": piece_locked,
        }

        return self.get_state(), reward, self.game_over, info

    def is_valid_action(self, action: int) -> bool:
        """Check if action is valid."""
        return 0 <= action < self.action_space_size

    def get_state(self) -> Dict[str, Any]:
        """Get current game state for rendering."""
        # Convert board to serializable format
        board_data = [
            [int(cell) if cell is not None else -1 for cell in row]
            for row in self.board[self.total_height - self.visible_height:]  # Only visible rows
        ]

        ghost = self._get_ghost_piece()

        state = {
            "board": board_data,
            "current_piece": self.current_piece.to_dict() if self.current_piece else None,
            "ghost_piece": ghost.to_dict() if ghost else None,
            "held_piece": int(self.held_piece) if self.held_piece is not None else -1,
            "can_hold": self.can_hold,
            "next_pieces": [int(p) for p in self.next_pieces],
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "level": self.level,
            "game_over": self.game_over,
            "frame": self.frame_count,
            "width": self.width,
            "height": self.visible_height,
            "pieces_placed": self.pieces_placed,
        }

        return state

    def get_score(self) -> int:
        """Get current score."""
        return self.score

    def start_recording(self) -> None:
        """Start recording game history."""
        self.recording = True
        self.history = []
        self._record_frame()

    def stop_recording(self) -> List[Dict[str, Any]]:
        """Stop recording and return history."""
        self.recording = False
        return self.history

    def _record_frame(self) -> None:
        """Record current frame to history."""
        self.history.append(self.get_state())

    # Helper methods for AI state encoding
    def get_column_heights(self) -> List[int]:
        """Get height of each column."""
        heights = []
        for x in range(self.width):
            height = 0
            for y in range(self.total_height):
                if self.board[y][x] is not None:
                    height = self.total_height - y
                    break
            heights.append(height)
        return heights

    def get_holes(self) -> int:
        """Count number of holes (empty cells below filled cells)."""
        holes = 0
        for x in range(self.width):
            found_block = False
            for y in range(self.total_height):
                if self.board[y][x] is not None:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def get_bumpiness(self) -> int:
        """Sum of absolute height differences between adjacent columns."""
        heights = self.get_column_heights()
        return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
