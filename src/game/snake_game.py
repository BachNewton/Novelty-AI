"""
Snake Game Core - Pure game logic without rendering.
Designed for high-speed training with optional visualization.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import IntEnum
import random


class Direction(IntEnum):
    """Snake movement directions."""
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


@dataclass
class Point:
    """A point on the game grid."""
    x: int
    y: int

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for serialization."""
        return {"x": self.x, "y": self.y}


class SnakeGame:
    """
    Core Snake game logic.

    The game is played on a grid where the snake moves and tries to eat food.
    When food is eaten, the snake grows longer. The game ends when the snake
    hits a wall or itself.
    """

    def __init__(self, width: int = 20, height: int = 20):
        """
        Initialize the game.

        Args:
            width: Grid width in cells
            height: Grid height in cells
        """
        self.width = width
        self.height = height

        # Game state (initialized in reset)
        self.direction: Direction = Direction.RIGHT
        self.snake: List[Point] = []
        self.head: Point = Point(0, 0)
        self.food: Point = Point(0, 0)
        self.score: int = 0
        self.frame_count: int = 0
        self.game_over: bool = False

        # For replay recording
        self.history: List[Dict[str, Any]] = []
        self.recording: bool = False

        self.reset()

    def reset(self) -> Dict[str, Any]:
        """
        Reset game state and return initial state.

        Returns:
            Dictionary containing the initial game state
        """
        # Start in center
        center_x = self.width // 2
        center_y = self.height // 2

        self.direction = Direction.RIGHT
        self.snake = [
            Point(center_x, center_y),
            Point(center_x - 1, center_y),
            Point(center_x - 2, center_y),
        ]
        self.head = self.snake[0]
        self.score = 0
        self.frame_count = 0
        self.game_over = False

        self._place_food()

        # Reset history if recording
        self.history = []
        if self.recording:
            self._record_frame()

        return self.get_state()

    def start_recording(self):
        """Start recording game history for replay."""
        self.recording = True
        self.history = []
        self._record_frame()

    def stop_recording(self) -> List[Dict[str, Any]]:
        """Stop recording and return the history."""
        self.recording = False
        return self.history

    def _record_frame(self):
        """Record the current frame to history."""
        self.history.append({
            "snake": [p.to_dict() for p in self.snake],
            "food": self.food.to_dict(),
            "direction": int(self.direction),
            "score": self.score,
            "frame": self.frame_count,
        })

    def _place_food(self):
        """Place food at random location not on snake."""
        attempts = 0
        max_attempts = self.width * self.height

        while attempts < max_attempts:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            food = Point(x, y)

            if food not in self.snake:
                self.food = food
                return

            attempts += 1

        # Fallback: find any empty cell (game is almost won)
        for x in range(self.width):
            for y in range(self.height):
                point = Point(x, y)
                if point not in self.snake:
                    self.food = point
                    return

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one game step.

        Args:
            action: 0 = straight, 1 = right turn, 2 = left turn

        Returns:
            Tuple of (state, reward, done, info)
        """
        self.frame_count += 1

        # Convert action to new direction
        # Action 0: continue straight
        # Action 1: turn right (clockwise)
        # Action 2: turn left (counter-clockwise)
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 1:  # Right turn
            new_idx = (idx + 1) % 4
        elif action == 2:  # Left turn
            new_idx = (idx - 1) % 4
        else:  # Straight
            new_idx = idx

        self.direction = clock_wise[new_idx]

        # Move head
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1

        self.head = Point(x, y)
        self.snake.insert(0, self.head)

        # Check collision
        reward = 0.0
        game_over = False

        if self._is_collision():
            game_over = True
            reward = -10.0
            self.game_over = True

            if self.recording:
                self._record_frame()

            return self.get_state(), reward, game_over, {"score": self.score}

        # Check food
        if self.head == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            self.snake.pop()  # Remove tail

        # Small negative reward for each step to encourage efficiency
        if reward == 0:
            reward = -0.01

        # Timeout penalty (prevents infinite loops)
        if self.frame_count > 100 * len(self.snake):
            game_over = True
            reward = -10.0
            self.game_over = True

        if self.recording:
            self._record_frame()

        return self.get_state(), reward, game_over, {"score": self.score}

    def step_direction(self, direction: Direction) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one game step with a direct direction (for human play).

        Args:
            direction: The direction to move

        Returns:
            Tuple of (state, reward, done, info)
        """
        # Prevent 180-degree turns
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }

        if direction == opposites.get(self.direction):
            direction = self.direction  # Ignore invalid turn

        # Convert direction to action
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_idx = clock_wise.index(self.direction)
        new_idx = clock_wise.index(direction)

        diff = (new_idx - current_idx) % 4
        if diff == 0:
            action = 0  # Straight
        elif diff == 1:
            action = 1  # Right
        elif diff == 3:
            action = 2  # Left
        else:
            action = 0  # Straight (shouldn't happen)

        return self.step(action)

    def _is_collision(self, point: Optional[Point] = None) -> bool:
        """
        Check if point collides with wall or snake body.

        Args:
            point: Point to check (defaults to head)

        Returns:
            True if collision detected
        """
        if point is None:
            point = self.head

        # Wall collision
        if point.x < 0 or point.x >= self.width:
            return True
        if point.y < 0 or point.y >= self.height:
            return True

        # Self collision (skip head)
        if point in self.snake[1:]:
            return True

        return False

    def is_danger(self, point: Point) -> bool:
        """
        Check if moving to a point would be dangerous.

        Args:
            point: Point to check

        Returns:
            True if the point is dangerous (wall or body)
        """
        return self._is_collision(point)

    def get_state(self) -> Dict[str, Any]:
        """
        Get current game state for rendering or AI.

        Returns:
            Dictionary containing full game state
        """
        return {
            "snake": [p.to_dict() for p in self.snake],
            "food": self.food.to_dict(),
            "direction": int(self.direction),
            "score": self.score,
            "game_over": self.game_over,
            "frame": self.frame_count,
            "width": self.width,
            "height": self.height,
        }

    def get_head_neighbors(self) -> Dict[str, Point]:
        """Get the points adjacent to the snake's head."""
        return {
            "left": Point(self.head.x - 1, self.head.y),
            "right": Point(self.head.x + 1, self.head.y),
            "up": Point(self.head.x, self.head.y - 1),
            "down": Point(self.head.x, self.head.y + 1),
        }
