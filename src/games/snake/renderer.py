"""
Snake Game Renderer - Pygame-based visualization implementing RendererInterface.
"""

import pygame
from typing import Dict, Any, Optional, Tuple

from ...core.renderer_interface import RendererInterface


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (30, 30, 40)
GRID_COLOR = (50, 50, 60)
SNAKE_HEAD_COLOR = (0, 220, 100)
SNAKE_BODY_COLOR = (0, 180, 80)
FOOD_COLOR = (220, 50, 50)
TEXT_COLOR = (220, 220, 220)


class SnakeRenderer(RendererInterface):
    """
    Renders the Snake game using Pygame, implementing RendererInterface.

    Can render to a surface (for embedding in dashboard) or
    manage its own window (for standalone play).
    """

    def __init__(
        self,
        cell_size: int = 25,
        grid_width: int = 20,
        grid_height: int = 20
    ):
        """
        Initialize the renderer.

        Args:
            cell_size: Size of each grid cell in pixels
            grid_width: Grid width in cells
            grid_height: Grid height in cells
        """
        self._cell_size = cell_size
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._offset_x = 0
        self._offset_y = 0
        self._render_width = grid_width * cell_size
        self._render_height = grid_height * cell_size

    def get_preferred_size(self) -> Tuple[int, int]:
        """Get the preferred render size."""
        return (self._render_width, self._render_height)

    def get_cell_size(self) -> int:
        """Get the current cell size."""
        return self._cell_size

    def set_cell_size(self, cell_size: int) -> None:
        """Set the cell size."""
        self._cell_size = cell_size
        self._render_width = self._grid_width * cell_size
        self._render_height = self._grid_height * cell_size

    def set_render_area(self, x: int, y: int, width: int, height: int) -> None:
        """Set the area where this renderer should draw."""
        self._offset_x = x
        self._offset_y = y
        # Adjust cell size to fit the area
        cell_w = width // self._grid_width
        cell_h = height // self._grid_height
        self._cell_size = min(cell_w, cell_h)
        self._render_width = self._grid_width * self._cell_size
        self._render_height = self._grid_height * self._cell_size

    def render(self, game_state: Dict[str, Any], surface: pygame.Surface) -> None:
        """
        Render the game state to a surface.

        Args:
            game_state: Dictionary containing game state
            surface: Pygame surface to draw on
        """
        width = game_state.get("width", self._grid_width)
        height = game_state.get("height", self._grid_height)

        # Calculate render area
        game_width = width * self._cell_size
        game_height = height * self._cell_size

        # Draw background
        game_rect = pygame.Rect(
            self._offset_x, self._offset_y,
            game_width, game_height
        )
        pygame.draw.rect(surface, DARK_GRAY, game_rect)

        # Draw grid lines (subtle)
        for x in range(width + 1):
            start = (self._offset_x + x * self._cell_size, self._offset_y)
            end = (self._offset_x + x * self._cell_size, self._offset_y + game_height)
            pygame.draw.line(surface, GRID_COLOR, start, end)

        for y in range(height + 1):
            start = (self._offset_x, self._offset_y + y * self._cell_size)
            end = (self._offset_x + game_width, self._offset_y + y * self._cell_size)
            pygame.draw.line(surface, GRID_COLOR, start, end)

        # Draw food
        food = game_state["food"]
        food_rect = pygame.Rect(
            self._offset_x + food["x"] * self._cell_size + 2,
            self._offset_y + food["y"] * self._cell_size + 2,
            self._cell_size - 4,
            self._cell_size - 4
        )
        pygame.draw.rect(surface, FOOD_COLOR, food_rect, border_radius=4)

        # Draw snake
        snake = game_state["snake"]
        for i, segment in enumerate(snake):
            color = SNAKE_HEAD_COLOR if i == 0 else SNAKE_BODY_COLOR
            seg_rect = pygame.Rect(
                self._offset_x + segment["x"] * self._cell_size + 1,
                self._offset_y + segment["y"] * self._cell_size + 1,
                self._cell_size - 2,
                self._cell_size - 2
            )
            border_radius = 6 if i == 0 else 3
            pygame.draw.rect(surface, color, seg_rect, border_radius=border_radius)

            # Draw eyes on head
            if i == 0:
                self._draw_eyes(surface, segment, game_state.get("direction", 0))

    def _draw_eyes(self, surface: pygame.Surface, head: Dict[str, int], direction: int):
        """Draw eyes on the snake's head."""
        cx = self._offset_x + head["x"] * self._cell_size + self._cell_size // 2
        cy = self._offset_y + head["y"] * self._cell_size + self._cell_size // 2

        eye_radius = max(2, self._cell_size // 8)
        eye_offset = self._cell_size // 4

        # Position eyes based on direction
        if direction == 0:  # RIGHT
            positions = [(cx + 2, cy - eye_offset), (cx + 2, cy + eye_offset)]
        elif direction == 1:  # DOWN
            positions = [(cx - eye_offset, cy + 2), (cx + eye_offset, cy + 2)]
        elif direction == 2:  # LEFT
            positions = [(cx - 2, cy - eye_offset), (cx - 2, cy + eye_offset)]
        else:  # UP
            positions = [(cx - eye_offset, cy - 2), (cx + eye_offset, cy - 2)]

        for pos in positions:
            pygame.draw.circle(surface, WHITE, pos, eye_radius)
            pygame.draw.circle(surface, BLACK, pos, eye_radius // 2)

    def render_icon(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        """
        Render a small snake icon for the game hub.

        Args:
            surface: Pygame surface to draw on
            rect: Rectangle area to draw the icon in
        """
        # Draw background
        pygame.draw.rect(surface, DARK_GRAY, rect, border_radius=8)

        # Draw a simple snake shape
        padding = 10
        inner_rect = pygame.Rect(
            rect.x + padding,
            rect.y + padding,
            rect.width - 2 * padding,
            rect.height - 2 * padding
        )

        # Draw snake body segments
        segment_size = min(inner_rect.width, inner_rect.height) // 4
        cx = inner_rect.centerx
        cy = inner_rect.centery

        # Draw body
        body_positions = [
            (cx - segment_size, cy),
            (cx, cy),
            (cx + segment_size, cy),
            (cx + segment_size, cy - segment_size),
        ]

        for i, (x, y) in enumerate(body_positions):
            color = SNAKE_HEAD_COLOR if i == len(body_positions) - 1 else SNAKE_BODY_COLOR
            seg_rect = pygame.Rect(
                x - segment_size // 2 + 2,
                y - segment_size // 2 + 2,
                segment_size - 4,
                segment_size - 4
            )
            pygame.draw.rect(surface, color, seg_rect, border_radius=4)

        # Draw food
        food_rect = pygame.Rect(
            cx + segment_size * 2 - segment_size // 2,
            cy - segment_size - segment_size // 2,
            segment_size - 4,
            segment_size - 4
        )
        pygame.draw.rect(surface, FOOD_COLOR, food_rect, border_radius=4)

    def get_game_size(self, grid_width: int, grid_height: int) -> Tuple[int, int]:
        """
        Get the pixel size needed for a game grid.

        Args:
            grid_width: Grid width in cells
            grid_height: Grid height in cells

        Returns:
            (width, height) in pixels
        """
        return (grid_width * self._cell_size, grid_height * self._cell_size)
