"""
Tetris Game Renderer - Pygame-based visualization implementing RendererInterface.
"""

import pygame
from typing import Dict, Any, Tuple, List, Optional

from ...core.renderer_interface import RendererInterface
from .game import PieceType, TETROMINOES


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (30, 30, 40)
GRID_COLOR = (50, 50, 60)
GHOST_COLOR = (100, 100, 100)
TEXT_COLOR = (220, 220, 220)
BORDER_COLOR = (80, 80, 100)

# Standard Tetris piece colors
PIECE_COLORS: Dict[int, Tuple[int, int, int]] = {
    PieceType.I: (0, 240, 240),    # Cyan
    PieceType.O: (240, 240, 0),    # Yellow
    PieceType.T: (160, 0, 240),    # Purple
    PieceType.S: (0, 240, 0),      # Green
    PieceType.Z: (240, 0, 0),      # Red
    PieceType.J: (0, 0, 240),      # Blue
    PieceType.L: (240, 160, 0),    # Orange
}


class TetrisRenderer(RendererInterface):
    """
    Renders Tetris game using Pygame, implementing RendererInterface.

    Layout:
    [Hold Box] [Main Board] [Preview Queue]
               [Score/Level/Lines]
    """

    def __init__(
        self,
        cell_size: int = 25,
        board_width: int = 10,
        board_height: int = 20
    ):
        """
        Initialize the renderer.

        Args:
            cell_size: Size of each grid cell in pixels
            board_width: Board width in cells
            board_height: Board height in cells
        """
        self._cell_size = cell_size
        self._board_width = board_width
        self._board_height = board_height
        self._offset_x = 0
        self._offset_y = 0

        # Layout dimensions
        self._hold_width = 6 * cell_size  # Hold box width
        self._preview_width = 6 * cell_size  # Preview queue width
        self._info_height = 80  # Score/level/lines area

        self._calculate_dimensions()

    def _calculate_dimensions(self) -> None:
        """Calculate total render dimensions."""
        board_w = self._board_width * self._cell_size
        board_h = self._board_height * self._cell_size

        self._render_width = self._hold_width + board_w + self._preview_width + 20
        self._render_height = board_h + self._info_height

    def get_preferred_size(self) -> Tuple[int, int]:
        """Get the preferred render size."""
        return (self._render_width, self._render_height)

    def get_cell_size(self) -> int:
        """Get current cell size."""
        return self._cell_size

    def set_cell_size(self, cell_size: int) -> None:
        """Set cell size."""
        self._cell_size = cell_size
        self._hold_width = 6 * cell_size
        self._preview_width = 6 * cell_size
        self._calculate_dimensions()

    def set_render_area(self, x: int, y: int, width: int, height: int) -> None:
        """Set the render area."""
        self._offset_x = x
        self._offset_y = y

        # Adjust cell size to fit
        available_height = height - self._info_height
        available_width = width - self._hold_width - self._preview_width - 20

        cell_w = available_width // self._board_width
        cell_h = available_height // self._board_height
        self._cell_size = max(10, min(cell_w, cell_h))

        self._hold_width = 6 * self._cell_size
        self._preview_width = 6 * self._cell_size
        self._calculate_dimensions()

    def render(self, game_state: Dict[str, Any], surface: pygame.Surface) -> None:
        """
        Render the game state to a surface.

        Args:
            game_state: Dictionary containing game state
            surface: Pygame surface to draw on
        """
        width = game_state.get("width", self._board_width)
        height = game_state.get("height", self._board_height)

        # Calculate positions
        hold_x = self._offset_x
        board_x = self._offset_x + self._hold_width + 5
        preview_x = board_x + width * self._cell_size + 5
        info_y = self._offset_y + height * self._cell_size + 5

        # Draw hold box
        self._draw_hold_box(surface, hold_x, self._offset_y, game_state)

        # Draw main board
        self._draw_board(surface, board_x, self._offset_y, game_state)

        # Draw current piece
        if game_state.get("current_piece"):
            self._draw_piece(surface, board_x, self._offset_y, game_state["current_piece"], ghost=False)

        # Draw ghost piece
        if game_state.get("ghost_piece"):
            self._draw_piece(surface, board_x, self._offset_y, game_state["ghost_piece"], ghost=True)

        # Draw preview queue
        self._draw_preview(surface, preview_x, self._offset_y, game_state)

        # Draw info (score, level, lines)
        self._draw_info(surface, board_x, info_y, game_state)

    def _draw_board(self, surface: pygame.Surface, x: int, y: int, game_state: Dict[str, Any]) -> None:
        """Draw the main game board."""
        width = game_state.get("width", self._board_width)
        height = game_state.get("height", self._board_height)
        board = game_state.get("board", [])

        board_w = width * self._cell_size
        board_h = height * self._cell_size

        # Background
        pygame.draw.rect(surface, DARK_GRAY, (x, y, board_w, board_h))

        # Grid lines
        for gx in range(width + 1):
            pygame.draw.line(surface, GRID_COLOR,
                           (x + gx * self._cell_size, y),
                           (x + gx * self._cell_size, y + board_h))
        for gy in range(height + 1):
            pygame.draw.line(surface, GRID_COLOR,
                           (x, y + gy * self._cell_size),
                           (x + board_w, y + gy * self._cell_size))

        # Draw placed blocks
        for row_idx, row in enumerate(board):
            for col_idx, cell in enumerate(row):
                if cell >= 0:  # -1 means empty
                    self._draw_cell(surface,
                                  x + col_idx * self._cell_size,
                                  y + row_idx * self._cell_size,
                                  PIECE_COLORS.get(cell, WHITE))

        # Border
        pygame.draw.rect(surface, BORDER_COLOR, (x, y, board_w, board_h), 2)

    def _draw_piece(self, surface: pygame.Surface, board_x: int, board_y: int,
                   piece_data: Dict[str, Any], ghost: bool = False) -> None:
        """Draw a tetris piece."""
        piece_type = piece_data.get("type", 0)
        px = piece_data.get("x", 0)
        py = piece_data.get("y", 0)
        rotation = piece_data.get("rotation", 0)

        # Get piece cells
        try:
            offsets = TETROMINOES[PieceType(piece_type)][rotation]
        except (KeyError, ValueError):
            return

        color = GHOST_COLOR if ghost else PIECE_COLORS.get(piece_type, WHITE)

        # Adjust y for hidden rows (piece spawns above visible area)
        visible_offset = 4  # Number of hidden rows

        for dx, dy in offsets:
            cell_x = px + dx
            cell_y = py + dy - visible_offset  # Adjust for hidden rows

            if cell_y >= 0:  # Only draw if in visible area
                screen_x = board_x + cell_x * self._cell_size
                screen_y = board_y + cell_y * self._cell_size

                if ghost:
                    # Ghost piece: just outline
                    rect = pygame.Rect(screen_x + 2, screen_y + 2,
                                      self._cell_size - 4, self._cell_size - 4)
                    pygame.draw.rect(surface, color, rect, 2)
                else:
                    self._draw_cell(surface, screen_x, screen_y, color)

    def _draw_cell(self, surface: pygame.Surface, x: int, y: int,
                  color: Tuple[int, int, int]) -> None:
        """Draw a single cell with 3D effect."""
        cs = self._cell_size

        # Main color
        pygame.draw.rect(surface, color, (x + 1, y + 1, cs - 2, cs - 2))

        # Highlight (lighter)
        highlight = tuple(min(255, c + 50) for c in color)
        pygame.draw.line(surface, highlight, (x + 2, y + 2), (x + cs - 3, y + 2))
        pygame.draw.line(surface, highlight, (x + 2, y + 2), (x + 2, y + cs - 3))

        # Shadow (darker)
        shadow = tuple(max(0, c - 50) for c in color)
        pygame.draw.line(surface, shadow, (x + cs - 2, y + 2), (x + cs - 2, y + cs - 2))
        pygame.draw.line(surface, shadow, (x + 2, y + cs - 2), (x + cs - 2, y + cs - 2))

    def _draw_hold_box(self, surface: pygame.Surface, x: int, y: int,
                      game_state: Dict[str, Any]) -> None:
        """Draw the hold box."""
        box_w = self._hold_width - 5
        box_h = 5 * self._cell_size

        # Background
        pygame.draw.rect(surface, DARK_GRAY, (x, y, box_w, box_h))
        pygame.draw.rect(surface, BORDER_COLOR, (x, y, box_w, box_h), 2)

        # Label
        font = pygame.font.Font(None, 20)
        label = font.render("HOLD", True, TEXT_COLOR)
        surface.blit(label, (x + 5, y + 5))

        # Draw held piece
        held = game_state.get("held_piece", -1)
        if held >= 0:
            can_hold = game_state.get("can_hold", True)
            self._draw_mini_piece(surface, x + box_w // 2, y + box_h // 2 + 10,
                                 held, dim=not can_hold)

    def _draw_preview(self, surface: pygame.Surface, x: int, y: int,
                     game_state: Dict[str, Any]) -> None:
        """Draw the preview queue."""
        next_pieces = game_state.get("next_pieces", [])
        box_w = self._preview_width - 5
        box_h = self._board_height * self._cell_size

        # Background
        pygame.draw.rect(surface, DARK_GRAY, (x, y, box_w, box_h))
        pygame.draw.rect(surface, BORDER_COLOR, (x, y, box_w, box_h), 2)

        # Label
        font = pygame.font.Font(None, 20)
        label = font.render("NEXT", True, TEXT_COLOR)
        surface.blit(label, (x + 5, y + 5))

        # Draw next pieces
        piece_spacing = 3 * self._cell_size
        for i, piece_type in enumerate(next_pieces[:5]):
            self._draw_mini_piece(surface, x + box_w // 2,
                                 y + 40 + i * piece_spacing, piece_type)

    def _draw_mini_piece(self, surface: pygame.Surface, cx: int, cy: int,
                        piece_type: int, dim: bool = False) -> None:
        """Draw a small piece preview centered at (cx, cy)."""
        try:
            offsets = TETROMINOES[PieceType(piece_type)][0]  # Use rotation 0
        except (KeyError, ValueError):
            return

        mini_size = self._cell_size * 2 // 3
        color = PIECE_COLORS.get(piece_type, WHITE)

        if dim:
            color = (color[0] // 2, color[1] // 2, color[2] // 2)

        # Calculate bounding box for centering
        min_x = min(ox for ox, oy in offsets)
        max_x = max(ox for ox, oy in offsets)
        min_y = min(oy for ox, oy in offsets)
        max_y = max(oy for ox, oy in offsets)

        width = (max_x - min_x + 1) * mini_size
        height = (max_y - min_y + 1) * mini_size

        start_x = cx - width // 2
        start_y = cy - height // 2

        for ox, oy in offsets:
            x = start_x + (ox - min_x) * mini_size
            y = start_y + (oy - min_y) * mini_size
            pygame.draw.rect(surface, color, (x + 1, y + 1, mini_size - 2, mini_size - 2))

    def _draw_info(self, surface: pygame.Surface, x: int, y: int,
                  game_state: Dict[str, Any]) -> None:
        """Draw score, level, and lines."""
        font = pygame.font.Font(None, 24)

        score = game_state.get("score", 0)
        level = game_state.get("level", 1)
        lines = game_state.get("lines_cleared", 0)

        texts = [
            f"Score: {score}",
            f"Level: {level}",
            f"Lines: {lines}",
        ]

        spacing = self._board_width * self._cell_size // 3

        for i, text in enumerate(texts):
            label = font.render(text, True, TEXT_COLOR)
            surface.blit(label, (x + i * spacing, y + 10))

    def render_icon(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        """Render a small Tetris icon for the game hub."""
        pygame.draw.rect(surface, DARK_GRAY, rect, border_radius=8)

        # Draw some tetris blocks
        padding = 15
        inner_rect = pygame.Rect(
            rect.x + padding,
            rect.y + padding,
            rect.width - 2 * padding,
            rect.height - 2 * padding
        )

        block_size = min(inner_rect.width, inner_rect.height) // 6

        # Draw a stack of blocks resembling a Tetris game
        blocks = [
            # Bottom row (almost complete)
            (0, 5, PieceType.S), (1, 5, PieceType.S), (2, 5, PieceType.T),
            (3, 5, PieceType.T), (4, 5, PieceType.T),
            # Second row
            (0, 4, PieceType.Z), (1, 4, PieceType.Z), (3, 4, PieceType.L),
            # Third row
            (1, 3, PieceType.Z), (3, 3, PieceType.L), (4, 3, PieceType.L),
            # Falling I piece
            (2, 0, PieceType.I), (2, 1, PieceType.I), (2, 2, PieceType.I),
        ]

        cx = inner_rect.centerx
        cy = inner_rect.centery

        for bx, by, piece_type in blocks:
            color = PIECE_COLORS[piece_type]
            x = cx + (bx - 2.5) * block_size
            y = cy + (by - 2.5) * block_size
            pygame.draw.rect(surface, color,
                           (x, y, block_size - 1, block_size - 1))
