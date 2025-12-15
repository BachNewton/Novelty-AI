"""
Abstract renderer interface for Novelty AI.

All game renderers must implement this interface for visualization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]  # Allow import without pygame for headless


class RendererInterface(ABC):
    """
    Abstract renderer for game visualization.

    Renderers draw game state to a pygame surface.
    """

    @abstractmethod
    def render(self, game_state: Dict[str, Any], surface: "pygame.Surface") -> None:
        """
        Render the game state to a surface.

        Args:
            game_state: Dictionary containing game state from get_game_state()
            surface: Pygame surface to draw on
        """
        pass

    @abstractmethod
    def get_preferred_size(self) -> Tuple[int, int]:
        """
        Get the preferred render size.

        Returns:
            Tuple of (width, height) in pixels
        """
        pass

    def set_render_area(self, x: int, y: int, width: int, height: int) -> None:
        """
        Set the area where this renderer should draw.

        Args:
            x: Left edge x coordinate
            y: Top edge y coordinate
            width: Width of render area
            height: Height of render area
        """
        pass

    def render_icon(self, surface: "pygame.Surface", rect: "pygame.Rect") -> None:
        """
        Render a small icon/preview for the game hub.

        Args:
            surface: Pygame surface to draw on
            rect: Rectangle area to draw the icon in
        """
        # Default: just fill with a color
        if pygame is not None:
            pygame.draw.rect(surface, (60, 60, 80), rect, border_radius=8)

    def get_cell_size(self) -> int:
        """
        Get the cell/unit size for grid-based games.

        Returns:
            Cell size in pixels (default 20)
        """
        return 20

    def set_cell_size(self, cell_size: int) -> None:
        """
        Set the cell/unit size for grid-based games.

        Args:
            cell_size: New cell size in pixels
        """
        pass
