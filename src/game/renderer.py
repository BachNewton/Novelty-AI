"""
Snake Game Renderer - Pygame-based visualization.
"""
import pygame
from typing import Dict, Any, Optional, Tuple


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (30, 30, 40)
GRID_COLOR = (50, 50, 60)
SNAKE_HEAD_COLOR = (0, 220, 100)
SNAKE_BODY_COLOR = (0, 180, 80)
FOOD_COLOR = (220, 50, 50)
TEXT_COLOR = (220, 220, 220)


class GameRenderer:
    """
    Renders the Snake game using Pygame.

    Can render to a surface (for embedding in dashboard) or
    manage its own window (for standalone play).
    """

    def __init__(
        self,
        cell_size: int = 25,
        surface: Optional[pygame.Surface] = None,
        offset: Tuple[int, int] = (0, 0)
    ):
        """
        Initialize the renderer.

        Args:
            cell_size: Size of each grid cell in pixels
            surface: Optional surface to render to (if None, creates window)
            offset: (x, y) offset for rendering on the surface
        """
        self.cell_size = cell_size
        self.surface = surface
        self.offset_x, self.offset_y = offset
        self.owns_surface = surface is None

    def render(
        self,
        game_state: Dict[str, Any],
        surface: Optional[pygame.Surface] = None
    ) -> pygame.Surface:
        """
        Render the game state.

        Args:
            game_state: Dictionary containing game state
            surface: Optional surface to render to

        Returns:
            The surface that was rendered to
        """
        target = surface or self.surface

        if target is None:
            raise ValueError("No surface to render to")

        width = game_state.get("width", 20)
        height = game_state.get("height", 20)

        # Calculate render area
        game_width = width * self.cell_size
        game_height = height * self.cell_size

        # Draw background
        game_rect = pygame.Rect(
            self.offset_x, self.offset_y,
            game_width, game_height
        )
        pygame.draw.rect(target, DARK_GRAY, game_rect)

        # Draw grid lines (subtle)
        for x in range(width + 1):
            start = (self.offset_x + x * self.cell_size, self.offset_y)
            end = (self.offset_x + x * self.cell_size, self.offset_y + game_height)
            pygame.draw.line(target, GRID_COLOR, start, end)

        for y in range(height + 1):
            start = (self.offset_x, self.offset_y + y * self.cell_size)
            end = (self.offset_x + game_width, self.offset_y + y * self.cell_size)
            pygame.draw.line(target, GRID_COLOR, start, end)

        # Draw food
        food = game_state["food"]
        food_rect = pygame.Rect(
            self.offset_x + food["x"] * self.cell_size + 2,
            self.offset_y + food["y"] * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        pygame.draw.rect(target, FOOD_COLOR, food_rect, border_radius=4)

        # Draw snake
        snake = game_state["snake"]
        for i, segment in enumerate(snake):
            color = SNAKE_HEAD_COLOR if i == 0 else SNAKE_BODY_COLOR
            seg_rect = pygame.Rect(
                self.offset_x + segment["x"] * self.cell_size + 1,
                self.offset_y + segment["y"] * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            border_radius = 6 if i == 0 else 3
            pygame.draw.rect(target, color, seg_rect, border_radius=border_radius)

            # Draw eyes on head
            if i == 0:
                self._draw_eyes(target, segment, game_state.get("direction", 0))

        return target

    def _draw_eyes(self, surface: pygame.Surface, head: Dict[str, int], direction: int):
        """Draw eyes on the snake's head."""
        cx = self.offset_x + head["x"] * self.cell_size + self.cell_size // 2
        cy = self.offset_y + head["y"] * self.cell_size + self.cell_size // 2

        eye_radius = max(2, self.cell_size // 8)
        eye_offset = self.cell_size // 4

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

    def get_game_size(self, grid_width: int, grid_height: int) -> Tuple[int, int]:
        """
        Get the pixel size needed for a game grid.

        Args:
            grid_width: Grid width in cells
            grid_height: Grid height in cells

        Returns:
            (width, height) in pixels
        """
        return (grid_width * self.cell_size, grid_height * self.cell_size)


class StandaloneRenderer(GameRenderer):
    """
    Standalone game renderer with its own window.
    Used for human play mode.
    """

    def __init__(
        self,
        grid_width: int = 20,
        grid_height: int = 20,
        cell_size: int = 30,
        title: str = "Snake Game"
    ):
        """
        Initialize standalone renderer with its own window.

        Args:
            grid_width: Grid width in cells
            grid_height: Grid height in cells
            cell_size: Size of each cell in pixels
            title: Window title
        """
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Calculate window size with padding
        padding = 40
        window_width = grid_width * cell_size + padding * 2
        window_height = grid_height * cell_size + padding * 2 + 60  # Extra for score

        pygame.init()
        surface = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(title)

        super().__init__(cell_size, surface, (padding, padding))

        self.window_width = window_width
        self.window_height = window_height
        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()

    def render_frame(self, game_state: Dict[str, Any], fps: int = 10) -> bool:
        """
        Render a frame and handle events.

        Args:
            game_state: Current game state
            fps: Frames per second limit

        Returns:
            False if window was closed, True otherwise
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # Clear screen
        self.surface.fill(BLACK)

        # Render game
        self.render(game_state)

        # Render score
        score_text = self.font.render(
            f"Score: {game_state['score']}",
            True, TEXT_COLOR
        )
        self.surface.blit(
            score_text,
            (self.window_width // 2 - score_text.get_width() // 2,
             self.window_height - 50)
        )

        # Update display
        pygame.display.flip()
        self.clock.tick(fps)

        return True

    def close(self):
        """Close the renderer and pygame."""
        pygame.quit()
