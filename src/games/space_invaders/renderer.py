"""
Space Invaders Game Renderer - Pygame-based visualization implementing RendererInterface.
Classic retro pixel art style.
"""

import pygame
from typing import Dict, Any, Tuple, List, Optional

from ...core.renderer_interface import RendererInterface


# Classic Space Invaders colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Game-specific colors
PLAYER_COLOR = GREEN
INVADER_TOP_COLOR = MAGENTA
INVADER_MIDDLE_COLOR = CYAN
INVADER_BOTTOM_COLOR = GREEN
PROJECTILE_COLOR = WHITE
BUNKER_COLOR = GREEN
MYSTERY_SHIP_COLOR = RED
TEXT_COLOR = WHITE
HUD_COLOR = GREEN


class SpaceInvadersRenderer(RendererInterface):
    """
    Renders Space Invaders using Pygame, implementing RendererInterface.

    Classic retro pixel art style with simple geometric shapes representing
    the player, invaders, bunkers, and projectiles.
    """

    def __init__(self, width: int = 224, height: int = 256):
        """
        Initialize the renderer.

        Args:
            width: Game width in pixels
            height: Game height in pixels
        """
        self._base_width = width
        self._base_height = height
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._render_width = width
        self._render_height = height
        self._animation_frame = 0
        self._font: Optional[pygame.font.Font] = None

    def get_preferred_size(self) -> Tuple[int, int]:
        """Get the preferred render size."""
        return (self._base_width * 2, self._base_height * 2)  # 2x scale for visibility

    def get_cell_size(self) -> int:
        """Get the current cell/unit size."""
        return int(8 * self._scale)

    def set_cell_size(self, cell_size: int) -> None:
        """Set the cell size (adjusts scale)."""
        self._scale = cell_size / 8.0
        self._render_width = int(self._base_width * self._scale)
        self._render_height = int(self._base_height * self._scale)

    def set_render_area(self, x: int, y: int, width: int, height: int) -> None:
        """Set the area where this renderer should draw."""
        self._offset_x = x
        self._offset_y = y
        # Calculate scale to fit
        scale_x = width / self._base_width
        scale_y = height / self._base_height
        self._scale = min(scale_x, scale_y)
        self._render_width = int(self._base_width * self._scale)
        self._render_height = int(self._base_height * self._scale)

    def _scale_x(self, x: float) -> int:
        """Scale X coordinate."""
        return int(self._offset_x + x * self._scale)

    def _scale_y(self, y: float) -> int:
        """Scale Y coordinate."""
        return int(self._offset_y + y * self._scale)

    def _scale_size(self, size: float) -> int:
        """Scale a size value."""
        return max(1, int(size * self._scale))

    def render(self, game_state: Dict[str, Any], surface: pygame.Surface) -> None:
        """
        Render the game state to a surface.

        Args:
            game_state: Dictionary containing game state
            surface: Pygame surface to draw on
        """
        # Get animation phase from game state frame count
        frame = game_state.get("frame", 0)

        # Draw background
        bg_rect = pygame.Rect(
            self._offset_x,
            self._offset_y,
            self._render_width,
            self._render_height,
        )
        pygame.draw.rect(surface, BLACK, bg_rect)

        # Draw bunkers
        self._draw_bunkers(surface, game_state.get("bunkers", []))

        # Draw invaders
        self._draw_invaders(surface, game_state.get("invaders", []), frame)

        # Draw mystery ship
        mystery = game_state.get("mystery_ship", {})
        if mystery.get("active", False):
            self._draw_mystery_ship(surface, mystery, frame)

        # Draw player
        self._draw_player(surface, game_state.get("player", {}))

        # Draw projectiles
        player_proj = game_state.get("player_projectile")
        if player_proj:
            self._draw_projectile(surface, player_proj, is_player=True)

        for proj in game_state.get("invader_projectiles", []):
            self._draw_projectile(surface, proj, is_player=False)

        # Draw HUD
        self._draw_hud(
            surface,
            game_state.get("score", 0),
            game_state.get("lives", 3),
            game_state.get("wave", 1),
        )

        # Draw game over text
        if game_state.get("game_over", False):
            self._draw_game_over(surface)

    def _draw_player(self, surface: pygame.Surface, player: Dict[str, Any]) -> None:
        """Draw the player's laser cannon."""
        if not player:
            return

        x = player.get("x", 0)
        y = player.get("y", 0)
        width = player.get("width", 26)
        height = player.get("height", 16)

        # Draw cannon base (rectangle)
        base_width = self._scale_size(width)
        base_height = self._scale_size(height * 0.5)
        base_rect = pygame.Rect(
            self._scale_x(x - width / 2),
            self._scale_y(y),
            base_width,
            base_height,
        )
        pygame.draw.rect(surface, PLAYER_COLOR, base_rect)

        # Draw cannon turret (narrower rectangle on top)
        turret_width = self._scale_size(width * 0.3)
        turret_height = self._scale_size(height * 0.5)
        turret_rect = pygame.Rect(
            self._scale_x(x - width * 0.15),
            self._scale_y(y - height * 0.5),
            turret_width,
            turret_height,
        )
        pygame.draw.rect(surface, PLAYER_COLOR, turret_rect)

        # Draw cannon barrel (small rectangle on top)
        barrel_width = self._scale_size(4)
        barrel_height = self._scale_size(6)
        barrel_rect = pygame.Rect(
            self._scale_x(x - 2),
            self._scale_y(y - height * 0.5 - 6),
            barrel_width,
            barrel_height,
        )
        pygame.draw.rect(surface, PLAYER_COLOR, barrel_rect)

    def _draw_invaders(
        self, surface: pygame.Surface, invaders: List[List[Dict[str, Any]]], frame: int = 0
    ) -> None:
        """Draw all invaders."""
        animation_phase = frame % 2  # Toggle every game step

        for row in invaders:
            for inv in row:
                if inv.get("alive", False):
                    self._draw_invader(surface, inv, animation_phase)

    def _draw_invader(
        self, surface: pygame.Surface, inv: Dict[str, Any], animation_phase: int
    ) -> None:
        """Draw a single invader with classic pixel art style."""
        x = inv.get("x", 0)
        y = inv.get("y", 0)
        inv_type = inv.get("type", 2)
        width = inv.get("width", 22)
        height = inv.get("height", 16)

        # Choose color based on type
        if inv_type == 0:  # TOP (30 points)
            color = INVADER_TOP_COLOR
        elif inv_type == 1:  # MIDDLE (20 points)
            color = INVADER_MIDDLE_COLOR
        else:  # BOTTOM (10 points)
            color = INVADER_BOTTOM_COLOR

        # Draw simple pixelated invader shape
        # Body
        body_width = self._scale_size(width * 0.8)
        body_height = self._scale_size(height * 0.6)
        body_rect = pygame.Rect(
            self._scale_x(x - width * 0.4),
            self._scale_y(y - height * 0.3),
            body_width,
            body_height,
        )
        pygame.draw.rect(surface, color, body_rect)

        # Head (smaller rectangle on top)
        head_width = self._scale_size(width * 0.4)
        head_height = self._scale_size(height * 0.3)
        head_rect = pygame.Rect(
            self._scale_x(x - width * 0.2),
            self._scale_y(y - height * 0.5),
            head_width,
            head_height,
        )
        pygame.draw.rect(surface, color, head_rect)

        # Legs/tentacles (animate between two positions)
        leg_width = self._scale_size(4)
        leg_height = self._scale_size(height * 0.3)

        if animation_phase == 0:
            # Legs down
            left_leg = pygame.Rect(
                self._scale_x(x - width * 0.35),
                self._scale_y(y + height * 0.2),
                leg_width,
                leg_height,
            )
            right_leg = pygame.Rect(
                self._scale_x(x + width * 0.25),
                self._scale_y(y + height * 0.2),
                leg_width,
                leg_height,
            )
        else:
            # Legs out
            left_leg = pygame.Rect(
                self._scale_x(x - width * 0.45),
                self._scale_y(y + height * 0.1),
                leg_width,
                leg_height,
            )
            right_leg = pygame.Rect(
                self._scale_x(x + width * 0.35),
                self._scale_y(y + height * 0.1),
                leg_width,
                leg_height,
            )

        pygame.draw.rect(surface, color, left_leg)
        pygame.draw.rect(surface, color, right_leg)

        # Eyes (small dots)
        eye_size = max(2, self._scale_size(3))
        left_eye_pos = (self._scale_x(x - 4), self._scale_y(y - height * 0.2))
        right_eye_pos = (self._scale_x(x + 4), self._scale_y(y - height * 0.2))
        pygame.draw.circle(surface, BLACK, left_eye_pos, eye_size)
        pygame.draw.circle(surface, BLACK, right_eye_pos, eye_size)

    def _draw_mystery_ship(
        self, surface: pygame.Surface, mystery: Dict[str, Any], frame: int = 0
    ) -> None:
        """Draw the mystery UFO ship."""
        x = mystery.get("x", 0)
        y = mystery.get("y", 0)
        width = mystery.get("width", 32)
        height = mystery.get("height", 14)

        # Draw saucer shape
        # Main body (ellipse-like using rectangles)
        body_width = self._scale_size(width)
        body_height = self._scale_size(height * 0.5)
        body_rect = pygame.Rect(
            self._scale_x(x - width / 2),
            self._scale_y(y),
            body_width,
            body_height,
        )
        pygame.draw.rect(surface, MYSTERY_SHIP_COLOR, body_rect)

        # Dome on top
        dome_width = self._scale_size(width * 0.4)
        dome_height = self._scale_size(height * 0.4)
        dome_rect = pygame.Rect(
            self._scale_x(x - width * 0.2),
            self._scale_y(y - height * 0.3),
            dome_width,
            dome_height,
        )
        pygame.draw.rect(surface, MYSTERY_SHIP_COLOR, dome_rect)

        # Blinking lights (blink every 5 game steps)
        if (frame // 5) % 2 == 0:
            light_size = max(2, self._scale_size(3))
            pygame.draw.circle(
                surface,
                YELLOW,
                (self._scale_x(x - width * 0.3), self._scale_y(y + height * 0.2)),
                light_size,
            )
            pygame.draw.circle(
                surface,
                YELLOW,
                (self._scale_x(x + width * 0.3), self._scale_y(y + height * 0.2)),
                light_size,
            )

    def _draw_bunkers(
        self, surface: pygame.Surface, bunkers: List[Dict[str, Any]]
    ) -> None:
        """Draw all defensive bunkers."""
        cell_size = 2  # Classic: 2px cells for 22×16 bunker

        for bunker in bunkers:
            bunker_x = bunker.get("x", 0)
            bunker_y = bunker.get("y", 0)
            cells = bunker.get("cells", [])

            for row_idx, row in enumerate(cells):
                for col_idx, cell in enumerate(row):
                    if cell and cell.get("health", 0) > 0:
                        health = cell["health"]
                        # Vary color based on health
                        intensity = int(255 * health / 4)
                        color = (0, intensity, 0)

                        cell_rect = pygame.Rect(
                            self._scale_x(bunker_x + col_idx * cell_size),
                            self._scale_y(bunker_y + row_idx * cell_size),
                            self._scale_size(cell_size),
                            self._scale_size(cell_size),
                        )
                        pygame.draw.rect(surface, color, cell_rect)

    def _draw_projectile(
        self, surface: pygame.Surface, proj: Dict[str, Any], is_player: bool
    ) -> None:
        """Draw a projectile."""
        x = proj.get("x", 0)
        y = proj.get("y", 0)
        width = proj.get("width", 3)
        height = proj.get("height", 10)

        color = WHITE if is_player else RED

        proj_rect = pygame.Rect(
            self._scale_x(x - width / 2),
            self._scale_y(y - height / 2),
            self._scale_size(width),
            self._scale_size(height),
        )
        pygame.draw.rect(surface, color, proj_rect)

    def _draw_hud(
        self, surface: pygame.Surface, score: int, lives: int, wave: int
    ) -> None:
        """Draw the heads-up display (score, lives, wave)."""
        # Initialize font if needed
        if self._font is None:
            try:
                self._font = pygame.font.Font(None, self._scale_size(20))
            except Exception:
                return

        # Score
        score_text = self._font.render(f"SCORE: {score}", True, HUD_COLOR)
        surface.blit(score_text, (self._scale_x(5), self._scale_y(5)))

        # Wave
        wave_text = self._font.render(f"WAVE: {wave}", True, HUD_COLOR)
        wave_rect = wave_text.get_rect()
        wave_rect.centerx = self._scale_x(self._base_width / 2)
        wave_rect.top = self._scale_y(5)
        surface.blit(wave_text, wave_rect)

        # Lives (draw small player icons)
        lives_text = self._font.render("LIVES:", True, HUD_COLOR)
        lives_x = self._scale_x(self._base_width - 80)
        surface.blit(lives_text, (lives_x, self._scale_y(5)))

        for i in range(lives):
            icon_x = lives_x + 50 + i * 15
            icon_rect = pygame.Rect(
                icon_x, self._scale_y(8), self._scale_size(10), self._scale_size(8)
            )
            pygame.draw.rect(surface, PLAYER_COLOR, icon_rect)

    def _draw_game_over(self, surface: pygame.Surface) -> None:
        """Draw game over text."""
        if self._font is None:
            try:
                self._font = pygame.font.Font(None, self._scale_size(20))
            except Exception:
                return

        # Create larger font for game over
        try:
            large_font = pygame.font.Font(None, self._scale_size(40))
        except Exception:
            large_font = self._font

        text = large_font.render("GAME OVER", True, RED)
        text_rect = text.get_rect()
        text_rect.center = (
            self._scale_x(self._base_width / 2),
            self._scale_y(self._base_height / 2),
        )
        surface.blit(text, text_rect)

    def render_icon(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        """
        Render a small Space Invaders icon for the game hub.

        Args:
            surface: Pygame surface to draw on
            rect: Rectangle area to draw the icon in
        """
        # Draw background
        pygame.draw.rect(surface, BLACK, rect, border_radius=8)

        # Draw a simple invader shape
        padding = 15
        inner_rect = pygame.Rect(
            rect.x + padding,
            rect.y + padding,
            rect.width - 2 * padding,
            rect.height - 2 * padding,
        )

        cx = inner_rect.centerx
        cy = inner_rect.centery
        size = min(inner_rect.width, inner_rect.height) // 3

        # Draw invader body
        body_rect = pygame.Rect(cx - size, cy - size // 2, size * 2, size)
        pygame.draw.rect(surface, INVADER_MIDDLE_COLOR, body_rect)

        # Draw invader head
        head_rect = pygame.Rect(cx - size // 2, cy - size, size, size // 2)
        pygame.draw.rect(surface, INVADER_MIDDLE_COLOR, head_rect)

        # Draw legs
        leg_size = size // 3
        left_leg = pygame.Rect(cx - size - leg_size // 2, cy + size // 2, leg_size, leg_size)
        right_leg = pygame.Rect(cx + size - leg_size // 2, cy + size // 2, leg_size, leg_size)
        pygame.draw.rect(surface, INVADER_MIDDLE_COLOR, left_leg)
        pygame.draw.rect(surface, INVADER_MIDDLE_COLOR, right_leg)

        # Draw player at bottom
        player_width = size * 2
        player_height = size // 2
        player_rect = pygame.Rect(
            cx - player_width // 2,
            rect.bottom - padding - player_height,
            player_width,
            player_height,
        )
        pygame.draw.rect(surface, PLAYER_COLOR, player_rect)

        # Draw projectile
        proj_rect = pygame.Rect(cx - 2, cy + size + 5, 4, 10)
        pygame.draw.rect(surface, WHITE, proj_rect)
