"""
Game Hub - Main game selection screen for Novelty AI.

Displays available games as visual cards with icons.
"""

import pygame
from typing import Dict, Any, Tuple, List, Optional

from .ui_components import (
    Button, BG_COLOR, TEXT_COLOR, ACCENT_COLOR, ACCENT_ORANGE,
    PANEL_COLOR, HOVER_COLOR, BORDER_COLOR, DISABLED_COLOR
)
from ..games.registry import GameRegistry
from ..core.game_interface import GameMetadata


class GameCard:
    """
    Clickable card representing a game.

    Displays game icon, name, and description.
    """

    def __init__(
        self,
        metadata: GameMetadata,
        x: int,
        y: int,
        width: int = 200,
        height: int = 250,
        available: bool = True
    ):
        """
        Initialize game card.

        Args:
            metadata: Game metadata
            x: X position
            y: Y position
            width: Card width
            height: Card height
            available: Whether game is playable (not a placeholder)
        """
        self.metadata = metadata
        self.rect = pygame.Rect(x, y, width, height)
        self.available = available
        self.hovered = False
        self._font_name = None
        self._font_desc = None

    @property
    def font_name(self):
        if self._font_name is None:
            self._font_name = pygame.font.Font(None, 32)
        return self._font_name

    @property
    def font_desc(self):
        if self._font_desc is None:
            self._font_desc = pygame.font.Font(None, 20)
        return self._font_desc

    def set_position(self, x: int, y: int):
        """Update card position."""
        self.rect.x = x
        self.rect.y = y

    def draw(self, surface: pygame.Surface, renderer=None):
        """
        Draw the game card.

        Args:
            surface: Surface to draw on
            renderer: Optional renderer for game icon
        """
        # Card background
        bg_color = PANEL_COLOR if self.available else (35, 38, 45)
        if self.hovered and self.available:
            bg_color = HOVER_COLOR
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=12)

        # Border (accent color when hovered)
        border_color = ACCENT_COLOR if (self.hovered and self.available) else BORDER_COLOR
        if not self.available:
            border_color = DISABLED_COLOR
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=12)

        # Icon area
        icon_rect = pygame.Rect(
            self.rect.x + 20,
            self.rect.y + 20,
            self.rect.width - 40,
            self.rect.height - 100
        )

        if self.available and renderer:
            # Use game-specific renderer for icon
            renderer.render_icon(surface, icon_rect)
        else:
            # Placeholder icon
            pygame.draw.rect(surface, (50, 54, 62), icon_rect, border_radius=8)
            if not self.available:
                # "Coming Soon" text
                coming_text = self.font_desc.render("Coming", True, ACCENT_ORANGE)
                soon_text = self.font_desc.render("Soon", True, ACCENT_ORANGE)
                surface.blit(coming_text, (
                    icon_rect.centerx - coming_text.get_width() // 2,
                    icon_rect.centery - 15
                ))
                surface.blit(soon_text, (
                    icon_rect.centerx - soon_text.get_width() // 2,
                    icon_rect.centery + 5
                ))

        # Game name
        name_color = ACCENT_COLOR if self.available else DISABLED_COLOR
        name_surf = self.font_name.render(self.metadata.name, True, name_color)
        name_x = self.rect.centerx - name_surf.get_width() // 2
        surface.blit(name_surf, (name_x, self.rect.bottom - 70))

        # Description (truncated)
        desc = self.metadata.description
        if len(desc) > 35:
            desc = desc[:32] + "..."
        desc_color = TEXT_COLOR if self.available else DISABLED_COLOR
        desc_surf = self.font_desc.render(desc, True, desc_color)
        desc_x = self.rect.centerx - desc_surf.get_width() // 2
        surface.blit(desc_surf, (desc_x, self.rect.bottom - 40))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.

        Returns:
            True if card was clicked (and is available)
        """
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and self.available:
                return True
        return False


class GameHub:
    """
    Main game selection hub for Novelty AI.

    Displays all available games as cards and allows selection.
    """

    MIN_WIDTH = 800
    MIN_HEIGHT = 600

    def __init__(
        self,
        screen: Optional[pygame.Surface] = None,
        window_width: int = 1400,
        window_height: int = 900
    ):
        """
        Initialize the game hub.

        Args:
            screen: Optional existing pygame surface
            window_width: Window width
            window_height: Window height
        """
        self.owns_screen = screen is None

        if screen is None:
            pygame.init()
            self.window_width = max(window_width, self.MIN_WIDTH)
            self.window_height = max(window_height, self.MIN_HEIGHT)
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height),
                pygame.RESIZABLE
            )
            pygame.display.set_caption("Novelty AI")
        else:
            self.screen = screen
            self.window_width = screen.get_width()
            self.window_height = screen.get_height()

        # Fonts
        self._init_fonts()

        # Get games and create cards
        self._create_game_cards()

        # Bottom buttons
        self.btn_settings = Button(0, 0, 150, 50, "Settings", font_size=24)
        self.btn_quit = Button(0, 0, 150, 50, "Quit", font_size=24)

        # Renderers for game icons
        self._renderers: Dict[str, Any] = {}

        # Calculate layout
        self._recalculate_layout()

        self.clock = pygame.time.Clock()

    def _init_fonts(self):
        """Initialize fonts."""
        self.font_title = pygame.font.Font(None, 72)
        self.font_subtitle = pygame.font.Font(None, 32)

    def _create_game_cards(self):
        """Create game cards for all games."""
        self.game_cards: List[GameCard] = []

        # Get available games
        available_games = GameRegistry.list_games()
        all_games = GameRegistry.list_all_games(include_placeholders=True)

        for metadata in all_games:
            is_available = metadata.id in [g.id for g in available_games]
            card = GameCard(
                metadata=metadata,
                x=0, y=0,
                available=is_available
            )
            self.game_cards.append(card)

    def _recalculate_layout(self):
        """Recalculate positions based on window size."""
        w, h = self.window_width, self.window_height
        cx = w // 2

        # Game cards grid
        card_width = 200
        card_height = 250
        card_gap = 30
        num_cards = len(self.game_cards)

        # Calculate grid layout
        total_width = num_cards * card_width + (num_cards - 1) * card_gap
        start_x = cx - total_width // 2

        for i, card in enumerate(self.game_cards):
            x = start_x + i * (card_width + card_gap)
            y = 200
            card.set_position(x, y)

        # Bottom buttons
        btn_y = h - 100
        self.btn_settings.set_position(cx - 160, btn_y)
        self.btn_quit.set_position(cx + 10, btn_y)

    def _get_renderer(self, game_id: str):
        """Get or create renderer for a game."""
        if game_id not in self._renderers:
            try:
                renderer = GameRegistry.create_renderer(game_id)
                self._renderers[game_id] = renderer
            except ValueError:
                return None
        return self._renderers.get(game_id)

    def run(self) -> Tuple[str, Dict[str, Any]]:
        """
        Run the hub until a game is selected.

        Returns:
            Tuple of (game_id, options) or ('quit', {}) or ('settings', {})
        """
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return ('quit', {})

                elif event.type == pygame.VIDEORESIZE:
                    self.window_width = max(event.w, self.MIN_WIDTH)
                    self.window_height = max(event.h, self.MIN_HEIGHT)
                    if self.owns_screen:
                        self.screen = pygame.display.set_mode(
                            (self.window_width, self.window_height),
                            pygame.RESIZABLE
                        )
                    self._recalculate_layout()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return ('quit', {})

                # Handle game card clicks
                for card in self.game_cards:
                    if card.handle_event(event):
                        return (card.metadata.id, {})

                # Handle button clicks
                if self.btn_quit.handle_event(event):
                    return ('quit', {})
                if self.btn_settings.handle_event(event):
                    return ('settings', {})

            self._draw()
            self.clock.tick(60)

        return ('quit', {})

    def _draw(self):
        """Render the hub."""
        self.screen.fill(BG_COLOR)

        w = self.window_width
        cx = w // 2

        # Title
        title = self.font_title.render("NOVELTY AI", True, ACCENT_COLOR)
        title_rect = title.get_rect(centerx=cx, y=40)
        self.screen.blit(title, title_rect)

        # Subtitle
        subtitle = self.font_subtitle.render("AI Training Hub", True, TEXT_COLOR)
        subtitle_rect = subtitle.get_rect(centerx=cx, y=110)
        self.screen.blit(subtitle, subtitle_rect)

        # Game cards
        for card in self.game_cards:
            renderer = self._get_renderer(card.metadata.id) if card.available else None
            card.draw(self.screen, renderer)

        # Buttons
        self.btn_settings.draw(self.screen)
        self.btn_quit.draw(self.screen)

        # Footer
        footer = pygame.font.Font(None, 20).render(
            "ESC to quit | Window is resizable",
            True, (100, 105, 115)
        )
        footer_rect = footer.get_rect(centerx=cx, y=self.window_height - 30)
        self.screen.blit(footer, footer_rect)

        pygame.display.flip()

    def close(self):
        """Clean up resources."""
        if self.owns_screen:
            pygame.quit()
