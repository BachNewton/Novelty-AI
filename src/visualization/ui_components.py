"""
Reusable UI components for the unified interface.

Provides Button, Dropdown, and Toggle classes for building menus and interfaces.
"""
import pygame
from typing import Callable, List, Optional, Tuple


# UI Theme Colors
BG_COLOR = (25, 25, 35)
PANEL_COLOR = (35, 35, 45)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (100, 200, 100)
WARNING_COLOR = (255, 200, 50)
DANGER_COLOR = (255, 100, 100)
CHART_BG = (35, 35, 45)
HOVER_COLOR = (50, 50, 65)
BUTTON_COLOR = (45, 45, 60)
BUTTON_HOVER = (60, 60, 80)
BORDER_COLOR = (70, 70, 85)


class Button:
    """Clickable button with hover effects."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
        callback: Optional[Callable] = None,
        font_size: int = 28
    ):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.font_size = font_size
        self.hovered = False
        self._font = None

    @property
    def font(self):
        if self._font is None:
            self._font = pygame.font.Font(None, self.font_size)
        return self._font

    def set_position(self, x: int, y: int):
        """Update button position."""
        self.rect.x = x
        self.rect.y = y

    def set_size(self, width: int, height: int):
        """Update button size."""
        self.rect.width = width
        self.rect.height = height

    def draw(self, surface: pygame.Surface):
        """Draw the button."""
        color = BUTTON_HOVER if self.hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, self.rect, 2, border_radius=8)

        text_surf = self.font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.

        Returns:
            True if button was clicked, False otherwise
        """
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()
                return True
        return False


class Dropdown:
    """Dropdown selector for models/replays."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        items: List[str],
        label: str = "",
        font_size: int = 24,
        max_visible: int = 5
    ):
        self.x = x
        self.y = y
        self.width = width
        self.item_height = 32
        self.label = label
        self.font_size = font_size
        self.max_visible = max_visible

        self.items = items if items else ["(none)"]
        self.selected_index = 0
        self.expanded = False
        self.hovered_index = -1
        self.scroll_offset = 0

        self._font = None
        self._label_font = None

    @property
    def font(self):
        if self._font is None:
            self._font = pygame.font.Font(None, self.font_size)
        return self._font

    @property
    def label_font(self):
        if self._label_font is None:
            self._label_font = pygame.font.Font(None, self.font_size)
        return self._label_font

    def set_position(self, x: int, y: int):
        """Update dropdown position."""
        self.x = x
        self.y = y

    def set_width(self, width: int):
        """Update dropdown width."""
        self.width = width

    @property
    def header_rect(self) -> pygame.Rect:
        """Get the header rectangle."""
        return pygame.Rect(self.x, self.y, self.width, self.item_height)

    @property
    def dropdown_rect(self) -> pygame.Rect:
        """Get the full dropdown area when expanded."""
        visible_count = min(len(self.items), self.max_visible)
        height = self.item_height * visible_count
        return pygame.Rect(self.x, self.y + self.item_height, self.width, height)

    def draw(self, surface: pygame.Surface):
        """Draw the dropdown."""
        # Draw label
        if self.label:
            label_surf = self.label_font.render(self.label, True, TEXT_COLOR)
            surface.blit(label_surf, (self.x - label_surf.get_width() - 10, self.y + 6))

        # Draw header (selected item)
        header = self.header_rect
        pygame.draw.rect(surface, BUTTON_COLOR, header, border_radius=6)
        pygame.draw.rect(surface, BORDER_COLOR, header, 2, border_radius=6)

        # Selected text
        selected_text = self.items[self.selected_index]
        if len(selected_text) > 35:
            selected_text = selected_text[:32] + "..."
        text_surf = self.font.render(selected_text, True, TEXT_COLOR)
        surface.blit(text_surf, (header.x + 10, header.y + 6))

        # Arrow indicator
        arrow = "v" if not self.expanded else "^"
        arrow_surf = self.font.render(arrow, True, TEXT_COLOR)
        surface.blit(arrow_surf, (header.right - 25, header.y + 6))

        # Draw expanded list
        if self.expanded:
            dropdown = self.dropdown_rect
            pygame.draw.rect(surface, PANEL_COLOR, dropdown, border_radius=6)
            pygame.draw.rect(surface, BORDER_COLOR, dropdown, 2, border_radius=6)

            visible_count = min(len(self.items), self.max_visible)
            for i in range(visible_count):
                item_index = self.scroll_offset + i
                if item_index >= len(self.items):
                    break

                item_rect = pygame.Rect(
                    self.x,
                    self.y + self.item_height * (i + 1),
                    self.width,
                    self.item_height
                )

                # Highlight hovered item
                if item_index == self.hovered_index:
                    pygame.draw.rect(surface, HOVER_COLOR, item_rect)

                # Item text
                item_text = self.items[item_index]
                if len(item_text) > 35:
                    item_text = item_text[:32] + "..."
                item_surf = self.font.render(item_text, True, TEXT_COLOR)
                surface.blit(item_surf, (item_rect.x + 10, item_rect.y + 6))

            # Scroll indicators
            if self.scroll_offset > 0:
                up_surf = self.font.render("^", True, (100, 100, 100))
                surface.blit(up_surf, (dropdown.right - 20, dropdown.top + 5))

            if self.scroll_offset + visible_count < len(self.items):
                down_surf = self.font.render("v", True, (100, 100, 100))
                surface.blit(down_surf, (dropdown.right - 20, dropdown.bottom - 20))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.

        Returns:
            True if selection changed, False otherwise
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.header_rect.collidepoint(event.pos):
                self.expanded = not self.expanded
                return False

            if self.expanded and self.dropdown_rect.collidepoint(event.pos):
                # Calculate clicked item
                rel_y = event.pos[1] - self.y - self.item_height
                item_index = self.scroll_offset + rel_y // self.item_height
                if 0 <= item_index < len(self.items):
                    self.selected_index = item_index
                    self.expanded = False
                    return True

            # Click outside - close dropdown
            self.expanded = False

        elif event.type == pygame.MOUSEMOTION:
            if self.expanded and self.dropdown_rect.collidepoint(event.pos):
                rel_y = event.pos[1] - self.y - self.item_height
                self.hovered_index = self.scroll_offset + rel_y // self.item_height
            else:
                self.hovered_index = -1

        elif event.type == pygame.MOUSEWHEEL and self.expanded:
            if self.dropdown_rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_offset -= event.y
                max_scroll = max(0, len(self.items) - self.max_visible)
                self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))

        return False

    def get_selected(self) -> Optional[str]:
        """Get currently selected item."""
        if 0 <= self.selected_index < len(self.items):
            return self.items[self.selected_index]
        return None

    def get_selected_index(self) -> int:
        """Get currently selected index."""
        return self.selected_index

    def refresh_items(self, items: List[str]):
        """Update the list of items."""
        self.items = items if items else ["(none)"]
        self.selected_index = 0
        self.scroll_offset = 0
        self.expanded = False


class Toggle:
    """On/off toggle switch."""

    def __init__(
        self,
        x: int,
        y: int,
        label: str,
        initial_state: bool = False,
        font_size: int = 24
    ):
        self.x = x
        self.y = y
        self.label = label
        self.font_size = font_size
        self.state = initial_state

        self.toggle_width = 50
        self.toggle_height = 26
        self._font = None

    @property
    def font(self):
        if self._font is None:
            self._font = pygame.font.Font(None, self.font_size)
        return self._font

    def set_position(self, x: int, y: int):
        """Update toggle position."""
        self.x = x
        self.y = y

    @property
    def toggle_rect(self) -> pygame.Rect:
        """Get the toggle switch rectangle."""
        return pygame.Rect(self.x, self.y, self.toggle_width, self.toggle_height)

    @property
    def full_rect(self) -> pygame.Rect:
        """Get the full clickable area including label."""
        label_width = self.font.size(self.label)[0] + 10
        return pygame.Rect(
            self.x,
            self.y,
            self.toggle_width + label_width,
            self.toggle_height
        )

    def draw(self, surface: pygame.Surface):
        """Draw the toggle switch."""
        # Draw track
        track = self.toggle_rect
        track_color = ACCENT_COLOR if self.state else (60, 60, 70)
        pygame.draw.rect(surface, track_color, track, border_radius=13)

        # Draw knob
        knob_radius = 10
        knob_x = track.right - knob_radius - 3 if self.state else track.left + knob_radius + 3
        knob_y = track.centery
        pygame.draw.circle(surface, TEXT_COLOR, (knob_x, knob_y), knob_radius)

        # Draw label
        label_surf = self.font.render(self.label, True, TEXT_COLOR)
        surface.blit(label_surf, (track.right + 10, self.y + 3))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.

        Returns:
            True if state changed, False otherwise
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.full_rect.collidepoint(event.pos):
                self.state = not self.state
                return True
        return False

    def is_on(self) -> bool:
        """Get current toggle state."""
        return self.state

    def set_state(self, state: bool):
        """Set toggle state."""
        self.state = state
