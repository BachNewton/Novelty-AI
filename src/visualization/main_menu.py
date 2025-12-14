"""
Main Menu - Unified entry point for all Snake AI modes.

Provides a graphical menu with access to Training, Watch AI, Play Human, and Replays.
"""
import pygame
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

from .ui_components import Button, Dropdown, Toggle, BG_COLOR, TEXT_COLOR, ACCENT_COLOR


class MainMenu:
    """
    Main menu screen providing access to all game modes.

    Layout (resizable, default 1400x900):
    +------------------------------------------+
    |            SNAKE AI                      |
    |                                          |
    |    [Training]      [Watch AI]            |
    |    [Play Human]    [Replays]             |
    |                                          |
    |    Model: [dropdown]                     |
    |    Replay: [dropdown]                    |
    |    [ ] Headless Training                 |
    |                                          |
    |              [Quit]                      |
    +------------------------------------------+
    """

    MIN_WIDTH = 800
    MIN_HEIGHT = 600

    def __init__(
        self,
        screen: Optional[pygame.Surface] = None,
        window_width: int = 1400,
        window_height: int = 900,
        models_dir: str = "models",
        replays_dir: str = "replays"
    ):
        self.owns_screen = screen is None
        self.models_dir = Path(models_dir)
        self.replays_dir = Path(replays_dir)

        if screen is None:
            pygame.init()
            self.window_width = max(window_width, self.MIN_WIDTH)
            self.window_height = max(window_height, self.MIN_HEIGHT)
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height),
                pygame.RESIZABLE
            )
            pygame.display.set_caption("Snake AI")
        else:
            self.screen = screen
            self.window_width = screen.get_width()
            self.window_height = screen.get_height()

        # Fonts
        self._init_fonts()

        # Create UI components
        self._create_components()

        # Selected mode and options
        self.selected_mode: Optional[str] = None
        self.options: Dict[str, Any] = {}

        self.clock = pygame.time.Clock()

    def _init_fonts(self):
        """Initialize fonts."""
        self.font_title = pygame.font.Font(None, 72)
        self.font_subtitle = pygame.font.Font(None, 32)
        self.font_label = pygame.font.Font(None, 28)

    def _create_components(self):
        """Create all UI components."""
        # Buttons - positions will be set in _recalculate_layout
        self.btn_training = Button(0, 0, 200, 70, "Training")
        self.btn_watch = Button(0, 0, 200, 70, "Watch AI")
        self.btn_human = Button(0, 0, 200, 70, "Play Human")
        self.btn_replays = Button(0, 0, 200, 70, "Replays")
        self.btn_quit = Button(0, 0, 150, 50, "Quit", font_size=24)

        self.buttons = {
            'training': self.btn_training,
            'play': self.btn_watch,
            'human': self.btn_human,
            'replays': self.btn_replays,
            'quit': self.btn_quit
        }

        # Dropdowns
        models = self._get_available_models()
        replays = self._get_available_replays()

        self.model_dropdown = Dropdown(0, 0, 350, models, label="Model:")
        self.replay_dropdown = Dropdown(0, 0, 350, replays, label="Replay:")

        # Toggle
        self.headless_toggle = Toggle(0, 0, "Headless Training (faster)")

        # Calculate initial layout
        self._recalculate_layout()

    def _recalculate_layout(self):
        """Recalculate all component positions based on window size."""
        w, h = self.window_width, self.window_height
        cx = w // 2  # center x

        # Button dimensions
        btn_w, btn_h = 200, 70
        gap = 40

        # Button grid (2x2) - centered
        left_col = cx - btn_w - gap // 2
        right_col = cx + gap // 2

        row1_y = 200
        row2_y = row1_y + btn_h + 30

        self.btn_training.set_position(left_col, row1_y)
        self.btn_training.set_size(btn_w, btn_h)

        self.btn_watch.set_position(right_col, row1_y)
        self.btn_watch.set_size(btn_w, btn_h)

        self.btn_human.set_position(left_col, row2_y)
        self.btn_human.set_size(btn_w, btn_h)

        self.btn_replays.set_position(right_col, row2_y)
        self.btn_replays.set_size(btn_w, btn_h)

        # Dropdowns - centered below buttons
        dropdown_x = cx - 100  # Account for label width
        dropdown_y = row2_y + btn_h + 60

        self.model_dropdown.set_position(dropdown_x, dropdown_y)
        self.replay_dropdown.set_position(dropdown_x, dropdown_y + 50)

        # Toggle - below dropdowns
        toggle_x = cx - 100
        toggle_y = dropdown_y + 120
        self.headless_toggle.set_position(toggle_x, toggle_y)

        # Quit button - bottom center
        quit_y = h - 80
        self.btn_quit.set_position(cx - 75, quit_y)

    def _get_available_models(self) -> List[str]:
        """Get list of available model files for dropdown."""
        if not self.models_dir.exists():
            return ["Start Fresh (no model)"]

        models = ["Start Fresh (no model)"]

        # Add final_model if it exists
        final = self.models_dir / "final_model.pth"
        if final.exists():
            models.append("final_model.pth (Latest)")

        # Add numbered checkpoints sorted by episode number (descending)
        checkpoints = list(self.models_dir.glob("model_ep*.pth"))

        def get_episode_num(p):
            match = re.search(r'ep(\d+)', p.name)
            return int(match.group(1)) if match else 0

        checkpoints.sort(key=get_episode_num, reverse=True)

        for cp in checkpoints[:15]:  # Limit to 15 most recent
            models.append(cp.name)

        return models

    def _get_available_replays(self) -> List[str]:
        """Get list of available replays for dropdown."""
        if not self.replays_dir.exists():
            return ["(no replays found)"]

        replays = ["All Replays (best first)"]

        replay_files = sorted(
            self.replays_dir.glob("replay_*.json"),
            reverse=True
        )

        for rf in replay_files[:10]:
            # Extract score from filename
            match = re.search(r'score(\d+)', rf.name)
            if match:
                score = match.group(1)
                replays.append(f"Score {score} - {rf.name}")
            else:
                replays.append(rf.name)

        return replays if len(replays) > 1 else ["(no replays found)"]

    def _refresh_dropdowns(self):
        """Refresh dropdown items."""
        self.model_dropdown.refresh_items(self._get_available_models())
        self.replay_dropdown.refresh_items(self._get_available_replays())

    def run(self) -> Tuple[str, Dict[str, Any]]:
        """
        Run menu loop until mode selected.

        Returns:
            Tuple of (mode_name, options_dict) where mode_name is one of:
            'training', 'play', 'human', 'replays', 'quit'
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

                # Handle button clicks
                for mode, button in self.buttons.items():
                    if button.handle_event(event):
                        return self._build_result(mode)

                # Handle dropdowns
                self.model_dropdown.handle_event(event)
                self.replay_dropdown.handle_event(event)

                # Handle toggle
                self.headless_toggle.handle_event(event)

            self._draw()
            self.clock.tick(60)

        return ('quit', {})

    def _build_result(self, mode: str) -> Tuple[str, Dict[str, Any]]:
        """Build result tuple with selected options."""
        options = {
            'headless': self.headless_toggle.is_on()
        }

        # Add selected model
        model_selection = self.model_dropdown.get_selected()
        if model_selection and "Start Fresh" not in model_selection:
            # Extract just the filename
            if " - " in model_selection:
                model_selection = model_selection.split(" - ")[-1]
            elif " (" in model_selection:
                model_selection = model_selection.split(" (")[0]
            options['model'] = str(self.models_dir / model_selection)
        else:
            options['model'] = None

        # Add selected replay
        replay_selection = self.replay_dropdown.get_selected()
        if replay_selection == "All Replays (best first)":
            options['replay'] = 'all'
        elif replay_selection and "(no replays" not in replay_selection:
            # Extract filename
            if " - " in replay_selection:
                replay_selection = replay_selection.split(" - ")[-1]
            options['replay'] = str(self.replays_dir / replay_selection)
        else:
            options['replay'] = None

        return (mode, options)

    def _draw(self):
        """Render menu screen."""
        self.screen.fill(BG_COLOR)

        w, h = self.window_width, self.window_height
        cx = w // 2

        # Title
        title = self.font_title.render("SNAKE AI", True, ACCENT_COLOR)
        title_rect = title.get_rect(centerx=cx, y=50)
        self.screen.blit(title, title_rect)

        # Subtitle
        subtitle = self.font_subtitle.render(
            "Train, Play, Watch",
            True, TEXT_COLOR
        )
        subtitle_rect = subtitle.get_rect(centerx=cx, y=120)
        self.screen.blit(subtitle, subtitle_rect)

        # Draw buttons
        for button in self.buttons.values():
            button.draw(self.screen)

        # Section label
        options_label = self.font_label.render("Options", True, TEXT_COLOR)
        options_y = self.model_dropdown.y - 35
        self.screen.blit(options_label, (cx - options_label.get_width() // 2, options_y))

        # Draw dropdowns (in reverse order so expanded ones overlay correctly)
        self.replay_dropdown.draw(self.screen)
        self.model_dropdown.draw(self.screen)

        # Draw toggle
        self.headless_toggle.draw(self.screen)

        # Footer hint
        hint = self.font_label.render(
            "ESC to quit | Window is resizable",
            True, (100, 100, 100)
        )
        hint_rect = hint.get_rect(centerx=cx, y=h - 30)
        self.screen.blit(hint, hint_rect)

        pygame.display.flip()

    def close(self):
        """Clean up resources."""
        if self.owns_screen:
            pygame.quit()
