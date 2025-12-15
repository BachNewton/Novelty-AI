"""
Game Menu - Per-game menu for training, playing, and watching AI.

Provides access to all game-specific features.
"""

import pygame
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import torch

from .ui_components import (
    Button, Dropdown, BG_COLOR, TEXT_COLOR, ACCENT_COLOR, ACCENT_ORANGE
)
from ..games.registry import GameRegistry
from ..core.game_interface import GameMetadata


class GameMenu:
    """
    Per-game menu screen providing access to all game modes.

    Layout:
    +------------------------------------------+
    |       NOVELTY AI > [Game Name]           |
    |                                          |
    |    [Train AI]      [Watch AI]            |
    |    [Play Human]    [Replays]             |
    |                                          |
    |    Model: [dropdown]                     |
    |    Device: [dropdown]                    |
    |                                          |
    |         [← Back to Hub]                  |
    +------------------------------------------+
    """

    MIN_WIDTH = 800
    MIN_HEIGHT = 600

    def __init__(
        self,
        game_id: str,
        screen: Optional[pygame.Surface] = None,
        window_width: int = 1400,
        window_height: int = 900,
        models_dir: str = "models",
        replays_dir: str = "replays"
    ):
        """
        Initialize the game menu.

        Args:
            game_id: ID of the game this menu is for
            screen: Optional existing pygame surface
            window_width: Window width
            window_height: Window height
            models_dir: Base directory for models
            replays_dir: Base directory for replays
        """
        self.game_id = game_id
        self.owns_screen = screen is None

        # Get game metadata
        game_data = GameRegistry.get_game(game_id)
        if game_data:
            self.metadata: GameMetadata = game_data['metadata']
        else:
            raise ValueError(f"Unknown game: {game_id}")

        # Per-game directories
        self.models_dir = Path(models_dir) / game_id
        self.replays_dir = Path(replays_dir) / game_id

        if screen is None:
            pygame.init()
            self.window_width = max(window_width, self.MIN_WIDTH)
            self.window_height = max(window_height, self.MIN_HEIGHT)
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height),
                pygame.RESIZABLE
            )
            pygame.display.set_caption(f"Novelty AI - {self.metadata.name}")
        else:
            self.screen = screen
            self.window_width = screen.get_width()
            self.window_height = screen.get_height()

        # Fonts
        self._init_fonts()

        # Create UI components
        self._create_components()

        self.clock = pygame.time.Clock()

    def _init_fonts(self):
        """Initialize fonts."""
        self.font_title = pygame.font.Font(None, 56)
        self.font_subtitle = pygame.font.Font(None, 32)
        self.font_label = pygame.font.Font(None, 28)

    def _create_components(self):
        """Create all UI components."""
        # Buttons
        self.btn_training = Button(0, 0, 200, 70, "Train AI")
        self.btn_watch = Button(0, 0, 200, 70, "Watch AI")
        self.btn_human = Button(0, 0, 200, 70, "Play Human")
        self.btn_replays = Button(0, 0, 200, 70, "Replays")
        self.btn_back = Button(0, 0, 200, 50, "← Back to Hub", font_size=24)

        self.buttons = {
            'training': self.btn_training,
            'play': self.btn_watch,
            'human': self.btn_human,
            'replays': self.btn_replays,
            'back': self.btn_back
        }

        # Disable human play if not supported
        if not self.metadata.supports_human:
            self.btn_human.text = "Play Human (N/A)"

        # Dropdowns
        models = self._get_available_models()
        replays = self._get_available_replays()
        devices = self._get_available_devices()

        self.model_dropdown = Dropdown(0, 0, 350, models, label="Model:")
        self.replay_dropdown = Dropdown(0, 0, 350, replays, label="Replay:")
        self.device_dropdown = Dropdown(0, 0, 350, devices, label="Device:")

        # Calculate initial layout
        self._recalculate_layout()

    def _recalculate_layout(self):
        """Recalculate all component positions based on window size."""
        w, h = self.window_width, self.window_height
        cx = w // 2

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
        self.device_dropdown.set_position(dropdown_x, dropdown_y + 100)

        # Back button - bottom center
        self.btn_back.set_position(cx - 100, h - 100)

    def _get_available_models(self) -> List[str]:
        """Get list of available model files."""
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
        """Get list of available replays."""
        if not self.replays_dir.exists():
            return ["(no replays found)"]

        replays = ["All Replays (best first)"]

        replay_files = sorted(
            self.replays_dir.glob("replay_*.json"),
            reverse=True
        )

        for rf in replay_files[:10]:
            match = re.search(r'score(\d+)', rf.name)
            if match:
                score = match.group(1)
                replays.append(f"Score {score} - {rf.name}")
            else:
                replays.append(rf.name)

        return replays if len(replays) > 1 else ["(no replays found)"]

    def _get_available_devices(self) -> List[str]:
        """Get list of available compute devices."""
        devices = []

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            devices.append(f"GPU ({gpu_name})")

        devices.append("CPU")

        return devices

    def _refresh_dropdowns(self):
        """Refresh dropdown items."""
        self.model_dropdown.refresh_items(self._get_available_models())
        self.replay_dropdown.refresh_items(self._get_available_replays())

    def run(self) -> Tuple[str, Dict[str, Any]]:
        """
        Run menu loop until mode selected.

        Returns:
            Tuple of (mode_name, options_dict) where mode_name is one of:
            'training', 'play', 'human', 'replays', 'back', 'quit'
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
                        return ('back', {})

                # Handle button clicks
                for mode, button in self.buttons.items():
                    if button.handle_event(event):
                        if mode == 'human' and not self.metadata.supports_human:
                            continue  # Ignore if human play not supported
                        return self._build_result(mode)

                # Handle dropdowns
                if not self.model_dropdown.handle_event(event):
                    if not self.replay_dropdown.handle_event(event):
                        self.device_dropdown.handle_event(event)

            self._draw()
            self.clock.tick(60)

        return ('quit', {})

    def _build_result(self, mode: str) -> Tuple[str, Dict[str, Any]]:
        """Build result tuple with selected options."""
        options = {
            'game_id': self.game_id,
            'headless': True,  # Default to headless, can toggle with H key
        }

        # Add selected model
        model_selection = self.model_dropdown.get_selected()
        if model_selection and "Start Fresh" not in model_selection:
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
            if " - " in replay_selection:
                replay_selection = replay_selection.split(" - ")[-1]
            options['replay'] = str(self.replays_dir / replay_selection)
        else:
            options['replay'] = None

        # Add selected device
        device_selection = self.device_dropdown.get_selected()
        if device_selection and device_selection.startswith("GPU"):
            options['device'] = 'cuda'
        else:
            options['device'] = 'cpu'

        return (mode, options)

    def _draw(self):
        """Render menu screen."""
        self.screen.fill(BG_COLOR)

        w, h = self.window_width, self.window_height
        cx = w // 2

        # Title with breadcrumb
        title_text = f"NOVELTY AI"
        title = self.font_title.render(title_text, True, ACCENT_COLOR)
        title_rect = title.get_rect(centerx=cx, y=40)
        self.screen.blit(title, title_rect)

        # Game name as subtitle
        game_title = self.font_subtitle.render(
            f"> {self.metadata.name}",
            True, TEXT_COLOR
        )
        game_rect = game_title.get_rect(centerx=cx, y=100)
        self.screen.blit(game_title, game_rect)

        # Draw buttons
        for button in self.buttons.values():
            button.draw(self.screen)

        # Section label
        options_label = self.font_label.render("Options", True, TEXT_COLOR)
        options_y = self.model_dropdown.y - 35
        self.screen.blit(options_label, (cx - options_label.get_width() // 2, options_y))

        # Draw dropdowns (in reverse order so expanded ones overlay correctly)
        self.device_dropdown.draw(self.screen)
        self.replay_dropdown.draw(self.screen)
        self.model_dropdown.draw(self.screen)

        # Footer hint
        hint = self.font_label.render(
            "ESC to go back | Window is resizable",
            True, (100, 105, 115)
        )
        hint_rect = hint.get_rect(centerx=cx, y=h - 30)
        self.screen.blit(hint, hint_rect)

        pygame.display.flip()

    def close(self):
        """Clean up resources."""
        if self.owns_screen:
            pygame.quit()
