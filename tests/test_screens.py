"""
Tests for screen initialization (GameHub, GameMenu, TrainingDashboard).

These tests verify that UI screens can be initialized without crashing.
They use a mocked pygame to avoid needing a display.
"""

import pytest


class TestGameHub:
    """Tests for the GameHub class."""

    def test_game_hub_init_with_screen(self, mock_screen, mock_pygame_module):
        """Test GameHub initialization with provided screen."""
        from src.visualization.game_hub import GameHub

        hub = GameHub(screen=mock_screen)

        assert hub.screen is mock_screen
        assert hub.owns_screen is False
        assert hub.window_width == 1400
        assert hub.window_height == 900

    def test_game_hub_init_without_screen(self, mock_pygame_module):
        """Test GameHub initialization without screen (creates own)."""
        from src.visualization.game_hub import GameHub

        hub = GameHub()

        assert hub.owns_screen is True
        assert hub.window_width >= GameHub.MIN_WIDTH
        assert hub.window_height >= GameHub.MIN_HEIGHT

    def test_game_hub_creates_game_cards(self, mock_screen, mock_pygame_module):
        """Test that GameHub creates game cards from registry."""
        from src.visualization.game_hub import GameHub

        hub = GameHub(screen=mock_screen)

        # Should have at least one game card (Snake)
        assert hasattr(hub, 'game_cards')
        assert isinstance(hub.game_cards, list)

    def test_game_hub_creates_buttons(self, mock_screen, mock_pygame_module):
        """Test that GameHub creates settings and quit buttons."""
        from src.visualization.game_hub import GameHub

        hub = GameHub(screen=mock_screen)

        assert hasattr(hub, 'btn_settings')
        assert hasattr(hub, 'btn_quit')

    def test_game_hub_creates_fonts(self, mock_screen, mock_pygame_module):
        """Test that GameHub initializes fonts."""
        from src.visualization.game_hub import GameHub

        hub = GameHub(screen=mock_screen)

        assert hasattr(hub, 'font_title')
        assert hasattr(hub, 'font_subtitle')


class TestGameMenu:
    """Tests for the GameMenu class."""

    def test_game_menu_init_snake(self, mock_screen, mock_pygame_module):
        """Test GameMenu initialization for Snake game."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir='models',
            replays_dir='replays'
        )

        assert menu.game_id == 'snake'
        assert menu.screen is mock_screen
        assert menu.owns_screen is False

    def test_game_menu_creates_buttons(self, mock_screen, mock_pygame_module):
        """Test that GameMenu creates all required buttons."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir='models',
            replays_dir='replays'
        )

        assert hasattr(menu, 'btn_training')
        assert hasattr(menu, 'btn_watch')
        assert hasattr(menu, 'btn_human')
        assert hasattr(menu, 'btn_replays')
        assert hasattr(menu, 'btn_back')

    def test_game_menu_creates_dropdowns(self, mock_screen, mock_pygame_module):
        """Test that GameMenu creates dropdowns."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir='models',
            replays_dir='replays'
        )

        # Dropdowns are named model_dropdown, replay_dropdown, device_dropdown
        assert hasattr(menu, 'model_dropdown')
        assert hasattr(menu, 'replay_dropdown')
        assert hasattr(menu, 'device_dropdown')

    def test_game_menu_sets_paths(self, mock_screen, mock_pygame_module):
        """Test that GameMenu sets correct paths for models and replays."""
        from src.visualization.game_menu import GameMenu
        from pathlib import Path

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir='models',
            replays_dir='replays'
        )

        # Should be models/snake and replays/snake
        assert menu.models_dir == Path('models') / 'snake'
        assert menu.replays_dir == Path('replays') / 'snake'

    def test_game_menu_invalid_game_raises(self, mock_screen, mock_pygame_module):
        """Test that GameMenu raises for invalid game ID."""
        from src.visualization.game_menu import GameMenu

        with pytest.raises(ValueError, match="Unknown game"):
            GameMenu(
                game_id='nonexistent_game',
                screen=mock_screen,
                models_dir='models',
                replays_dir='replays'
            )

    def test_game_menu_creates_fonts(self, mock_screen, mock_pygame_module):
        """Test that GameMenu initializes fonts."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir='models',
            replays_dir='replays'
        )

        assert hasattr(menu, 'font_title')
        assert hasattr(menu, 'font_subtitle')
        assert hasattr(menu, 'font_label')


class TestTrainingDashboard:
    """Tests for the TrainingDashboard class."""

    def test_dashboard_import(self, mock_pygame_module):
        """Test TrainingDashboard can be imported."""
        from src.visualization.dashboard import TrainingDashboard

        assert TrainingDashboard is not None

    def test_dashboard_class_has_min_dimensions(self, mock_pygame_module):
        """Test TrainingDashboard has minimum dimension constants."""
        from src.visualization.dashboard import TrainingDashboard

        assert hasattr(TrainingDashboard, 'MIN_WIDTH')
        assert hasattr(TrainingDashboard, 'MIN_HEIGHT')
        assert TrainingDashboard.MIN_WIDTH >= 800
        assert TrainingDashboard.MIN_HEIGHT >= 600

    def test_dashboard_init_headless_mode(self, mock_screen, mock_pygame_module):
        """Test TrainingDashboard in headless mode (no game display)."""
        from src.visualization.dashboard import TrainingDashboard

        # Note: Full initialization requires renderer which may not work in tests
        # Testing that the class exists and has expected structure
        dashboard = TrainingDashboard(
            screen=mock_screen,
            show_game=False
        )

        assert dashboard.show_game is False


class TestGameCard:
    """Tests for the GameCard class."""

    def test_game_card_init(self, mock_pygame_module):
        """Test GameCard initialization."""
        from src.visualization.game_hub import GameCard
        from src.core.game_interface import GameMetadata

        metadata = GameMetadata(
            id='test',
            name='Test Game',
            description='A test game',
            supports_human=True,
            recommended_algorithms=['dqn']
        )

        card = GameCard(metadata, x=100, y=200)

        assert card.metadata is metadata
        assert card.rect.x == 100
        assert card.rect.y == 200
        assert card.available is True
        assert card.hovered is False

    def test_game_card_unavailable(self, mock_pygame_module):
        """Test GameCard for unavailable (coming soon) game."""
        from src.visualization.game_hub import GameCard
        from src.core.game_interface import GameMetadata

        metadata = GameMetadata(
            id='tetris',
            name='Tetris',
            description='Coming soon',
            supports_human=True,
            recommended_algorithms=[]
        )

        card = GameCard(metadata, x=0, y=0, available=False)

        assert card.available is False

    def test_game_card_set_position(self, mock_pygame_module):
        """Test GameCard position update."""
        from src.visualization.game_hub import GameCard
        from src.core.game_interface import GameMetadata

        metadata = GameMetadata(
            id='test',
            name='Test',
            description='Test',
            supports_human=True,
            recommended_algorithms=[]
        )

        card = GameCard(metadata, x=0, y=0)
        card.set_position(300, 400)

        assert card.rect.x == 300
        assert card.rect.y == 400
