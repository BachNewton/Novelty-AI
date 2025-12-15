"""
Integration smoke tests for UI entry points.

These tests verify the main UI flows can be initialized without crashing.
"""

import pytest
from pathlib import Path


class TestUIImports:
    """Tests that UI modules can be imported without errors."""

    def test_import_ui_components(self, mock_pygame_module):
        """Test importing ui_components module."""
        from src.visualization import ui_components
        assert ui_components is not None

    def test_import_game_hub(self, mock_pygame_module):
        """Test importing game_hub module."""
        from src.visualization import game_hub
        assert game_hub is not None

    def test_import_game_menu(self, mock_pygame_module):
        """Test importing game_menu module."""
        from src.visualization import game_menu
        assert game_menu is not None

    def test_import_dashboard(self, mock_pygame_module):
        """Test importing dashboard module."""
        from src.visualization import dashboard
        assert dashboard is not None


class TestModelDropdownPopulation:
    """Tests for model dropdown population logic."""

    def test_get_available_models_empty_dir(self, mock_screen, mock_pygame_module, tmp_path):
        """Test model dropdown with empty directory."""
        from src.visualization.game_menu import GameMenu

        # Create empty models directory
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir=str(models_dir),
            replays_dir='replays'
        )

        models = menu._get_available_models()

        # Should have at least "Start Fresh" option
        assert len(models) >= 1
        assert any('Fresh' in m or 'none' in m.lower() for m in models)

    def test_get_available_models_with_files(self, mock_screen, mock_pygame_module, temp_models_dir):
        """Test model dropdown with model files present."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir=str(temp_models_dir),
            replays_dir='replays'
        )

        models = menu._get_available_models()

        # Should include the sample model files
        assert len(models) >= 2  # At least "Start Fresh" + some models
        assert any('final_model' in m for m in models)

    def test_get_available_models_sorts_by_episode(self, mock_screen, mock_pygame_module, temp_models_dir):
        """Test that models are sorted by episode number."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir=str(temp_models_dir),
            replays_dir='replays'
        )

        models = menu._get_available_models()

        # Find checkpoint models (not final_model or Start Fresh)
        checkpoints = [m for m in models if 'ep' in m and 'final' not in m.lower()]

        if len(checkpoints) >= 2:
            # Extract episode numbers
            def get_ep(name):
                import re
                match = re.search(r'ep(\d+)', name)
                return int(match.group(1)) if match else 0

            ep_nums = [get_ep(c) for c in checkpoints]
            # Should be in descending order (newest first)
            assert ep_nums == sorted(ep_nums, reverse=True)


class TestReplayDropdownPopulation:
    """Tests for replay dropdown population logic."""

    def test_get_available_replays_empty_dir(self, mock_screen, mock_pygame_module, tmp_path):
        """Test replay dropdown with empty directory."""
        from src.visualization.game_menu import GameMenu

        # Create empty replays directory
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir()

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir='models',
            replays_dir=str(replays_dir)
        )

        replays = menu._get_available_replays()

        # Should have "(none)" or similar when empty
        assert len(replays) >= 1

    def test_get_available_replays_with_files(self, mock_screen, mock_pygame_module, temp_replays_dir):
        """Test replay dropdown with replay files present."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir='models',
            replays_dir=str(temp_replays_dir)
        )

        replays = menu._get_available_replays()

        # Should find the sample replay files
        assert len(replays) >= 1


class TestDeviceDropdown:
    """Tests for device (GPU/CPU) dropdown."""

    def test_device_dropdown_has_options(self, mock_screen, mock_pygame_module):
        """Test that device dropdown has at least one option."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir='models',
            replays_dir='replays'
        )

        # Device dropdown should exist (named device_dropdown)
        assert menu.device_dropdown is not None

        # Should have at least CPU option
        items = menu.device_dropdown.items
        assert len(items) >= 1


class TestScreenDimensions:
    """Tests for screen dimension handling."""

    def test_game_hub_enforces_min_size(self, mock_pygame_module):
        """Test that GameHub enforces minimum dimensions."""
        from src.visualization.game_hub import GameHub

        # Try to create with dimensions below minimum
        hub = GameHub(window_width=400, window_height=300)

        assert hub.window_width >= GameHub.MIN_WIDTH
        assert hub.window_height >= GameHub.MIN_HEIGHT

    def test_game_menu_enforces_min_size(self, mock_pygame_module):
        """Test that GameMenu enforces minimum dimensions."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            window_width=400,
            window_height=300,
            models_dir='models',
            replays_dir='replays'
        )

        assert menu.window_width >= GameMenu.MIN_WIDTH
        assert menu.window_height >= GameMenu.MIN_HEIGHT

    def test_dashboard_has_min_size_constants(self, mock_pygame_module):
        """Test that TrainingDashboard has minimum dimension constants."""
        from src.visualization.dashboard import TrainingDashboard

        # Just verify the class has min size constants defined
        assert hasattr(TrainingDashboard, 'MIN_WIDTH')
        assert hasattr(TrainingDashboard, 'MIN_HEIGHT')
        assert TrainingDashboard.MIN_WIDTH >= 800
        assert TrainingDashboard.MIN_HEIGHT >= 600


class TestWatchAIMode:
    """Tests for Watch AI mode initialization."""

    def test_renderer_set_render_area(self, mock_pygame_module):
        """Test that renderer's set_render_area works."""
        from src.games.registry import GameRegistry

        renderer = GameRegistry.create_renderer('snake')

        # Should not raise
        renderer.set_render_area(10, 10, 400, 400)

    def test_renderer_set_cell_size(self, mock_pygame_module):
        """Test that renderer's set_cell_size works."""
        from src.games.registry import GameRegistry

        renderer = GameRegistry.create_renderer('snake')

        # Should not raise
        renderer.set_cell_size(25)
        assert renderer.get_cell_size() == 25

    def test_ui_renderer_setup_code(self, mock_pygame_module):
        """
        Test that the renderer setup pattern used in ui.py works.

        This test replicates the exact code pattern from _run_play, _run_human,
        and _run_replays to catch method mismatches early.
        """
        from src.games.registry import GameRegistry

        # This is the exact pattern used in ui.py
        renderer = GameRegistry.create_renderer('snake')

        # Simulate the setup code from ui.py
        grid_w, grid_h = 20, 20
        window_width, window_height = 1400, 900

        cell_size = min(
            (window_width - 40) // grid_w,
            (window_height - 100) // grid_h
        )
        game_width = cell_size * grid_w
        game_height = cell_size * grid_h
        offset_x = (window_width - game_width) // 2
        offset_y = (window_height - game_height - 60) // 2

        # These are the actual calls made in ui.py - if any fail, ui.py will crash
        renderer.set_cell_size(cell_size)
        renderer.set_render_area(offset_x, offset_y, game_width, game_height)


class TestPathConstruction:
    """Tests for path construction to catch double-path bugs."""

    def test_game_menu_correct_model_path(self, mock_screen, mock_pygame_module):
        """Test that GameMenu constructs correct model path."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir='models',
            replays_dir='replays'
        )

        # Path should be models/snake, not models/snake/snake
        expected = Path('models') / 'snake'
        assert menu.models_dir == expected
        assert 'snake/snake' not in str(menu.models_dir)

    def test_game_menu_correct_replay_path(self, mock_screen, mock_pygame_module):
        """Test that GameMenu constructs correct replay path."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir='models',
            replays_dir='replays'
        )

        # Path should be replays/snake, not replays/snake/snake
        expected = Path('replays') / 'snake'
        assert menu.replays_dir == expected
        assert 'snake/snake' not in str(menu.replays_dir)
