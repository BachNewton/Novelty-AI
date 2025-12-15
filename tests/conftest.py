"""
Pytest configuration and fixtures for Novelty AI tests.

This module sets up pygame mocking to allow testing UI components
without requiring a display or actual pygame initialization.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest


# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def create_mock_pygame():
    """Create a comprehensive mock of the pygame module."""
    mock_pygame = MagicMock()

    # Basic initialization
    mock_pygame.init.return_value = (6, 0)  # (success, fail) count
    mock_pygame.quit.return_value = None

    # Display
    mock_surface = MagicMock()
    mock_surface.get_width.return_value = 1400
    mock_surface.get_height.return_value = 900
    mock_surface.fill.return_value = None
    mock_surface.blit.return_value = None
    mock_pygame.display.set_mode.return_value = mock_surface
    mock_pygame.display.set_caption.return_value = None
    mock_pygame.display.flip.return_value = None
    mock_pygame.display.update.return_value = None

    # Fonts
    mock_font = MagicMock()
    mock_font.render.return_value = MagicMock()  # Returns a surface
    mock_font.size.return_value = (100, 30)  # (width, height)
    mock_pygame.font.Font.return_value = mock_font
    mock_pygame.font.SysFont.return_value = mock_font
    mock_pygame.font.init.return_value = None

    # Drawing
    mock_pygame.draw.rect.return_value = None
    mock_pygame.draw.line.return_value = None
    mock_pygame.draw.circle.return_value = None
    mock_pygame.draw.aaline.return_value = None

    # Events
    mock_pygame.event.get.return_value = []
    mock_pygame.event.poll.return_value = MagicMock(type=0)

    # Constants
    mock_pygame.QUIT = 256
    mock_pygame.KEYDOWN = 768
    mock_pygame.KEYUP = 769
    mock_pygame.MOUSEBUTTONDOWN = 1025
    mock_pygame.MOUSEBUTTONUP = 1026
    mock_pygame.MOUSEMOTION = 1024
    mock_pygame.VIDEORESIZE = 16
    mock_pygame.RESIZABLE = 16
    mock_pygame.K_ESCAPE = 27
    mock_pygame.K_RETURN = 13
    mock_pygame.K_SPACE = 32
    mock_pygame.K_UP = 273
    mock_pygame.K_DOWN = 274
    mock_pygame.K_LEFT = 276
    mock_pygame.K_RIGHT = 275
    mock_pygame.K_w = 119
    mock_pygame.K_a = 97
    mock_pygame.K_s = 115
    mock_pygame.K_d = 100
    mock_pygame.K_r = 114
    mock_pygame.K_n = 110
    mock_pygame.K_PLUS = 61
    mock_pygame.K_MINUS = 45
    mock_pygame.K_EQUALS = 61

    # Mouse
    mock_pygame.mouse.get_pos.return_value = (0, 0)
    mock_pygame.mouse.get_pressed.return_value = (False, False, False)

    # Time
    mock_clock = MagicMock()
    mock_clock.tick.return_value = 16  # ~60fps
    mock_clock.get_fps.return_value = 60.0
    mock_pygame.time.Clock.return_value = mock_clock
    mock_pygame.time.get_ticks.return_value = 0

    # Rect
    mock_pygame.Rect = MagicMock(side_effect=lambda *args: MagicMock(
        x=args[0] if args else 0,
        y=args[1] if len(args) > 1 else 0,
        width=args[2] if len(args) > 2 else 0,
        height=args[3] if len(args) > 3 else 0,
        collidepoint=MagicMock(return_value=False)
    ))

    # Surface creation
    mock_pygame.Surface.return_value = mock_surface

    return mock_pygame


@pytest.fixture(scope="session", autouse=True)
def mock_pygame_module():
    """
    Session-scoped fixture that mocks pygame before any imports.

    This runs automatically for all tests and ensures pygame
    is mocked before any visualization modules are imported.
    """
    mock_pygame = create_mock_pygame()

    # Store original module if it exists
    original_pygame = sys.modules.get('pygame')

    # Install mock
    sys.modules['pygame'] = mock_pygame

    yield mock_pygame

    # Restore original (or remove mock)
    if original_pygame:
        sys.modules['pygame'] = original_pygame
    else:
        del sys.modules['pygame']


@pytest.fixture
def mock_screen(mock_pygame_module):
    """Provide a mock pygame screen surface."""
    screen = MagicMock()
    screen.get_width.return_value = 1400
    screen.get_height.return_value = 900
    screen.fill.return_value = None
    screen.blit.return_value = None
    return screen


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create a temporary models directory with sample files."""
    models_dir = tmp_path / "models" / "snake"
    models_dir.mkdir(parents=True)

    # Create sample model files
    (models_dir / "final_model.pth").touch()
    (models_dir / "model_ep100.pth").touch()
    (models_dir / "model_ep500.pth").touch()
    (models_dir / "model_ep1000.pth").touch()

    return tmp_path / "models"


@pytest.fixture
def temp_replays_dir(tmp_path):
    """Create a temporary replays directory with sample replay files."""
    import json

    replays_dir = tmp_path / "replays" / "snake"
    replays_dir.mkdir(parents=True)

    # Create valid replay data with frames that renderer can use
    # Format matches SnakeGame.get_state() which uses Point.to_dict()
    sample_frame = {
        "snake": [{"x": 5, "y": 5}, {"x": 4, "y": 5}, {"x": 3, "y": 5}],
        "food": {"x": 10, "y": 10},
        "direction": 1,  # int representation of Direction
        "score": 0,
    }
    replay_data = {
        "frames": [sample_frame],
        "score": 50,
        "episode": 100,
        "timestamp": "20241201_120000",
        "duration_frames": 1,
    }

    (replays_dir / "replay_ep100_score50_20241201_120000.json").write_text(
        json.dumps(replay_data)
    )

    replay_data2 = {
        "frames": [sample_frame],
        "score": 75,
        "episode": 200,
        "timestamp": "20241202_130000",
        "duration_frames": 1,
    }
    (replays_dir / "replay_ep200_score75_20241202_130000.json").write_text(
        json.dumps(replay_data2)
    )

    return tmp_path / "replays"


@pytest.fixture
def sample_config():
    """Provide sample configuration for tests."""
    return {
        'training': {
            'episodes': 1000,
            'batch_size': 64,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.02,
            'epsilon_decay': 0.998,
            'target_update': 10,
            'memory_size': 100000,
            'checkpoint_interval': 100,
            'save_best_replays': 10,
        },
        'game': {
            'grid_width': 20,
            'grid_height': 20,
        },
        'window': {
            'width': 1400,
            'height': 900,
        }
    }
