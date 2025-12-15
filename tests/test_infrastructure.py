"""
Tests for infrastructure components (GameRegistry, config loading).

These tests verify that core infrastructure works correctly.
"""

import pytest
from pathlib import Path


class TestGameRegistry:
    """Tests for the GameRegistry class."""

    def test_registry_list_games(self, mock_pygame_module):
        """Test that registry lists available games."""
        from src.games.registry import GameRegistry

        games = GameRegistry.list_games()

        assert isinstance(games, list)
        # Snake should be registered
        game_ids = [g.id for g in games]
        assert 'snake' in game_ids

    def test_registry_list_all_games(self, mock_pygame_module):
        """Test that registry lists all games including placeholders."""
        from src.games.registry import GameRegistry

        all_games = GameRegistry.list_all_games(include_placeholders=True)

        assert isinstance(all_games, list)
        assert len(all_games) >= 1

    def test_registry_get_game(self, mock_pygame_module):
        """Test getting a specific game by ID."""
        from src.games.registry import GameRegistry

        game_data = GameRegistry.get_game('snake')

        assert game_data is not None
        assert 'game_class' in game_data
        assert 'env_class' in game_data
        assert 'renderer_class' in game_data
        assert 'metadata' in game_data

    def test_registry_get_game_unknown(self, mock_pygame_module):
        """Test getting unknown game returns None."""
        from src.games.registry import GameRegistry

        game_data = GameRegistry.get_game('nonexistent')

        assert game_data is None

    def test_registry_is_available(self, mock_pygame_module):
        """Test checking game availability."""
        from src.games.registry import GameRegistry

        assert GameRegistry.is_available('snake') is True
        assert GameRegistry.is_available('nonexistent') is False

    def test_registry_get_metadata(self, mock_pygame_module):
        """Test getting game metadata."""
        from src.games.registry import GameRegistry

        metadata = GameRegistry.get_metadata('snake')

        assert metadata is not None
        assert metadata.id == 'snake'
        assert metadata.name == 'Snake'
        assert hasattr(metadata, 'description')
        assert hasattr(metadata, 'supports_human')
        assert hasattr(metadata, 'recommended_algorithms')

    def test_registry_get_metadata_unknown(self, mock_pygame_module):
        """Test getting metadata for unknown game."""
        from src.games.registry import GameRegistry

        metadata = GameRegistry.get_metadata('nonexistent')

        assert metadata is None

    def test_registry_create_env(self, mock_pygame_module):
        """Test creating environment from registry."""
        from src.games.registry import GameRegistry

        env = GameRegistry.create_env('snake')

        assert env is not None
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'state_size')
        assert hasattr(env, 'action_size')

    def test_registry_create_env_unknown(self, mock_pygame_module):
        """Test creating env for unknown game raises."""
        from src.games.registry import GameRegistry

        with pytest.raises(ValueError, match="Unknown game"):
            GameRegistry.create_env('nonexistent')


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_config(self, mock_pygame_module):
        """Test loading main config file."""
        from src.utils.config_loader import load_config

        config = load_config()

        assert config is not None
        # Config is a dataclass, not a dict
        assert hasattr(config, 'training')
        assert hasattr(config, 'game')

    def test_load_game_config(self, mock_pygame_module):
        """Test loading game-specific config."""
        from src.utils.config_loader import load_game_config

        config = load_game_config('snake')

        assert config is not None
        # Config is a dataclass with training and game attributes
        assert hasattr(config, 'training')
        assert hasattr(config, 'game')

    def test_config_has_training_settings(self, mock_pygame_module):
        """Test that config includes training parameters."""
        from src.utils.config_loader import load_game_config

        config = load_game_config('snake')

        # Config.training is a TrainingConfig dataclass
        assert hasattr(config.training, 'episodes')
        assert hasattr(config.training, 'batch_size')
        assert config.training.episodes > 0


class TestGameMetadata:
    """Tests for GameMetadata dataclass."""

    def test_metadata_fields(self, mock_pygame_module):
        """Test that GameMetadata has required fields."""
        from src.core.game_interface import GameMetadata

        metadata = GameMetadata(
            id='test',
            name='Test Game',
            description='A test game for testing',
            supports_human=True,
            recommended_algorithms=['dqn', 'ppo']
        )

        assert metadata.id == 'test'
        assert metadata.name == 'Test Game'
        assert metadata.description == 'A test game for testing'
        assert metadata.supports_human is True
        assert metadata.recommended_algorithms == ['dqn', 'ppo']


class TestSnakeGame:
    """Tests for Snake game components."""

    def test_snake_game_class_exists(self, mock_pygame_module):
        """Test that SnakeGame class can be imported."""
        from src.games.snake.game import SnakeGame

        assert SnakeGame is not None

    def test_snake_env_class_exists(self, mock_pygame_module):
        """Test that SnakeEnv class can be imported."""
        from src.games.snake.env import SnakeEnv

        assert SnakeEnv is not None

    def test_snake_renderer_class_exists(self, mock_pygame_module):
        """Test that SnakeRenderer class can be imported."""
        from src.games.snake.renderer import SnakeRenderer

        assert SnakeRenderer is not None

    def test_snake_env_init(self, mock_pygame_module):
        """Test SnakeEnv initialization."""
        from src.games.snake.env import SnakeEnv

        env = SnakeEnv()

        assert env.state_size > 0
        assert env.action_size > 0

    def test_snake_env_reset(self, mock_pygame_module):
        """Test SnakeEnv reset."""
        from src.games.snake.env import SnakeEnv
        import numpy as np

        env = SnakeEnv()
        state = env.reset()

        assert state is not None
        assert isinstance(state, np.ndarray)
        assert len(state) == env.state_size

    def test_snake_env_step(self, mock_pygame_module):
        """Test SnakeEnv step."""
        from src.games.snake.env import SnakeEnv

        env = SnakeEnv()
        env.reset()

        # Take a step with action 0 (usually forward or up)
        state, reward, done, info = env.step(0)

        assert state is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_training_config_from_loaded_config(self, mock_pygame_module):
        """Test creating TrainingConfig from loaded config object."""
        from src.training.trainer import TrainingConfig
        from src.utils.config_loader import load_game_config

        loaded_config = load_game_config('snake')
        config = TrainingConfig.from_config(loaded_config)

        assert config is not None
        assert config.episodes > 0
        assert config.batch_size > 0

    def test_training_config_defaults(self, mock_pygame_module):
        """Test TrainingConfig with default values."""
        from src.training.trainer import TrainingConfig

        # Create with defaults
        config = TrainingConfig()

        # Should have default values
        assert config.episodes > 0
        assert config.batch_size > 0
        assert config.gamma > 0
        assert config.learning_rate > 0
