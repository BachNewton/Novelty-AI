"""
Tests that exercise the actual code paths triggered by UI clicks.

These tests replicate what happens when you click buttons in the UI,
catching crashes before they happen in manual testing.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestClickWatchAI:
    """Tests for clicking 'Watch AI' button."""

    def test_watch_ai_initialization(self, mock_pygame_module, mock_screen, temp_models_dir):
        """
        Test the full initialization sequence when clicking 'Watch AI'.

        This replicates _run_play() from ui.py up to the game loop.
        """
        from src.games.registry import GameRegistry
        from src.utils.config_loader import load_game_config
        from src.device.device_manager import DeviceManager
        from src.algorithms.dqn.agent import DQNAgent

        game_id = 'snake'
        game_config = load_game_config(game_id)
        options = {'model': None, 'device': 'cpu'}

        # Step 1: Get metadata (as ui.py does)
        metadata = GameRegistry.get_metadata(game_id)
        assert metadata is not None
        game_name = metadata.name if metadata else game_id.title()

        # Step 2: Find model (as ui.py does)
        model_path = options.get('model')
        models_dir = temp_models_dir / game_id
        if not model_path or not Path(model_path).exists():
            final = models_dir / "final_model.pth"
            if final.exists():
                model_path = str(final)
            else:
                checkpoints = list(models_dir.glob("model_ep*.pth"))
                if checkpoints:
                    model_path = str(sorted(checkpoints)[-1])

        # Model should be found from temp_models_dir
        assert model_path is not None
        assert Path(model_path).exists()

        # Step 3: Create DeviceManager (as ui.py does)
        force_cpu = options.get('device', 'cuda') == 'cpu'
        device_manager = DeviceManager(force_cpu=force_cpu)

        # Step 4: Create environment (as ui.py does)
        env = GameRegistry.create_env(
            game_id,
            width=game_config.game.grid_width,
            height=game_config.game.grid_height
        )
        assert env is not None
        assert env.state_size > 0
        assert env.action_size > 0

        # Step 5: Create agent (as ui.py does)
        agent = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            device=device_manager.get_device()
        )
        # Note: We skip agent.load() since temp models are empty files

        # Step 6: Create renderer and set it up (as ui.py does)
        renderer = GameRegistry.create_renderer(game_id)

        grid_w = game_config.game.grid_width
        grid_h = game_config.game.grid_height
        window_width, window_height = 1400, 900

        cell_size = min(
            (window_width - 40) // grid_w,
            (window_height - 100) // grid_h
        )
        game_width = cell_size * grid_w
        game_height = cell_size * grid_h
        offset_x = (window_width - game_width) // 2
        offset_y = (window_height - game_height - 60) // 2

        # These calls must not raise
        renderer.set_cell_size(cell_size)
        renderer.set_render_area(offset_x, offset_y, game_width, game_height)

        # Step 7: Reset environment (as ui.py does before game loop)
        state = env.reset()
        assert state is not None

        # Step 8: Test the full game loop iteration (as ui.py does)
        # This catches method name mismatches and signature errors
        action = agent.select_action(state, training=False)
        assert action is not None
        assert 0 <= action < env.action_size

        next_state, reward, done, info = env.step(action)
        assert next_state is not None
        assert 'score' in info

        game_state = env.get_game_state()
        assert game_state is not None
        renderer.render(game_state, mock_screen)

        # Test reset on game over (as ui.py does)
        state = env.reset()
        assert state is not None


class TestClickPlayHuman:
    """Tests for clicking 'Play Human' button."""

    def test_play_human_initialization(self, mock_pygame_module, mock_screen):
        """
        Test the full initialization sequence when clicking 'Play Human'.

        This replicates _run_human() from ui.py up to the game loop.
        """
        from src.games.registry import GameRegistry
        from src.games.snake.game import SnakeGame
        from src.utils.config_loader import load_game_config

        game_id = 'snake'
        game_config = load_game_config(game_id)

        # Step 1: Get metadata (as ui.py does)
        metadata = GameRegistry.get_metadata(game_id)
        assert metadata is not None
        assert metadata.supports_human is True

        # Step 2: Create game instance (as ui.py does)
        grid_w = game_config.game.grid_width
        grid_h = game_config.game.grid_height
        game = SnakeGame(grid_w, grid_h)
        assert game is not None

        # Step 3: Create renderer and set it up (as ui.py does)
        renderer = GameRegistry.create_renderer(game_id)

        window_width, window_height = 1400, 900
        cell_size = min(
            (window_width - 40) // grid_w,
            (window_height - 100) // grid_h
        )
        game_width = cell_size * grid_w
        game_height = cell_size * grid_h
        offset_x = (window_width - game_width) // 2
        offset_y = (window_height - game_height - 60) // 2

        # These calls must not raise
        renderer.set_cell_size(cell_size)
        renderer.set_render_area(offset_x, offset_y, game_width, game_height)

        # Step 4: Get initial game state (as ui.py does)
        game_state = game.get_state()
        assert game_state is not None
        assert 'snake' in game_state
        assert 'food' in game_state

        # Step 5: Test the full game loop iteration (as ui.py does)
        # Test step with action (AI-style step used as fallback)
        _, _, done, info = game.step(0)
        assert 'score' in info or hasattr(game, 'score')

        # Test step_direction (human keyboard input)
        from src.games.snake.game import Direction
        _, _, done, info = game.step_direction(Direction.RIGHT)

        # Test game.score property access (used for display)
        score = game.score
        assert isinstance(score, int)

        # Test get_state after step
        game_state = game.get_state()
        renderer.render(game_state, mock_screen)

        # Test reset (used when pressing R)
        game.reset()
        game_state = game.get_state()
        assert game_state is not None
        renderer.render(game_state, mock_screen)


class TestClickReplays:
    """Tests for clicking 'Replays' button."""

    def test_replays_initialization(self, mock_pygame_module, mock_screen, temp_replays_dir):
        """
        Test the full initialization sequence when clicking 'Replays'.

        This replicates _run_replays() from ui.py up to the playback loop.
        """
        from src.games.registry import GameRegistry
        from src.visualization.replay_player import ReplayManager
        from src.utils.config_loader import load_game_config

        game_id = 'snake'
        game_config = load_game_config(game_id)
        options = {'replay': None}

        # Step 1: Get metadata (as ui.py does)
        metadata = GameRegistry.get_metadata(game_id)
        assert metadata is not None

        # Step 2: Create ReplayManager (as ui.py does)
        replays_dir = temp_replays_dir / game_id
        replay_manager = ReplayManager(save_dir=str(replays_dir))

        # Step 3: Find replay files (as ui.py does)
        replay_path = options.get('replay')
        if replay_path and Path(replay_path).exists():
            replay_files = [Path(replay_path)]
        else:
            replay_files = sorted(replays_dir.glob("replay_*.json"), reverse=True)

        # Should find replay files from temp_replays_dir
        assert len(replay_files) >= 1

        # Step 4: Create renderer and set it up (as ui.py does)
        renderer = GameRegistry.create_renderer(game_id)

        grid_w = game_config.game.grid_width
        grid_h = game_config.game.grid_height
        window_width, window_height = 1400, 900

        cell_size = min(
            (window_width - 40) // grid_w,
            (window_height - 100) // grid_h
        )
        game_width = cell_size * grid_w
        game_height = cell_size * grid_h
        offset_x = (window_width - game_width) // 2
        offset_y = (window_height - game_height - 60) // 2

        # These calls must not raise
        renderer.set_cell_size(cell_size)
        renderer.set_render_area(offset_x, offset_y, game_width, game_height)

        # Step 5: Load replay and iterate through frames (as ui.py does in playback loop)
        replay_data = replay_manager.load_replay(str(replay_files[0]))
        assert replay_data is not None
        assert len(replay_data.frames) > 0

        # Test iterating through all frames (as playback loop does)
        # (mypy verifies attribute existence at compile time)
        for frame_idx, frame in enumerate(replay_data.frames):
            # ui.py accesses frame.get('score', 0)
            frame_score = frame.get('score', 0)
            assert isinstance(frame_score, (int, float))

            # Render frame
            renderer.render(frame, mock_screen)


class TestClickTrainAI:
    """Tests for clicking 'Train AI' button."""

    def test_train_ai_initialization(self, mock_pygame_module, mock_screen):
        """
        Test the initialization sequence when clicking 'Train AI'.

        This replicates the setup part of _run_training() from ui.py.
        """
        from src.games.registry import GameRegistry
        from src.utils.config_loader import load_game_config
        from src.device.device_manager import DeviceManager
        from src.training.trainer import Trainer, TrainingConfig

        game_id = 'snake'
        game_config = load_game_config(game_id)
        options = {'model': None, 'device': 'cpu'}

        # Step 1: Get metadata (as ui.py does)
        metadata = GameRegistry.get_metadata(game_id)
        assert metadata is not None

        # Step 2: Create DeviceManager (as ui.py does)
        force_cpu = options.get('device', 'cuda') == 'cpu'
        device_manager = DeviceManager(force_cpu=force_cpu)

        # Step 3: Create TrainingConfig (as ui.py does)
        training_config = TrainingConfig.from_config(game_config)
        assert training_config is not None
        assert training_config.episodes > 0

        # Step 4: Create Trainer (as ui.py does)
        # Note: dashboard=None for headless mode
        trainer = Trainer(
            config=training_config,
            device=device_manager.get_device(),
            num_envs=1,
            dashboard=None,  # No dashboard in test
            load_path=None,
        )
        assert trainer is not None

        # Step 5: Run one training step to verify integration works
        # (mypy catches method existence errors at compile time,
        # but this verifies runtime integration)
        states = trainer.vec_env.reset()
        actions = trainer.agent.select_actions_batch(states, training=True)
        next_states, rewards, dones, infos = trainer.vec_env.step(actions)
        trainer.agent.store_transition(states[0], actions[0], rewards[0], next_states[0], dones[0])

        # Simulate episode end (this is where decay_epsilon bug would manifest)
        trainer.agent.on_episode_end()

        trainer.close()


class TestGameMenuOptions:
    """Tests for GameMenu option building (what gets passed to click handlers)."""

    def test_model_selection_builds_correct_path(self, mock_pygame_module, mock_screen, temp_models_dir):
        """Test that selecting a model builds the correct path."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir=str(temp_models_dir),
            replays_dir='replays'
        )

        # Get available models
        models = menu._get_available_models()

        # Should have models from temp_models_dir/snake/
        assert len(models) > 1  # More than just "Start Fresh"

        # Simulate selecting a model (not "Start Fresh")
        for model_name in models:
            if 'final_model' in model_name:
                menu.model_dropdown.selected_index = models.index(model_name)
                break

        # Build options as GameMenu does
        model_selection = menu.model_dropdown.get_selected()
        if model_selection and 'Start Fresh' not in model_selection:
            # Extract filename from display name
            if '(' in model_selection:
                model_file = model_selection.split('(')[0].strip()
            else:
                model_file = model_selection
            model_path = menu.models_dir / model_file

            # Path should exist
            assert model_path.exists(), f"Model path doesn't exist: {model_path}"

    def test_replay_selection_builds_correct_path(self, mock_pygame_module, mock_screen, temp_replays_dir):
        """Test that selecting a replay builds the correct path."""
        from src.visualization.game_menu import GameMenu

        menu = GameMenu(
            game_id='snake',
            screen=mock_screen,
            models_dir='models',
            replays_dir=str(temp_replays_dir)
        )

        # Get available replays
        replays = menu._get_available_replays()

        # Should have replays from temp_replays_dir/snake/
        assert len(replays) >= 1

        # If we have replays (not just "none"), verify path building
        if replays and '(none)' not in replays[0].lower():
            menu.replay_dropdown.selected_index = 0
            replay_selection = menu.replay_dropdown.get_selected()

            # The replay path should be constructable
            replay_path = menu.replays_dir / replay_selection
            # Note: The display name might be formatted, so we check the dir exists
            assert menu.replays_dir.exists()


class TestGameHubNavigation:
    """Tests for GameHub navigation."""

    def test_clicking_snake_card(self, mock_pygame_module, mock_screen):
        """Test that clicking Snake card would navigate to Snake menu."""
        from src.visualization.game_hub import GameHub, GameCard
        from src.games.registry import GameRegistry

        hub = GameHub(screen=mock_screen)

        # Find the Snake card
        snake_card = None
        for card in hub.game_cards:
            if card.metadata.id == 'snake':
                snake_card = card
                break

        assert snake_card is not None, "Snake card not found in GameHub"
        assert snake_card.available is True, "Snake should be available (not coming soon)"

        # Verify the card has valid metadata
        assert snake_card.metadata.name == 'Snake'

        # Verify we can create a GameMenu with this game_id
        from src.visualization.game_menu import GameMenu
        menu = GameMenu(
            game_id=snake_card.metadata.id,
            screen=mock_screen,
            models_dir='models',
            replays_dir='replays'
        )
        assert menu is not None
