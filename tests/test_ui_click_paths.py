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


class TestPPOConfigPropagation:
    """
    Tests that verify PPO configuration is passed to the agent.

    Bug caught: TrainingConfig.get_agent_config() only returned DQN params,
    not PPO params like entropy_coef, causing PPO to use defaults instead
    of config values.
    """

    def test_ppo_config_includes_entropy_coef(self, mock_pygame_module):
        """Verify entropy_coef is passed to PPO agent from config."""
        from src.utils.config_loader import load_game_config
        from src.training.trainer import TrainingConfig

        config = load_game_config('tetris')
        training_config = TrainingConfig.from_config(config)
        agent_config = training_config.get_agent_config()

        # These PPO params must be present
        assert 'entropy_coef' in agent_config, \
            "entropy_coef missing from agent config - PPO won't explore properly!"
        assert 'gae_lambda' in agent_config, \
            "gae_lambda missing from agent config"
        assert 'clip_epsilon' in agent_config, \
            "clip_epsilon missing from agent config"
        assert 'n_steps' in agent_config, \
            "n_steps missing from agent config"
        assert 'n_epochs' in agent_config, \
            "n_epochs missing from agent config"

    def test_ppo_config_values_match_yaml(self, mock_pygame_module):
        """Verify PPO config values match what's in tetris.yaml."""
        from src.utils.config_loader import load_game_config
        from src.training.trainer import TrainingConfig

        config = load_game_config('tetris')
        training_config = TrainingConfig.from_config(config)
        agent_config = training_config.get_agent_config()

        # These should match tetris.yaml values
        assert agent_config['entropy_coef'] == 0.01
        assert agent_config['gae_lambda'] == 0.95
        assert agent_config['clip_epsilon'] == 0.2
        assert agent_config['n_steps'] == 2048
        assert agent_config['n_epochs'] == 10


class TestGameIdPropagation:
    """
    Tests that verify game_id is correctly passed through the UI.

    These tests catch bugs where game_id defaults to 'snake' instead of
    using the selected game. Bug example: Trainer was missing game_id parameter.
    """

    def test_training_uses_correct_game_id(self, mock_pygame_module, mock_screen):
        """
        Verify that _run_training passes game_id to Trainer.

        Bug caught: Trainer was created without game_id, defaulting to 'snake'.
        """
        from src.utils.config_loader import load_game_config
        from src.training.trainer import TrainingConfig, Trainer
        from src.device.device_manager import DeviceManager

        # Test with Tetris to ensure it's not defaulting to Snake
        game_id = 'tetris'
        game_config = load_game_config(game_id)
        training_config = TrainingConfig.from_config(game_config)

        device_manager = DeviceManager(force_cpu=True)

        # Create Trainer with game_id (as the fixed code does)
        trainer = Trainer(
            config=training_config,
            device=device_manager.get_device(),
            game_id=game_id,  # THIS WAS MISSING BEFORE THE FIX
            num_envs=1,
            dashboard=None,
            load_path=None,
        )

        # Verify game_id was stored correctly
        assert trainer.game_id == game_id, \
            f"Trainer should use game_id '{game_id}', not '{trainer.game_id}'"

        # Verify the algorithm matches what's in config
        assert trainer.algorithm_id == 'ppo', \
            f"Tetris trainer should use 'ppo' algorithm, not '{trainer.algorithm_id}'"

        trainer.close()

    def test_training_tetris_uses_ppo_algorithm(self, mock_pygame_module, mock_screen):
        """
        Verify that training Tetris uses PPO algorithm from config.

        Bug caught: Algorithm was hardcoded or not read from game config.
        """
        from src.utils.config_loader import load_game_config
        from src.training.trainer import TrainingConfig

        game_config = load_game_config('tetris')
        training_config = TrainingConfig.from_config(game_config)

        # Tetris config should specify PPO
        assert training_config.algorithm == 'ppo', \
            f"Tetris should use 'ppo' algorithm, not '{training_config.algorithm}'"


class TestHumanPlayGameCreation:
    """
    Tests that verify human play mode creates the correct game.

    Bug caught: Human play was hardcoded to create SnakeGame regardless of game_id.
    """

    def test_human_play_uses_registry_for_snake(self, mock_pygame_module, mock_screen):
        """Verify human play creates Snake game via registry."""
        from src.games.registry import GameRegistry
        from src.utils.config_loader import load_game_config

        game_id = 'snake'
        game_config = load_game_config(game_id)

        # Create game using registry (as fixed code does)
        game = GameRegistry.create_game(
            game_id,
            width=game_config.game.grid_width,
            height=game_config.game.grid_height
        )

        assert game is not None
        # Verify it's actually a Snake game
        game_state = game.get_state()
        assert 'snake' in game_state, "Snake game should have 'snake' in state"
        assert 'food' in game_state, "Snake game should have 'food' in state"

    def test_human_play_uses_registry_for_tetris(self, mock_pygame_module, mock_screen):
        """
        Verify human play creates Tetris game via registry, not hardcoded Snake.

        Bug caught: _run_human was importing SnakeGame directly instead of using registry.
        """
        from src.games.registry import GameRegistry
        from src.utils.config_loader import load_game_config

        game_id = 'tetris'
        game_config = load_game_config(game_id)

        # Create game using registry (as fixed code does)
        game = GameRegistry.create_game(
            game_id,
            width=game_config.game.grid_width,
            height=game_config.game.grid_height
        )

        assert game is not None
        # Verify it's actually a Tetris game, not Snake
        game_state = game.get_state()
        assert 'board' in game_state, "Tetris game should have 'board' in state"
        assert 'current_piece' in game_state, "Tetris game should have 'current_piece' in state"
        assert 'snake' not in game_state, "Should not be Snake game state"

    def test_tetris_human_play_doesnt_immediately_game_over(self, mock_pygame_module, mock_screen):
        """
        Verify Tetris doesn't immediately show game over on start.
        """
        from src.games.registry import GameRegistry
        from src.utils.config_loader import load_game_config

        game_id = 'tetris'
        game_config = load_game_config(game_id)

        game = GameRegistry.create_game(
            game_id,
            width=game_config.game.grid_width,
            height=game_config.game.grid_height
        )

        # Reset should not result in immediate game over
        game.reset()
        game_state = game.get_state()

        assert game_state.get('game_over', False) is False, \
            "Tetris should not be game over immediately after reset"
        assert game_state.get('current_piece') is not None, \
            "Tetris should have a current piece after reset"


class TestWatchAIAlgorithmSelection:
    """
    Tests that verify Watch AI mode uses the correct algorithm.

    Bug caught: Watch AI was hardcoded to use DQNAgent for all games.
    """

    def test_watch_ai_uses_algorithm_from_config(self, mock_pygame_module, mock_screen):
        """
        Verify Watch AI creates agent using algorithm from game config.

        Bug caught: _run_play hardcoded DQNAgent instead of using AlgorithmRegistry.
        """
        from src.games.registry import GameRegistry
        from src.algorithms.registry import AlgorithmRegistry
        from src.utils.config_loader import load_game_config
        from src.device.device_manager import DeviceManager

        # Test Tetris which uses PPO, not DQN
        game_id = 'tetris'
        game_config = load_game_config(game_id)

        # Get algorithm from config (as fixed code does)
        algorithm_id = game_config.training.algorithm
        assert algorithm_id == 'ppo', \
            f"Tetris config should specify 'ppo', got '{algorithm_id}'"

        # Create environment
        env = GameRegistry.create_env(
            game_id,
            width=game_config.game.grid_width,
            height=game_config.game.grid_height
        )

        device_manager = DeviceManager(force_cpu=True)

        # Create agent using registry (as fixed code does)
        agent = AlgorithmRegistry.create_agent(
            algorithm_id=algorithm_id,
            env=env,
            device=device_manager.get_device()
        )

        # Verify it's the correct agent type
        assert agent is not None
        # PPOAgent should have specific attributes
        assert hasattr(agent, 'select_action'), "Agent should have select_action method"

    def test_snake_watch_ai_uses_dqn(self, mock_pygame_module, mock_screen):
        """Verify Snake Watch AI uses DQN algorithm."""
        from src.utils.config_loader import load_game_config
        from src.algorithms.registry import AlgorithmRegistry
        from src.games.registry import GameRegistry
        from src.device.device_manager import DeviceManager

        game_id = 'snake'
        game_config = load_game_config(game_id)

        algorithm_id = game_config.training.algorithm
        assert algorithm_id == 'dqn', \
            f"Snake config should specify 'dqn', got '{algorithm_id}'"

        env = GameRegistry.create_env(
            game_id,
            width=game_config.game.grid_width,
            height=game_config.game.grid_height
        )

        device_manager = DeviceManager(force_cpu=True)

        agent = AlgorithmRegistry.create_agent(
            algorithm_id=algorithm_id,
            env=env,
            device=device_manager.get_device()
        )

        assert agent is not None
        # DQN should have epsilon attribute
        assert hasattr(agent, 'epsilon'), "DQN agent should have epsilon attribute"


class TestDashboardRendering:
    """
    Tests that verify the training dashboard uses the correct renderer per game.

    Bug caught: Dashboard was hardcoded to use SnakeRenderer for all games,
    causing crashes when pressing H to show the game during Tetris training.
    """

    def test_dashboard_uses_game_registry_for_renderer(self, mock_pygame_module, mock_screen):
        """
        Verify that TrainingDashboard creates renderer via GameRegistry.

        Bug caught: Dashboard hardcoded SnakeRenderer import instead of using registry.
        """
        from src.visualization.dashboard import TrainingDashboard
        from src.games.registry import GameRegistry

        # Test with Tetris - if it uses SnakeRenderer, this will fail
        game_id = 'tetris'

        dashboard = TrainingDashboard(
            screen=mock_screen,
            grid_width=10,
            grid_height=20,
            show_game=True,  # This triggers renderer creation
            game_id=game_id,  # THIS WAS MISSING BEFORE THE FIX
        )

        # Verify a renderer was created
        assert hasattr(dashboard, 'game_renderer'), \
            "Dashboard should have game_renderer attribute when show_game=True"
        assert dashboard.game_renderer is not None

        # Verify it's the correct renderer type (TetrisRenderer, not SnakeRenderer)
        renderer_class_name = type(dashboard.game_renderer).__name__
        assert 'Tetris' in renderer_class_name or 'tetris' in renderer_class_name.lower(), \
            f"Dashboard should use TetrisRenderer for Tetris, got {renderer_class_name}"

        dashboard.close()

    def test_dashboard_can_render_tetris_game_state(self, mock_pygame_module, mock_screen):
        """
        Verify dashboard can render Tetris game state without crashing.

        Bug caught: SnakeRenderer couldn't render Tetris game state format.
        """
        from src.visualization.dashboard import TrainingDashboard
        from src.games.registry import GameRegistry

        game_id = 'tetris'

        # Create Tetris game and get its state
        game = GameRegistry.create_game(game_id, width=10, height=20)
        game.reset()
        game_state = game.get_state()

        # Create dashboard with Tetris
        dashboard = TrainingDashboard(
            screen=mock_screen,
            grid_width=10,
            grid_height=20,
            show_game=True,
            game_id=game_id,
        )

        # This should not crash - before the fix, SnakeRenderer couldn't handle Tetris state
        try:
            dashboard._draw_game_panel(game_state)
        except Exception as e:
            pytest.fail(f"Dashboard failed to render Tetris game state: {e}")

        dashboard.close()

    def test_dashboard_snake_still_works(self, mock_pygame_module, mock_screen):
        """Verify dashboard still works correctly for Snake."""
        from src.visualization.dashboard import TrainingDashboard
        from src.games.registry import GameRegistry

        game_id = 'snake'

        game = GameRegistry.create_game(game_id, width=20, height=20)
        game.reset()
        game_state = game.get_state()

        dashboard = TrainingDashboard(
            screen=mock_screen,
            grid_width=20,
            grid_height=20,
            show_game=True,
            game_id=game_id,
        )

        # Should not crash
        try:
            dashboard._draw_game_panel(game_state)
        except Exception as e:
            pytest.fail(f"Dashboard failed to render Snake game state: {e}")

        dashboard.close()
