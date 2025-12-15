#!/usr/bin/env python3
"""
Novelty AI - Unified UI Entry Point

Provides a graphical interface to access the AI Training Hub:
- Game Hub for selecting games
- Per-game menus for Training, Watch AI, Play Human, Replays

All modes share a single window - no window closing/reopening.

Usage:
    python scripts/ui.py              # Opens Game Hub
    python scripts/ui.py --game snake # Goes directly to Snake menu
    python scripts/ui.py -g snake     # Short form
"""
import sys
import os
import ctypes
import warnings
import subprocess
import argparse
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress pygame messages
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='pygame')

import pygame

from src.utils.config_loader import load_config, load_game_config
from src.visualization.game_hub import GameHub
from src.visualization.game_menu import GameMenu
from src.visualization.dashboard import TrainingDashboard
from src.training import Trainer, TrainingConfig, get_default_num_envs
from src.games.registry import GameRegistry


@contextmanager
def prevent_sleep():
    """Prevent Windows from sleeping while training."""
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


class UnifiedUI:
    """
    Unified UI that maintains a single window throughout all modes.
    Supports both Game Hub (multi-game selection) and direct game access.
    """

    MIN_WIDTH = 800
    MIN_HEIGHT = 600

    def __init__(self, config, initial_game: Optional[str] = None):
        """
        Initialize the UI.

        Args:
            config: Default configuration
            initial_game: If specified, skip hub and go directly to this game
        """
        self.config = config
        self.initial_game = initial_game
        self.current_game = initial_game

        # Initialize pygame and create the main window
        pygame.init()
        self.window_width = config.visualization.window_width
        self.window_height = config.visualization.window_height
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height),
            pygame.RESIZABLE
        )
        pygame.display.set_caption("Novelty AI")

        self.running = True

    def run(self):
        """Main loop - show hub/menu and run selected modes."""
        while self.running:
            # If we have a game selected, show game menu
            # Otherwise show the game hub
            if self.current_game:
                result = self._run_game_menu(self.current_game)
                if result == 'back':
                    # Go back to hub (unless we started with --game)
                    if self.initial_game:
                        break  # Exit if started with --game
                    self.current_game = None
                elif result == 'quit':
                    break
            else:
                # Show game hub
                action = self._run_game_hub()
                if action == 'quit':
                    break
                elif action == 'settings':
                    # Settings not implemented yet - just continue
                    print("[Settings] Not yet implemented")
                    continue
                elif action:
                    self.current_game = action

        pygame.quit()

    def _run_game_hub(self):
        """Run the game hub for game selection."""
        pygame.display.set_caption("Novelty AI - Game Hub")

        hub = GameHub(screen=self.screen)
        result, options = hub.run()  # Unpack tuple
        hub.close()

        return result  # Return just the action/game_id

    def _run_game_menu(self, game_id: str):
        """Run the game-specific menu."""
        # Load game-specific config
        game_config = load_game_config(game_id)
        self.training_config = TrainingConfig.from_config(game_config)

        # Get game metadata
        metadata = GameRegistry.get_metadata(game_id)
        game_name = metadata.name if metadata else game_id.title()

        pygame.display.set_caption(f"Novelty AI - {game_name}")

        menu = GameMenu(
            screen=self.screen,
            game_id=game_id,
            models_dir="models",
            replays_dir="replays"
        )
        mode, options = menu.run()
        menu.close()

        if mode == 'quit':
            return 'quit'
        elif mode == 'back':
            return 'back'
        elif mode == 'training':
            self._run_training(options, game_id, game_config)
        elif mode == 'play':
            self._run_play(options, game_id, game_config)
        elif mode == 'human':
            self._run_human(game_id, game_config)
        elif mode == 'replays':
            self._run_replays(options, game_id, game_config)

        return 'continue'

    def _handle_resize(self, event):
        """Handle window resize event."""
        self.window_width = max(event.w, self.MIN_WIDTH)
        self.window_height = max(event.h, self.MIN_HEIGHT)
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height),
            pygame.RESIZABLE
        )

    def _show_loading_screen(self, message="Loading..."):
        """Show a loading screen while initializing."""
        self.screen.fill((40, 44, 52))  # Novelty AI background
        font_large = pygame.font.Font(None, 48)
        font_small = pygame.font.Font(None, 28)

        # Draw loading message
        text = font_large.render(message, True, (220, 220, 220))
        self.screen.blit(text, (self.window_width // 2 - text.get_width() // 2,
                                self.window_height // 2 - 30))

        hint = font_small.render("Please wait...", True, (100, 100, 100))
        self.screen.blit(hint, (self.window_width // 2 - hint.get_width() // 2,
                                self.window_height // 2 + 20))

        pygame.display.flip()

    def _run_training(self, options, game_id: str, game_config):
        """Run training mode using shared Trainer."""
        metadata = GameRegistry.get_metadata(game_id)
        game_name = metadata.name if metadata else game_id.title()

        self._show_loading_screen(f"Initializing {game_name} Training...")
        pygame.display.set_caption(f"Novelty AI - {game_name} Training")

        from src.device.device_manager import DeviceManager

        print("\n" + "=" * 60)
        print(f"Novelty AI - {game_name} Training")
        print("=" * 60)

        # Use device selection from menu (defaults to GPU if available)
        force_cpu = options.get('device', 'cuda') == 'cpu'
        device_manager = DeviceManager(force_cpu=force_cpu)
        device_manager.print_device_info()

        num_envs = get_default_num_envs()
        show_game = not options.get('headless', False)

        print(f"Mode: {'Visual' if show_game else 'Headless'}")
        print(f"Parallel Environments: {num_envs}")
        print("Press H to toggle between visual/headless mode")
        print("Press ESC to return to menu")
        print("=" * 60 + "\n")

        # Initialize dashboard with the shared screen
        dashboard = TrainingDashboard(
            screen=self.screen,
            grid_width=game_config.game.grid_width,
            grid_height=game_config.game.grid_height,
            chart_update_interval=game_config.visualization.chart_update_interval,
            show_game=show_game,
            total_episodes=game_config.training.episodes,
            num_envs=num_envs,
        )

        # Load model path if specified
        model_path = options.get('model')
        if model_path and not Path(model_path).exists():
            model_path = None

        # Create trainer with shared training logic
        trainer = Trainer(
            config=self.training_config,
            device=device_manager.get_device(),
            num_envs=num_envs,
            dashboard=dashboard,
            on_high_score=self._play_high_score_replay,
            load_path=model_path,
        )

        try:
            with prevent_sleep():
                high_score = trainer.train(render_fps=game_config.visualization.render_fps)
        except KeyboardInterrupt:
            print("\n[Training] Interrupted by user")
            high_score = trainer.high_score
        finally:
            trainer.save_final_model()
            trainer.close()

        print(f"\n[Training] Complete! High score: {high_score}")
        pygame.display.set_caption("Novelty AI")

    def _run_play(self, options, game_id: str, game_config):
        """Run AI watch mode."""
        from src.device.device_manager import DeviceManager
        from src.algorithms.dqn.agent import DQNAgent

        metadata = GameRegistry.get_metadata(game_id)
        game_name = metadata.name if metadata else game_id.title()

        pygame.display.set_caption(f"Novelty AI - {game_name} Watch Mode")

        # Find model
        model_path = options.get('model')
        models_dir = Path(f"models/{game_id}")
        if not model_path or not Path(model_path).exists():
            final = models_dir / "final_model.pth"
            if final.exists():
                model_path = str(final)
            else:
                checkpoints = list(models_dir.glob("model_ep*.pth"))
                if checkpoints:
                    model_path = str(sorted(checkpoints)[-1])

        if not model_path or not Path(model_path).exists():
            print("Error: No model found. Train a model first.")
            self._show_error("No model found. Train a model first.")
            return

        print(f"\nLoading model: {model_path}")

        # Use device selection from menu
        force_cpu = options.get('device', 'cuda') == 'cpu'
        device_manager = DeviceManager(force_cpu=force_cpu)

        # Create environment using registry
        env = GameRegistry.create_env(game_id, width=game_config.game.grid_width, height=game_config.game.grid_height)

        agent = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            device=device_manager.get_device()
        )
        agent.load(model_path)
        agent.epsilon = 0

        # Create renderer using registry
        renderer = GameRegistry.create_renderer(game_id)

        # Calculate cell size for centered game
        cell_size = min(
            (self.window_width - 40) // game_config.game.grid_width,
            (self.window_height - 100) // game_config.game.grid_height
        )
        game_width = cell_size * game_config.game.grid_width
        game_height = cell_size * game_config.game.grid_height
        offset_x = (self.window_width - game_width) // 2
        offset_y = (self.window_height - game_height - 60) // 2
        renderer.set_cell_size(cell_size)
        renderer.set_render_area(offset_x, offset_y, game_width, game_height)

        print("\nControls: SPACE=pause, +/-=speed, R=restart, ESC=menu\n")

        running = True
        paused = False
        games_played = 0
        total_score = 0
        high_score = 0
        fps = 10
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 32)
        state = env.reset()

        grid_w = game_config.game.grid_width
        grid_h = game_config.game.grid_height

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event)
                    # Recalculate layout
                    cell_size = min(
                        (self.window_width - 40) // grid_w,
                        (self.window_height - 100) // grid_h
                    )
                    game_width = cell_size * grid_w
                    game_height = cell_size * grid_h
                    offset_x = (self.window_width - game_width) // 2
                    offset_y = (self.window_height - game_height - 60) // 2
                    renderer.set_cell_size(cell_size)
                    renderer.set_render_area(offset_x, offset_y, game_width, game_height)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        state = env.reset()
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        fps = min(60, fps + 5)
                    elif event.key == pygame.K_MINUS:
                        fps = max(1, fps - 5)

            if paused:
                clock.tick(10)
                continue

            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state

            self.screen.fill((40, 44, 52))  # Use Novelty AI background
            game_state = env.get_game_state()
            renderer.render(game_state, self.screen)

            # Draw info
            score_text = font.render(f"Score: {info['score']}", True, (220, 220, 220))
            self.screen.blit(score_text, (self.window_width // 2 - score_text.get_width() // 2, self.window_height - 70))

            stats_text = font.render(
                f"Games: {games_played} | High: {high_score} | Avg: {total_score / max(1, games_played):.1f}",
                True, (150, 150, 150)
            )
            self.screen.blit(stats_text, (self.window_width // 2 - stats_text.get_width() // 2, self.window_height - 40))

            pygame.display.flip()

            if done:
                games_played += 1
                score = info["score"]
                total_score += score
                if score > high_score:
                    high_score = score
                    print(f"Game {games_played}: NEW HIGH SCORE {score}!")
                else:
                    print(f"Game {games_played}: Score {score}")
                pygame.time.wait(500)
                state = env.reset()

            clock.tick(fps)

        pygame.display.set_caption("Novelty AI")

    def _run_human(self, game_id: str, game_config):
        """Run human play mode."""
        # Currently Snake-specific - future games can add their own human play logic
        from src.games.snake.game import SnakeGame, Direction

        metadata = GameRegistry.get_metadata(game_id)
        game_name = metadata.name if metadata else game_id.title()

        if not metadata or not metadata.supports_human:
            self._show_error(f"{game_name} doesn't support human play mode.")
            return

        pygame.display.set_caption(f"Novelty AI - {game_name} Human Mode")

        grid_w = game_config.game.grid_width
        grid_h = game_config.game.grid_height
        game = SnakeGame(grid_w, grid_h)

        # Create renderer using registry
        renderer = GameRegistry.create_renderer(game_id)

        cell_size = min(
            (self.window_width - 40) // grid_w,
            (self.window_height - 100) // grid_h
        )
        game_width = cell_size * grid_w
        game_height = cell_size * grid_h
        offset_x = (self.window_width - game_width) // 2
        offset_y = (self.window_height - game_height - 60) // 2
        renderer.set_cell_size(cell_size)
        renderer.set_render_area(offset_x, offset_y, game_width, game_height)

        print("\nControls: Arrow Keys/WASD=move, R=restart, ESC=menu\n")

        running = True
        game_over = False
        clock = pygame.time.Clock()
        move_delay = 100
        last_move_time = 0
        pending_direction = None
        high_score = 0
        font = pygame.font.Font(None, 32)
        font_large = pygame.font.Font(None, 72)

        while running:
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event)
                    cell_size = min(
                        (self.window_width - 40) // grid_w,
                        (self.window_height - 100) // grid_h
                    )
                    game_width = cell_size * grid_w
                    game_height = cell_size * grid_h
                    offset_x = (self.window_width - game_width) // 2
                    offset_y = (self.window_height - game_height - 60) // 2
                    renderer.set_cell_size(cell_size)
                    renderer.set_render_area(offset_x, offset_y, game_width, game_height)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.reset()
                        game_over = False
                        pending_direction = None
                    elif not game_over:
                        if event.key in (pygame.K_UP, pygame.K_w):
                            pending_direction = Direction.UP
                        elif event.key in (pygame.K_DOWN, pygame.K_s):
                            pending_direction = Direction.DOWN
                        elif event.key in (pygame.K_LEFT, pygame.K_a):
                            pending_direction = Direction.LEFT
                        elif event.key in (pygame.K_RIGHT, pygame.K_d):
                            pending_direction = Direction.RIGHT

            if not game_over and current_time - last_move_time >= move_delay:
                if pending_direction is not None:
                    _, _, game_over, info = game.step_direction(pending_direction)
                else:
                    _, _, game_over, info = game.step(0)
                last_move_time = current_time

                if game_over:
                    final_score = game.score
                    if final_score > high_score:
                        high_score = final_score
                        print(f"NEW HIGH SCORE: {final_score}!")
                    else:
                        print(f"Game Over! Score: {final_score}")

            self.screen.fill((40, 44, 52))  # Novelty AI background
            game_state = game.get_state()
            renderer.render(game_state, self.screen)

            score_text = font.render(f"Score: {game.score}", True, (220, 220, 220))
            self.screen.blit(score_text, (self.window_width // 2 - score_text.get_width() // 2, self.window_height - 70))

            high_text = font.render(f"High Score: {high_score}", True, (150, 150, 150))
            self.screen.blit(high_text, (self.window_width // 2 - high_text.get_width() // 2, self.window_height - 40))

            if game_over:
                game_over_text = font_large.render("GAME OVER", True, (255, 100, 100))
                self.screen.blit(game_over_text, (self.window_width // 2 - game_over_text.get_width() // 2, self.window_height // 2 - 60))
                restart_text = font.render("Press R to restart", True, (200, 200, 200))
                self.screen.blit(restart_text, (self.window_width // 2 - restart_text.get_width() // 2, self.window_height // 2 + 20))

            pygame.display.flip()
            clock.tick(60)

        pygame.display.set_caption("Novelty AI")

    def _run_replays(self, options, game_id: str, game_config):
        """Run replay viewer."""
        from src.visualization.replay_player import ReplayManager

        metadata = GameRegistry.get_metadata(game_id)
        game_name = metadata.name if metadata else game_id.title()

        pygame.display.set_caption(f"Novelty AI - {game_name} Replay Viewer")

        replays_dir = f"replays/{game_id}"
        replay_manager = ReplayManager(save_dir=replays_dir)

        replay_selection = options.get('replay', 'all')

        if replay_selection == 'all':
            replay_files = replay_manager.list_replays()
            if replay_files:
                scored = []
                for path in replay_files:
                    try:
                        replay = replay_manager.load_replay(path)
                        scored.append((replay.score, path))
                    except:
                        scored.append((0, path))
                scored.sort(key=lambda x: x[0], reverse=True)
                replay_files = [path for _, path in scored]
        elif replay_selection and Path(replay_selection).exists():
            replay_files = [replay_selection]
        else:
            replay_files = []

        if not replay_files:
            print("No replays found.")
            self._show_error("No replays found. Train the AI first.")
            return

        grid_w = game_config.game.grid_width
        grid_h = game_config.game.grid_height

        # Create renderer using registry
        renderer = GameRegistry.create_renderer(game_id)

        cell_size = min(
            (self.window_width - 40) // grid_w,
            (self.window_height - 100) // grid_h
        )
        game_width = cell_size * grid_w
        game_height = cell_size * grid_h
        offset_x = (self.window_width - game_width) // 2
        offset_y = (self.window_height - game_height - 60) // 2
        renderer.set_cell_size(cell_size)
        renderer.set_render_area(offset_x, offset_y, game_width, game_height)

        print(f"\nReplays to watch: {len(replay_files)}")
        print("Controls: SPACE=pause, LEFT/RIGHT=skip, +/-=speed, N=next, ESC=menu\n")

        fps = 10
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 28)

        for i, filepath in enumerate(replay_files, 1):
            try:
                replay_data = replay_manager.load_replay(filepath)
            except Exception as e:
                print(f"Error loading replay: {e}")
                continue

            print(f"Playing replay {i}/{len(replay_files)}: Episode {replay_data.episode}, Score {replay_data.score}")

            frame_idx = 0
            paused = False
            playing = True

            while playing and frame_idx < len(replay_data.frames):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        return
                    elif event.type == pygame.VIDEORESIZE:
                        self._handle_resize(event)
                        cell_size = min(
                            (self.window_width - 40) // grid_w,
                            (self.window_height - 100) // grid_h
                        )
                        game_width = cell_size * grid_w
                        game_height = cell_size * grid_h
                        offset_x = (self.window_width - game_width) // 2
                        offset_y = (self.window_height - game_height - 60) // 2
                        renderer.set_cell_size(cell_size)
                        renderer.set_render_area(offset_x, offset_y, game_width, game_height)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.display.set_caption("Novelty AI")
                            return
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_n:
                            playing = False
                        elif event.key == pygame.K_LEFT:
                            frame_idx = max(0, frame_idx - 10)
                        elif event.key == pygame.K_RIGHT:
                            frame_idx = min(len(replay_data.frames) - 1, frame_idx + 10)
                        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                            fps = min(60, fps + 5)
                        elif event.key == pygame.K_MINUS:
                            fps = max(1, fps - 5)

                if paused:
                    clock.tick(10)
                    continue

                frame = replay_data.frames[frame_idx]
                self.screen.fill((40, 44, 52))  # Novelty AI background
                renderer.render(frame, self.screen)

                score_text = font.render(f"Score: {frame.get('score', 0)}", True, (220, 220, 220))
                self.screen.blit(score_text, (10, 10))

                ep_text = font.render(f"Episode {replay_data.episode} | Final Score: {replay_data.score}", True, (150, 150, 150))
                self.screen.blit(ep_text, (self.window_width // 2 - ep_text.get_width() // 2, self.window_height - 60))

                progress_text = font.render(f"Frame {frame_idx + 1}/{len(replay_data.frames)} | Replay {i}/{len(replay_files)}", True, (100, 100, 100))
                self.screen.blit(progress_text, (self.window_width // 2 - progress_text.get_width() // 2, self.window_height - 30))

                pygame.display.flip()
                frame_idx += 1
                clock.tick(fps)

            pygame.time.wait(500)

        pygame.display.set_caption("Novelty AI")

    def _play_high_score_replay(self, replay_path, score, episode):
        """Queue a high score replay. Launches viewer if not already running."""
        # Ensure absolute path for the replay file
        replay_path_abs = Path(replay_path)
        if not replay_path_abs.is_absolute():
            replay_path_abs = project_root / replay_path
        replay_path_str = str(replay_path_abs)

        # Queue file for replay viewer
        queue_file = project_root / "replay_queue.txt"

        # Add replay to queue
        with open(queue_file, "a") as f:
            f.write(f"{replay_path_str}|{score}|{episode}\n")

        # Check if replay viewer is already running
        if hasattr(self, '_replay_process') and self._replay_process is not None:
            if self._replay_process.poll() is None:
                print(f"[Replay] Queued: Score {score} (Episode {episode})")
                return
            else:
                self._replay_process = None

        # Launch replay viewer
        try:
            script_path = str(project_root / "scripts" / "watch_replays.py")

            env = os.environ.copy()
            env["SDL_VIDEO_WINDOW_POS"] = "100,100"

            if sys.platform == "win32":
                python_exe = sys.executable
                if python_exe.endswith("python.exe"):
                    pythonw = python_exe.replace("python.exe", "pythonw.exe")
                    if Path(pythonw).exists():
                        python_exe = pythonw

                self._replay_process = subprocess.Popen(
                    [python_exe, script_path, "--queue", str(queue_file)],
                    env=env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    close_fds=True
                )
            else:
                self._replay_process = subprocess.Popen(
                    [sys.executable, script_path, "--queue", str(queue_file)],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
            print(f"[Replay] Playing: Score {score} (Episode {episode})")
        except Exception as e:
            print(f"[Replay] Failed to launch replay viewer: {e}")

    def _show_error(self, message):
        """Show error message for a few seconds."""
        font = pygame.font.Font(None, 36)
        clock = pygame.time.Clock()

        for _ in range(120):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                elif event.type == pygame.KEYDOWN:
                    return

            self.screen.fill((40, 44, 52))  # Novelty AI background
            text = font.render(message, True, (255, 100, 100))
            self.screen.blit(text, (self.window_width // 2 - text.get_width() // 2, self.window_height // 2))

            hint = font.render("Press any key to continue", True, (150, 150, 150))
            self.screen.blit(hint, (self.window_width // 2 - hint.get_width() // 2, self.window_height // 2 + 50))

            pygame.display.flip()
            clock.tick(60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Novelty AI - AI Training Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ui.py              # Opens Game Hub
  python scripts/ui.py --game snake # Goes directly to Snake menu
  python scripts/ui.py -g snake     # Short form
"""
    )
    parser.add_argument(
        "-g", "--game",
        type=str,
        metavar="GAME_ID",
        help="Go directly to a specific game menu (e.g., 'snake')"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load default config
    config = load_config()

    print("\n" + "=" * 60)
    print("Novelty AI - AI Training Hub")
    print("=" * 60)

    # Validate game ID if provided
    if args.game:
        available = GameRegistry.list_games()
        available_ids = [g.id for g in available if hasattr(g, 'id')]
        if args.game not in available_ids:
            print(f"\nError: Unknown game '{args.game}'")
            print(f"Available games: {', '.join(available_ids)}")
            sys.exit(1)
        print(f"Starting with game: {args.game}")

    ui = UnifiedUI(config, initial_game=args.game)
    ui.run()

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
