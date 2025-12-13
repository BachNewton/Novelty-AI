#!/usr/bin/env python3
"""
Watch AI Play - Load a trained model and watch the AI play Snake.

Usage:
    python scripts/play.py                           # Use latest model
    python scripts/play.py --model models/best.pth   # Use specific model
    python scripts/play.py --games 10                # Watch 10 games
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pygame
import torch
from src.game.snake_env import SnakeEnv
from src.ai.dqn_agent import DQNAgent
from src.device.device_manager import DeviceManager
from src.game.renderer import StandaloneRenderer
from src.utils.config_loader import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Watch AI play Snake")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (defaults to latest)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=0,
        help="Number of games to play (0 = infinite)"
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=10,
        help="Game speed in FPS (default: 10)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode"
    )

    return parser.parse_args()


def find_latest_model(model_dir: str = "models") -> str:
    """Find the most recent model checkpoint."""
    model_path = Path(model_dir)

    if not model_path.exists():
        return None

    # Look for final model first
    final_model = model_path / "final_model.pth"
    if final_model.exists():
        return str(final_model)

    # Find latest numbered checkpoint
    checkpoints = list(model_path.glob("model_ep*.pth"))
    if not checkpoints:
        return None

    # Sort by episode number
    def get_episode(p):
        try:
            return int(p.stem.split("ep")[1])
        except (IndexError, ValueError):
            return 0

    checkpoints.sort(key=get_episode, reverse=True)
    return str(checkpoints[0])


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config()

    # Find model
    model_path = args.model
    if model_path is None:
        model_path = find_latest_model(config.training.checkpoint_dir)

    if model_path is None or not Path(model_path).exists():
        print("Error: No model found. Train a model first using:")
        print("  python scripts/train.py")
        sys.exit(1)

    print(f"Loading model: {model_path}")

    # Initialize device
    device_manager = DeviceManager(force_cpu=args.cpu)
    device_manager.print_device_info()

    # Initialize environment
    env = SnakeEnv(config.game.grid_width, config.game.grid_height)

    # Initialize agent and load model
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        device=device_manager.get_device()
    )
    agent.load(model_path)
    agent.epsilon = 0  # No exploration during play

    # Initialize renderer
    renderer = StandaloneRenderer(
        grid_width=config.game.grid_width,
        grid_height=config.game.grid_height,
        cell_size=30,
        title="Snake AI - Watch Mode"
    )

    print("\n" + "=" * 50)
    print("Snake AI - Watch Mode")
    print("=" * 50)
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  +/-: Speed up/slow down")
    print("  R: Restart game")
    print("  ESC: Quit")
    print("=" * 50 + "\n")

    # Game loop
    running = True
    paused = False
    games_played = 0
    total_score = 0
    high_score = 0
    fps = args.speed
    clock = pygame.time.Clock()

    state = env.reset()

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_r:
                    state = env.reset()
                    print("Game restarted")
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    fps = min(60, fps + 5)
                    print(f"Speed: {fps} FPS")
                elif event.key == pygame.K_MINUS:
                    fps = max(1, fps - 5)
                    print(f"Speed: {fps} FPS")

        if paused:
            clock.tick(10)
            continue

        # Get AI action
        action = agent.select_action(state, training=False)

        # Execute action
        next_state, reward, done, info = env.step(action)
        state = next_state

        # Render
        game_state = env.get_game_state()
        renderer.surface.fill((0, 0, 0))
        renderer.render(game_state)

        # Draw info
        score_text = renderer.font.render(
            f"Score: {info['score']}",
            True, (220, 220, 220)
        )
        renderer.surface.blit(
            score_text,
            (renderer.window_width // 2 - score_text.get_width() // 2,
             renderer.window_height - 70)
        )

        stats_font = pygame.font.Font(None, 24)
        stats_text = stats_font.render(
            f"Games: {games_played} | High: {high_score} | Avg: {total_score / max(1, games_played):.1f}",
            True, (150, 150, 150)
        )
        renderer.surface.blit(
            stats_text,
            (renderer.window_width // 2 - stats_text.get_width() // 2,
             renderer.window_height - 40)
        )

        pygame.display.flip()

        # Handle game over
        if done:
            games_played += 1
            score = info["score"]
            total_score += score

            if score > high_score:
                high_score = score
                print(f"Game {games_played}: NEW HIGH SCORE {score}!")
            else:
                print(f"Game {games_played}: Score {score}")

            # Check if we've played enough games
            if args.games > 0 and games_played >= args.games:
                running = False
            else:
                # Short pause before restart
                pygame.time.wait(500)
                state = env.reset()

        clock.tick(fps)

    renderer.close()

    # Print final stats
    print("\n" + "=" * 50)
    print("Final Statistics")
    print("=" * 50)
    print(f"Games Played: {games_played}")
    print(f"High Score: {high_score}")
    print(f"Average Score: {total_score / max(1, games_played):.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
