#!/usr/bin/env python3
"""
Novelty AI - Play Script

Watch AI play or play as a human.

Usage:
    python scripts/play.py --game snake                # Watch AI play
    python scripts/play.py --game snake --human        # Play as human
    python scripts/play.py --game snake --model models/snake/model.pth
"""
import sys
import os
import argparse
import warnings
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress pygame messages
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='pygame')

import pygame

from src.device.device_manager import DeviceManager
from src.utils.config_loader import load_game_config
from src.games.registry import GameRegistry
from src.algorithms.dqn.agent import DQNAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Novelty AI - Watch AI play or play as human",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/play.py --game snake              # Watch AI play
  python scripts/play.py --game snake --human      # Play as human
  python scripts/play.py --game snake --model models/snake/model.pth
"""
    )

    parser.add_argument(
        "-g", "--game",
        type=str,
        required=True,
        metavar="GAME_ID",
        help="Game to play (e.g., 'snake')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (default: latest model)"
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Play as human instead of watching AI"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Playback speed (default: 10)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=0,
        help="Number of games to play (0 = infinite)"
    )

    return parser.parse_args()


def find_latest_model(game_id: str) -> Optional[str]:
    """Find the latest model for a game."""
    models_dir = Path(f"models/{game_id}")

    if not models_dir.exists():
        return None

    final = models_dir / "final_model.pth"
    if final.exists():
        return str(final)

    checkpoints = list(models_dir.glob("model_ep*.pth"))
    if checkpoints:
        return str(sorted(checkpoints)[-1])

    return None


def watch_ai_play(game_id: str, model_path: str, device_manager, fps: int = 10, max_games: int = 0):
    """Watch AI play the game."""
    config = load_game_config(game_id)
    metadata = GameRegistry.get_metadata(game_id)
    game_name = metadata.name if metadata else game_id.title()

    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption(f"Novelty AI - {game_name} (AI Playing)")

    # Create environment and renderer
    env = GameRegistry.create_env(
        game_id,
        width=config.game.grid_width,
        height=config.game.grid_height
    )
    renderer = GameRegistry.create_renderer(game_id)

    # Load agent
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        device=device_manager.get_device()
    )
    agent.load(model_path)
    agent.epsilon = 0

    print(f"\nWatching {game_name} AI play...")
    print("Controls: SPACE=pause, +/-=speed, R=restart, ESC=quit\n")

    # Update renderer layout
    grid_w = config.game.grid_width
    grid_h = config.game.grid_height
    cell_size = min((width - 40) // grid_w, (height - 100) // grid_h)
    game_width = cell_size * grid_w
    game_height = cell_size * grid_h
    offset_x = (width - game_width) // 2
    offset_y = (height - game_height - 60) // 2
    renderer.set_cell_size(cell_size)
    renderer.set_render_area(offset_x, offset_y, game_width, game_height)

    running = True
    paused = False
    games_played = 0
    total_score = 0
    high_score = 0
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    state = env.reset()
    current_fps = fps

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                cell_size = min((width - 40) // grid_w, (height - 100) // grid_h)
                game_width = cell_size * grid_w
                game_height = cell_size * grid_h
                offset_x = (width - game_width) // 2
                offset_y = (height - game_height - 60) // 2
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
                    current_fps = min(60, current_fps + 5)
                elif event.key == pygame.K_MINUS:
                    current_fps = max(1, current_fps - 5)

        if paused:
            clock.tick(10)
            continue

        action = agent.select_action(state, training=False)
        state, reward, done, info = env.step(action)

        screen.fill((40, 44, 52))
        game_state = env.get_game_state()
        renderer.render(game_state, screen)

        score_text = font.render(f"Score: {info['score']}", True, (220, 220, 220))
        screen.blit(score_text, (width // 2 - score_text.get_width() // 2, height - 70))

        stats_text = font.render(
            f"Games: {games_played} | High: {high_score} | Avg: {total_score / max(1, games_played):.1f}",
            True, (150, 150, 150)
        )
        screen.blit(stats_text, (width // 2 - stats_text.get_width() // 2, height - 40))

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

            if max_games > 0 and games_played >= max_games:
                running = False
            else:
                pygame.time.wait(500)
                state = env.reset()

        clock.tick(current_fps)

    pygame.quit()
    print(f"\nSession stats: {games_played} games, High: {high_score}, Avg: {total_score / max(1, games_played):.1f}")


def play_human(game_id: str):
    """Play game as a human."""
    from src.games.snake.game import SnakeGame, Direction

    config = load_game_config(game_id)
    metadata = GameRegistry.get_metadata(game_id)
    game_name = metadata.name if metadata else game_id.title()

    if not metadata or not metadata.supports_human:
        print(f"Error: {game_name} doesn't support human play mode")
        sys.exit(1)

    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption(f"Novelty AI - {game_name} (Human Playing)")

    grid_w = config.game.grid_width
    grid_h = config.game.grid_height
    game = SnakeGame(grid_w, grid_h)

    renderer = GameRegistry.create_renderer(game_id)

    cell_size = min((width - 40) // grid_w, (height - 100) // grid_h)
    game_width = cell_size * grid_w
    game_height = cell_size * grid_h
    offset_x = (width - game_width) // 2
    offset_y = (height - game_height - 60) // 2
    renderer.set_cell_size(cell_size)
    renderer.set_render_area(offset_x, offset_y, game_width, game_height)

    print(f"\nPlaying {game_name} as human...")
    print("Controls: Arrow Keys/WASD=move, R=restart, ESC=quit\n")

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
                running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                cell_size = min((width - 40) // grid_w, (height - 100) // grid_h)
                game_width = cell_size * grid_w
                game_height = cell_size * grid_h
                offset_x = (width - game_width) // 2
                offset_y = (height - game_height - 60) // 2
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

        screen.fill((40, 44, 52))
        game_state = game.get_state()
        renderer.render(game_state, screen)

        score_text = font.render(f"Score: {game.score}", True, (220, 220, 220))
        screen.blit(score_text, (width // 2 - score_text.get_width() // 2, height - 70))

        high_text = font.render(f"High Score: {high_score}", True, (150, 150, 150))
        screen.blit(high_text, (width // 2 - high_text.get_width() // 2, height - 40))

        if game_over:
            game_over_text = font_large.render("GAME OVER", True, (255, 100, 100))
            screen.blit(game_over_text, (width // 2 - game_over_text.get_width() // 2, height // 2 - 60))
            restart_text = font.render("Press R to restart", True, (200, 200, 200))
            screen.blit(restart_text, (width // 2 - restart_text.get_width() // 2, height // 2 + 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate game
    available = GameRegistry.list_games()
    available_ids = [g.id for g in available if hasattr(g, 'id')]

    if args.game not in available_ids:
        print(f"Error: Unknown game '{args.game}'")
        print(f"Available games: {', '.join(available_ids)}")
        sys.exit(1)

    metadata = GameRegistry.get_metadata(args.game)
    game_name = metadata.name if metadata else args.game.title()

    print("=" * 60)
    print(f"Novelty AI - {game_name}")
    print("=" * 60)

    if args.human:
        play_human(args.game)
    else:
        # Find model
        model_path = args.model or find_latest_model(args.game)

        if not model_path or not Path(model_path).exists():
            print(f"Error: No model found for game '{args.game}'")
            print("Train a model first or specify --model path")
            sys.exit(1)

        print(f"Model: {model_path}")

        device_manager = DeviceManager(force_cpu=args.cpu)
        watch_ai_play(args.game, model_path, device_manager, fps=args.fps, max_games=args.games)


if __name__ == "__main__":
    main()
