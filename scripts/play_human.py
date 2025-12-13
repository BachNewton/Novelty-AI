#!/usr/bin/env python3
"""
Human Play Mode - Play the Snake game yourself.

Controls:
    Arrow Keys or WASD: Move the snake
    R: Restart game
    ESC: Quit
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pygame
from src.game.snake_game import SnakeGame, Direction
from src.game.renderer import StandaloneRenderer
from src.utils.config_loader import load_config


def main():
    """Main entry point for human play mode."""
    # Load config
    config = load_config()

    # Initialize game
    game = SnakeGame(config.game.grid_width, config.game.grid_height)

    # Initialize renderer
    renderer = StandaloneRenderer(
        grid_width=config.game.grid_width,
        grid_height=config.game.grid_height,
        cell_size=30,
        title="Snake Game - Human Mode"
    )

    print("\n" + "=" * 50)
    print("Snake Game - Human Mode")
    print("=" * 50)
    print("Controls:")
    print("  Arrow Keys / WASD: Move")
    print("  R: Restart")
    print("  ESC: Quit")
    print("=" * 50 + "\n")

    # Game loop
    running = True
    game_over = False
    clock = pygame.time.Clock()
    move_delay = 100  # ms between moves
    last_move_time = 0
    pending_direction = None
    high_score = 0

    while running:
        current_time = pygame.time.get_ticks()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    # Restart game
                    game.reset()
                    game_over = False
                    pending_direction = None

                elif not game_over:
                    # Movement keys
                    if event.key in (pygame.K_UP, pygame.K_w):
                        pending_direction = Direction.UP
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        pending_direction = Direction.DOWN
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        pending_direction = Direction.LEFT
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        pending_direction = Direction.RIGHT

        # Process movement at regular intervals
        if not game_over and current_time - last_move_time >= move_delay:
            if pending_direction is not None:
                _, _, game_over, info = game.step_direction(pending_direction)
            else:
                # Continue in current direction
                _, _, game_over, info = game.step(0)

            last_move_time = current_time

            if game_over:
                final_score = game.score
                if final_score > high_score:
                    high_score = final_score
                    print(f"NEW HIGH SCORE: {final_score}!")
                else:
                    print(f"Game Over! Score: {final_score} | High Score: {high_score}")

        # Render
        renderer.surface.fill((0, 0, 0))
        game_state = game.get_state()
        renderer.render(game_state)

        # Draw score
        score_text = renderer.font.render(
            f"Score: {game.score}",
            True, (220, 220, 220)
        )
        renderer.surface.blit(
            score_text,
            (renderer.window_width // 2 - score_text.get_width() // 2,
             renderer.window_height - 70)
        )

        # Draw high score
        high_score_text = pygame.font.Font(None, 28).render(
            f"High Score: {high_score}",
            True, (150, 150, 150)
        )
        renderer.surface.blit(
            high_score_text,
            (renderer.window_width // 2 - high_score_text.get_width() // 2,
             renderer.window_height - 40)
        )

        # Draw game over message
        if game_over:
            font_large = pygame.font.Font(None, 72)
            game_over_text = font_large.render("GAME OVER", True, (255, 100, 100))
            renderer.surface.blit(
                game_over_text,
                (renderer.window_width // 2 - game_over_text.get_width() // 2,
                 renderer.window_height // 2 - 60)
            )

            font_medium = pygame.font.Font(None, 36)
            restart_text = font_medium.render("Press R to restart", True, (200, 200, 200))
            renderer.surface.blit(
                restart_text,
                (renderer.window_width // 2 - restart_text.get_width() // 2,
                 renderer.window_height // 2 + 20)
            )

        pygame.display.flip()
        clock.tick(60)

    renderer.close()
    print(f"\nFinal High Score: {high_score}")


if __name__ == "__main__":
    main()
