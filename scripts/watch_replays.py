#!/usr/bin/env python3
"""
Watch Replays - Play back saved high-score game replays.

Usage:
    python scripts/watch_replays.py              # Watch all replays (best first)
    python scripts/watch_replays.py --latest     # Watch most recent replay
    python scripts/watch_replays.py --list       # List available replays
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pygame
from src.visualization.replay_player import ReplayManager
from src.game.renderer import StandaloneRenderer
from src.utils.config_loader import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Watch saved replays")

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available replays and exit"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Watch only the most recent replay"
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Watch only the best (highest score) replay"
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=10,
        help="Playback speed in FPS (default: 10)"
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Path to specific replay file"
    )

    return parser.parse_args()


def get_replay_info(replay_manager, filepath):
    """Get score and episode from replay file."""
    try:
        replay = replay_manager.load_replay(filepath)
        return replay.score, replay.episode, len(replay.frames)
    except Exception:
        return 0, 0, 0


def list_replays(replay_manager):
    """List all available replays."""
    replays = replay_manager.list_replays()

    if not replays:
        print("No replays found.")
        return

    print("\nAvailable Replays:")
    print("=" * 60)
    print(f"{'Score':<10} {'Episode':<10} {'Frames':<10} {'File'}")
    print("-" * 60)

    replay_data = []
    for path in replays:
        score, episode, frames = get_replay_info(replay_manager, path)
        replay_data.append((score, episode, frames, path))

    # Sort by score (highest first)
    replay_data.sort(key=lambda x: x[0], reverse=True)

    for score, episode, frames, path in replay_data:
        filename = Path(path).name
        print(f"{score:<10} {episode:<10} {frames:<10} {filename}")

    print("=" * 60)
    print(f"Total: {len(replays)} replays")


def play_replay(renderer, replay_data, fps, replay_num, total_replays):
    """Play a single replay. Returns True to continue, False to quit."""
    clock = pygame.time.Clock()
    frame_idx = 0
    paused = False
    running = True

    print(f"\nPlaying replay {replay_num}/{total_replays}: "
          f"Episode {replay_data.episode}, Score {replay_data.score}")
    print("Controls: SPACE=pause, LEFT/RIGHT=skip, +/-=speed, N=next, ESC=quit")

    while running and frame_idx < len(replay_data.frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    # Skip to next replay
                    return True
                elif event.key == pygame.K_LEFT:
                    frame_idx = max(0, frame_idx - 10)
                elif event.key == pygame.K_RIGHT:
                    frame_idx = min(len(replay_data.frames) - 1, frame_idx + 10)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    fps = min(60, fps + 5)
                    print(f"Speed: {fps} FPS")
                elif event.key == pygame.K_MINUS:
                    fps = max(1, fps - 5)
                    print(f"Speed: {fps} FPS")

        if paused:
            clock.tick(10)
            continue

        # Get current frame
        frame = replay_data.frames[frame_idx]

        # Render
        renderer.surface.fill((0, 0, 0))
        renderer.render(frame)

        # Draw info overlay
        info_font = pygame.font.Font(None, 28)

        # Score
        score_text = info_font.render(
            f"Score: {frame.get('score', 0)}",
            True, (220, 220, 220)
        )
        renderer.surface.blit(score_text, (10, 10))

        # Episode info
        ep_text = info_font.render(
            f"Episode {replay_data.episode} | Final Score: {replay_data.score}",
            True, (150, 150, 150)
        )
        renderer.surface.blit(
            ep_text,
            (renderer.window_width // 2 - ep_text.get_width() // 2,
             renderer.window_height - 60)
        )

        # Progress
        progress_text = info_font.render(
            f"Frame {frame_idx + 1}/{len(replay_data.frames)} | "
            f"Replay {replay_num}/{total_replays}",
            True, (100, 100, 100)
        )
        renderer.surface.blit(
            progress_text,
            (renderer.window_width // 2 - progress_text.get_width() // 2,
             renderer.window_height - 30)
        )

        pygame.display.flip()

        frame_idx += 1
        clock.tick(fps)

    # Brief pause at end
    pygame.time.wait(1000)
    return True


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config()

    replay_manager = ReplayManager(save_dir=config.replay.save_dir)

    # List mode
    if args.list:
        list_replays(replay_manager)
        return

    # Get replays to play
    if args.replay:
        replay_files = [args.replay]
    elif args.best:
        best = replay_manager.get_best_replay()
        if best:
            replay_files = [best]
        else:
            print("No replays found.")
            return
    elif args.latest:
        replays = replay_manager.list_replays()
        if replays:
            replay_files = [replays[0]]  # Already sorted newest first
        else:
            print("No replays found.")
            return
    else:
        # All replays, sorted by score (best first)
        replay_files = replay_manager.list_replays()
        if replay_files:
            # Sort by score
            scored = []
            for path in replay_files:
                score, _, _ = get_replay_info(replay_manager, path)
                scored.append((score, path))
            scored.sort(key=lambda x: x[0], reverse=True)
            replay_files = [path for _, path in scored]

    if not replay_files:
        print("No replays found. Train the AI first:")
        print("  python scripts/train.py --headless")
        return

    # Initialize renderer
    renderer = StandaloneRenderer(
        grid_width=config.game.grid_width,
        grid_height=config.game.grid_height,
        cell_size=30,
        title="Snake AI - Replay Viewer"
    )

    print("\n" + "=" * 50)
    print("Snake AI - Replay Viewer")
    print("=" * 50)
    print(f"Replays to watch: {len(replay_files)}")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  LEFT/RIGHT: Skip frames")
    print("  +/-: Speed up/slow down")
    print("  N: Next replay")
    print("  ESC: Quit")
    print("=" * 50)

    fps = args.speed

    # Play each replay
    for i, filepath in enumerate(replay_files, 1):
        try:
            replay_data = replay_manager.load_replay(filepath)
            if not play_replay(renderer, replay_data, fps, i, len(replay_files)):
                break
        except Exception as e:
            print(f"Error loading replay: {e}")
            continue

    renderer.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
