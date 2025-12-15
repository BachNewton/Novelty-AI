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
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom window title"
    )
    parser.add_argument(
        "--queue",
        type=str,
        default=None,
        help="Queue file for continuous replay mode"
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


def play_replay(renderer, replay_data, fps, replay_num, total_replays, wait_at_end=False):
    """Play a single replay. Returns True to continue, False to quit."""
    clock = pygame.time.Clock()
    frame_idx = 0
    paused = False
    running = True

    print(f"\nPlaying replay {replay_num}/{total_replays}: "
          f"Episode {replay_data.episode}, Score {replay_data.score}")
    print("Controls: SPACE=pause, LEFT/RIGHT=skip, +/-=speed, N=next, ESC=quit")

    # Clear any pending events and let pygame stabilize
    pygame.time.wait(100)
    pygame.event.clear()

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

    # Show end screen if wait_at_end is True
    if wait_at_end:
        info_font = pygame.font.Font(None, 36)
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN):
                        return False

            # Show final frame with "Press any key" message
            renderer.surface.fill((0, 0, 0))
            if replay_data.frames:
                renderer.render(replay_data.frames[-1])

            # Draw completion message
            msg = info_font.render(
                f"Replay Complete! Score: {replay_data.score}",
                True, (100, 255, 100)
            )
            renderer.surface.blit(
                msg,
                (renderer.window_width // 2 - msg.get_width() // 2, 10)
            )

            hint = info_font.render(
                "Press SPACE or ESC to close",
                True, (150, 150, 150)
            )
            renderer.surface.blit(
                hint,
                (renderer.window_width // 2 - hint.get_width() // 2,
                 renderer.window_height - 40)
            )

            pygame.display.flip()
            clock.tick(30)
        return False
    else:
        # Brief pause at end
        pygame.time.wait(1000)
        return True


def run_queue_mode(queue_file: str, config):
    """Run in queue mode - continuously play replays from a queue file."""
    from src.game.renderer import StandaloneRenderer

    queue_path = Path(queue_file)
    replay_manager = ReplayManager(save_dir=config.replay.save_dir)

    # Initialize renderer
    renderer = StandaloneRenderer(
        grid_width=config.game.grid_width,
        grid_height=config.game.grid_height,
        cell_size=30,
        title="High Score Replays"
    )

    clock = pygame.time.Clock()
    fps = 15

    # Clear any pending events
    pygame.time.wait(100)
    pygame.event.clear()

    def read_next_from_queue():
        """Read and remove the first entry from the queue file."""
        if not queue_path.exists():
            return None

        with open(queue_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return None

        # Get first entry
        first_line = lines[0].strip()
        if not first_line:
            # Remove empty line and try again
            with open(queue_path, "w") as f:
                f.writelines(lines[1:])
            return read_next_from_queue()

        # Write remaining entries back
        with open(queue_path, "w") as f:
            f.writelines(lines[1:])

        # Parse entry: path|score|episode
        parts = first_line.split("|")
        if len(parts) >= 3:
            return {"path": parts[0], "score": int(parts[1]), "episode": int(parts[2])}
        return None

    running = True
    replays_played = 0

    while running:
        entry = read_next_from_queue()

        if entry is None:
            # No more replays, exit
            break

        replay_path = entry["path"]
        score = entry["score"]
        episode = entry["episode"]

        # Check if file exists
        if not Path(replay_path).exists():
            print(f"[Replay] File not found: {replay_path}")
            continue

        try:
            replay_data = replay_manager.load_replay(replay_path)
        except Exception as e:
            print(f"[Replay] Error loading {replay_path}: {e}")
            continue

        replays_played += 1

        # Update window title
        pygame.display.set_caption(f"High Score: {score} (Episode {episode})")

        # Play the replay
        frame_idx = 0
        while frame_idx < len(replay_data.frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    elif event.key == pygame.K_SPACE:
                        # Skip to next replay
                        frame_idx = len(replay_data.frames)
                        break
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        fps = min(60, fps + 5)
                    elif event.key == pygame.K_MINUS:
                        fps = max(5, fps - 5)

            if not running:
                break

            if frame_idx >= len(replay_data.frames):
                break

            frame = replay_data.frames[frame_idx]
            assert renderer.surface is not None  # Guaranteed after pygame init
            renderer.surface.fill((0, 0, 0))
            renderer.render(frame)

            # Draw info
            info_font = pygame.font.Font(None, 28)

            title_text = info_font.render(
                f"NEW HIGH SCORE: {score} (Episode {episode})",
                True, (100, 255, 100)
            )
            renderer.surface.blit(title_text, (10, 10))

            score_text = info_font.render(
                f"Score: {frame.get('score', 0)}",
                True, (220, 220, 220)
            )
            renderer.surface.blit(
                score_text,
                (renderer.window_width // 2 - score_text.get_width() // 2,
                 renderer.window_height - 60)
            )

            progress_text = info_font.render(
                f"Frame {frame_idx + 1}/{len(replay_data.frames)} | SPACE=skip, +/-=speed, ESC=close",
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

        # Brief pause between replays
        if running:
            pygame.time.wait(500)

    # Clean up queue file
    if queue_path.exists():
        queue_path.unlink()

    renderer.close()
    print(f"[Replay] Played {replays_played} replay(s)")


def main():
    """Main entry point."""
    import traceback

    args = parse_args()

    # Queue mode for high score replays
    if args.queue:
        try:
            config = load_config()
            run_queue_mode(args.queue, config)
        except Exception as e:
            log_path = Path(__file__).parent.parent / "replay_error.log"
            with open(log_path, "w") as f:
                f.write(f"Error in queue mode: {e}\n")
                f.write(traceback.format_exc())
        return

    # If launched with --title (high score popup), log errors to file since no console
    if args.title:
        try:
            _main_impl(args)
        except Exception as e:
            # Log error to file for debugging
            log_path = Path(__file__).parent.parent / "replay_error.log"
            with open(log_path, "w") as f:
                f.write(f"Error: {e}\n")
                f.write(traceback.format_exc())
            raise
    else:
        _main_impl(args)


def _main_impl(args):
    """Actual main implementation."""
    config = load_config()

    replay_manager = ReplayManager(save_dir=config.replay.save_dir)

    # List mode
    if args.list:
        list_replays(replay_manager)
        return

    # Get replays to play
    if args.replay:
        # Verify the replay file exists
        replay_path = Path(args.replay)
        if not replay_path.exists():
            print(f"Error: Replay file not found: {args.replay}")
            # Log for debugging when no console
            if args.title:
                log_path = Path(__file__).parent.parent / "replay_error.log"
                with open(log_path, "w") as f:
                    f.write(f"Replay file not found: {args.replay}\n")
            return
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
    window_title = args.title if args.title else "Snake AI - Replay Viewer"
    renderer = StandaloneRenderer(
        grid_width=config.game.grid_width,
        grid_height=config.game.grid_height,
        cell_size=30,
        title=window_title
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

    # If single replay with custom title (high score popup), wait at end
    wait_at_end = args.title is not None and len(replay_files) == 1

    # Play each replay
    for i, filepath in enumerate(replay_files, 1):
        try:
            replay_data = replay_manager.load_replay(filepath)
            if not play_replay(renderer, replay_data, fps, i, len(replay_files), wait_at_end):
                break
        except Exception as e:
            print(f"Error loading replay: {e}")
            continue

    renderer.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
