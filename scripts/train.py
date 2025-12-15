#!/usr/bin/env python3
"""
Novelty AI - Training Script

Train AI on any registered game via command line.

Usage:
    python scripts/train.py --game snake              # Train Snake with visualization
    python scripts/train.py --game snake --headless   # Headless (faster, for servers)
    python scripts/train.py --game snake --episodes 5000 --load models/snake/checkpoint.pth
    python scripts/train.py --game snake --json       # Machine-readable output
"""
import sys
import json
import argparse
import ctypes
from pathlib import Path
from contextlib import contextmanager

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.device.device_manager import DeviceManager
from src.utils.config_loader import load_game_config
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Novelty AI - Train AI on games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --game snake              # Train with visualization
  python scripts/train.py --game snake --headless   # Headless (faster)
  python scripts/train.py --game snake --episodes 5000
  python scripts/train.py --game snake --json       # Machine-readable output
"""
    )

    parser.add_argument(
        "-g", "--game",
        type=str,
        required=True,
        metavar="GAME_ID",
        help="Game to train on (e.g., 'snake')"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without visualization (faster training)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to train"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (ignore GPU)"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load model from checkpoint"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: auto-detect)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format (for automation)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only errors and final results)"
    )

    args = parser.parse_args()

    # Auto-detect num_envs if not specified
    if args.num_envs is None:
        args.num_envs = get_default_num_envs()

    return args


def validate_game(game_id: str) -> bool:
    """Check if game is registered."""
    available = GameRegistry.list_games()
    available_ids = [g.id for g in available if hasattr(g, 'id')]
    return game_id in available_ids


def train_headless(config, device_manager, game_id: str, load_path: str = None,
                   num_envs: int = 1, quiet: bool = False):
    """Train without visualization (maximum speed)."""
    from src.visualization.replay_player import ReplayWindow

    if not quiet:
        print("\n[Training] Starting headless training...")
        print("[Training] Replays will open in separate window on new high scores")
        print("[Training] Press Ctrl+C to stop training\n")

    # Initialize replay window for high score playback
    replay_window = ReplayWindow(
        grid_width=config.grid_width,
        grid_height=config.grid_height,
        fps=10,
        max_replays=config.max_replays
    )

    def on_high_score(replay_path, score, episode):
        """Callback when high score achieved."""
        from src.visualization.replay_player import ReplayManager
        replay_manager = ReplayManager(save_dir=config.replay_dir)
        try:
            replay_data = replay_manager.load_replay(replay_path)
            replay_window.queue_replay(
                frames=replay_data.frames,
                score=score,
                episode=episode,
                save=False
            )
        except Exception as e:
            if not quiet:
                print(f"[Replay] Error loading replay: {e}")

    trainer = Trainer(
        config=config,
        device=device_manager.get_device(),
        num_envs=num_envs,
        dashboard=None,
        on_high_score=on_high_score,
        load_path=load_path,
    )

    try:
        high_score = trainer.train()
    except KeyboardInterrupt:
        if not quiet:
            print("\n[Training] Interrupted by user")
        high_score = trainer.high_score
    finally:
        final_path = trainer.save_final_model()
        trainer.close()
        replay_window.stop()

    return {
        "high_score": high_score,
        "model_path": final_path,
        "episodes_completed": trainer.step_count // num_envs if hasattr(trainer, 'step_count') else 0
    }


def train_with_visualization(config, device_manager, game_id: str, load_path: str = None,
                              num_envs: int = 1, quiet: bool = False):
    """Train with real-time visualization dashboard."""
    from src.visualization.dashboard import TrainingDashboard

    if not quiet:
        print("\n[Training] Starting training with visualization...")
        print("[Training] Close the window or press ESC to stop")
        print("[Training] Press H to toggle headless mode\n")

    # Initialize dashboard
    dashboard = TrainingDashboard(
        window_width=1400,
        window_height=900,
        grid_width=config.grid_width,
        grid_height=config.grid_height,
        chart_update_interval=25,
        total_episodes=config.episodes,
        num_envs=num_envs,
    )

    trainer = Trainer(
        config=config,
        device=device_manager.get_device(),
        num_envs=num_envs,
        dashboard=dashboard,
        on_high_score=None,
        load_path=load_path,
    )

    try:
        high_score = trainer.train(render_fps=30)
    except KeyboardInterrupt:
        if not quiet:
            print("\n[Training] Interrupted by user")
        high_score = trainer.high_score
    finally:
        final_path = trainer.save_final_model()
        trainer.close()

    return {
        "high_score": high_score,
        "model_path": final_path,
        "episodes_completed": trainer.step_count // num_envs if hasattr(trainer, 'step_count') else 0
    }


def main():
    """Main entry point."""
    args = parse_args()

    # Validate game
    if not validate_game(args.game):
        available = GameRegistry.list_games()
        available_ids = [g.id for g in available if hasattr(g, 'id')]
        if args.json:
            print(json.dumps({"error": f"Unknown game '{args.game}'", "available": available_ids}))
        else:
            print(f"Error: Unknown game '{args.game}'")
            print(f"Available games: {', '.join(available_ids)}")
        sys.exit(1)

    # Load game-specific configuration
    raw_config = load_game_config(args.game)

    # Override config with command line args
    if args.episodes:
        raw_config.training.episodes = args.episodes

    # Create training config from loaded config
    config = TrainingConfig.from_config(raw_config)

    # Update paths for game-specific directories
    config.checkpoint_dir = f"models/{args.game}"
    config.replay_dir = f"replays/{args.game}"

    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.replay_dir).mkdir(parents=True, exist_ok=True)

    # Initialize device manager
    preferred = None
    if raw_config.device.preferred != "auto":
        preferred = raw_config.device.preferred

    device_manager = DeviceManager(
        preferred=preferred,
        force_cpu=args.cpu or raw_config.device.force_cpu
    )

    # Get game metadata
    metadata = GameRegistry.get_metadata(args.game)
    game_name = metadata.name if metadata else args.game.title()

    if not args.quiet and not args.json:
        print("=" * 60)
        print(f"Novelty AI - {game_name} Training")
        print("=" * 60)
        device_manager.print_device_info()
        print(f"Episodes: {config.episodes}")
        print(f"Visualization: {'Disabled' if args.headless else 'Enabled'}")
        print(f"Parallel environments: {args.num_envs}")
        print("=" * 60)

    # Start training
    with prevent_sleep():
        if args.headless:
            result = train_headless(
                config, device_manager, args.game,
                load_path=args.load, num_envs=args.num_envs, quiet=args.quiet
            )
        else:
            result = train_with_visualization(
                config, device_manager, args.game,
                load_path=args.load, num_envs=args.num_envs, quiet=args.quiet
            )

    # Output results
    if args.json:
        result["game"] = args.game
        result["success"] = True
        print(json.dumps(result))
    elif not args.quiet:
        print(f"\n[Training] Complete! High score: {result['high_score']}")
        print(f"[Training] Model saved to: {result['model_path']}")


if __name__ == "__main__":
    main()
