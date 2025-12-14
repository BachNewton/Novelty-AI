#!/usr/bin/env python3
"""
Training Script - Main entry point for training the Snake AI.

Usage:
    python scripts/train.py                    # Train with visualization
    python scripts/train.py --headless         # Train without visualization (faster)
    python scripts/train.py --episodes 5000    # Custom episode count
"""
import sys
import argparse
import ctypes
from pathlib import Path
from contextlib import contextmanager

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.device.device_manager import DeviceManager
from src.utils.config_loader import load_config
from src.training import Trainer, TrainingConfig, get_default_num_envs


@contextmanager
def prevent_sleep():
    """Prevent Windows from sleeping while training."""
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        print("[System] Sleep prevention enabled")
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("[System] Sleep prevention disabled")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Snake AI")

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
        "--config",
        type=str,
        default=None,
        help="Path to config file"
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
        help="Number of parallel environments (default: auto-detect based on CPU cores)"
    )

    args = parser.parse_args()

    # Auto-detect num_envs if not specified
    if args.num_envs is None:
        args.num_envs = get_default_num_envs()

    return args


def train_headless(config, device_manager, load_path: str = None, num_envs: int = 1):
    """Train without visualization (maximum speed)."""
    from src.visualization.replay_player import ReplayWindow

    print("\n[Training] Starting headless training...")
    print("[Training] Replays will open in a separate window when new high scores are achieved")
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
                save=False  # Already saved by trainer
            )
        except Exception as e:
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
        print("\n[Training] Interrupted by user")
        high_score = trainer.high_score
    finally:
        trainer.save_final_model()
        trainer.close()
        replay_window.stop()

    print(f"\n[Training] Complete! High score: {high_score}")


def train_with_visualization(config, device_manager, load_path: str = None, num_envs: int = 1):
    """Train with real-time visualization dashboard."""
    from src.visualization.dashboard import TrainingDashboard

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
        on_high_score=None,  # Dashboard handles high score display
        load_path=load_path,
    )

    try:
        high_score = trainer.train(render_fps=30)
    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user")
        high_score = trainer.high_score
    finally:
        trainer.save_final_model()
        trainer.close()

    print(f"\n[Training] Complete! High score: {high_score}")


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    raw_config = load_config(args.config)

    # Override config with command line args
    if args.episodes:
        raw_config.training.episodes = args.episodes

    # Create training config from loaded config
    config = TrainingConfig.from_config(raw_config)

    # Initialize device manager
    preferred = None
    if raw_config.device.preferred != "auto":
        preferred = raw_config.device.preferred

    device_manager = DeviceManager(
        preferred=preferred,
        force_cpu=args.cpu or raw_config.device.force_cpu
    )

    print("=" * 60)
    print("Snake AI Training")
    print("=" * 60)
    device_manager.print_device_info()
    print(f"Episodes: {config.episodes}")
    print(f"Visualization: {'Disabled' if args.headless else 'Enabled'}")
    print(f"Parallel environments: {args.num_envs}")
    print("=" * 60)

    # Start training
    with prevent_sleep():
        if args.headless:
            train_headless(config, device_manager, load_path=args.load, num_envs=args.num_envs)
        else:
            train_with_visualization(config, device_manager, load_path=args.load, num_envs=args.num_envs)


if __name__ == "__main__":
    main()
