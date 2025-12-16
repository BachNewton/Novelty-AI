#!/usr/bin/env python3
"""
Training Speed Benchmark - Outputs JSON metrics for automated iteration.

Runs training for a fixed number of episodes and outputs performance metrics
in JSON format. Used by Claude to iterate on training speed optimizations.

Usage:
    python scripts/benchmark_training.py --game tetris --episodes 200 --num-envs 8

Output (JSON only):
    {"eps_per_sec": 45.2, "steps_per_sec": 1200, "episodes": 200, ...}
"""
import sys
import os
import json
import argparse
import time
import io
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def benchmark(game_id: str, episodes: int, num_envs: int, device: str) -> dict:
    """Run training and return performance metrics."""
    # Suppress all output during import and training
    from src.device.device_manager import DeviceManager
    from src.utils.config_loader import load_game_config
    from src.training import Trainer, TrainingConfig, get_default_num_envs

    # Load config
    game_config = load_game_config(game_id)
    training_config = TrainingConfig.from_config(game_config)
    training_config.episodes = episodes

    # Disable replay saving for benchmark (reduces I/O overhead)
    training_config.replay_enabled = False

    # Setup device
    device_manager = DeviceManager(force_cpu=(device == "cpu"))
    device_info = device_manager.detect_device()

    # Use default num_envs if not specified
    if num_envs is None:
        num_envs = get_default_num_envs()

    # Create trainer (no dashboard for speed)
    trainer = Trainer(
        config=training_config,
        device=device_info.device,
        game_id=game_id,
        num_envs=num_envs,
        dashboard=None,
    )

    # Run training with timing
    start_time = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start_time

    # Calculate metrics
    eps_per_sec = episodes / elapsed
    steps_per_sec = trainer.step_count / elapsed

    trainer.close()

    return {
        "eps_per_sec": round(eps_per_sec, 2),
        "steps_per_sec": round(steps_per_sec, 1),
        "episodes": episodes,
        "elapsed_sec": round(elapsed, 2),
        "num_envs": num_envs,
        "total_steps": trainer.step_count,
        "game": game_id,
        "device": str(device_info.device),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark training speed - outputs JSON metrics"
    )
    parser.add_argument(
        "-g", "--game",
        type=str,
        default="tetris",
        help="Game to benchmark (default: tetris)"
    )
    parser.add_argument(
        "-e", "--episodes",
        type=int,
        default=200,
        help="Number of episodes to run (default: 200)"
    )
    parser.add_argument(
        "-n", "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: auto)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use (default: auto)"
    )
    args = parser.parse_args()

    # Suppress ALL output except our final JSON
    # Redirect stdout/stderr to suppress pygame init messages, etc.
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        # Suppress output during benchmark
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        # Also suppress pygame welcome message
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

        result = benchmark(
            game_id=args.game,
            episodes=args.episodes,
            num_envs=args.num_envs,
            device=args.device if args.device != "auto" else None
        )
    finally:
        # Restore output
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    # Output ONLY the JSON result
    print(json.dumps(result))


if __name__ == "__main__":
    main()
