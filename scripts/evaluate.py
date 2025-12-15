#!/usr/bin/env python3
"""
Novelty AI - Model Evaluation Script

Evaluate trained models with statistical metrics.
Useful for AI iteration and benchmarking.

Usage:
    python scripts/evaluate.py --game snake                           # Evaluate latest model
    python scripts/evaluate.py --game snake --model models/snake/model.pth
    python scripts/evaluate.py --game snake --episodes 100 --json     # Machine-readable output
"""
import sys
import json
import argparse
import statistics
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.device.device_manager import DeviceManager
from src.utils.config_loader import load_game_config
from src.games.registry import GameRegistry
from src.algorithms.dqn.agent import DQNAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Novelty AI - Evaluate trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py --game snake
  python scripts/evaluate.py --game snake --model models/snake/model.pth
  python scripts/evaluate.py --game snake --episodes 100 --json
"""
    )

    parser.add_argument(
        "-g", "--game",
        type=str,
        required=True,
        metavar="GAME_ID",
        help="Game to evaluate (e.g., 'snake')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (default: latest model)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate (default: 100)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )

    return parser.parse_args()


def find_latest_model(game_id: str) -> Optional[str]:
    """Find the latest model for a game."""
    models_dir = Path(f"models/{game_id}")

    if not models_dir.exists():
        return None

    # Try final model first
    final = models_dir / "final_model.pth"
    if final.exists():
        return str(final)

    # Then try latest checkpoint
    checkpoints = list(models_dir.glob("model_ep*.pth"))
    if checkpoints:
        return str(sorted(checkpoints)[-1])

    return None


def evaluate_model(game_id: str, model_path: str, episodes: int,
                   device_manager, quiet: bool = False) -> dict:
    """Evaluate a model over multiple episodes."""
    # Load game config
    config = load_game_config(game_id)

    # Create environment
    env = GameRegistry.create_env(
        game_id,
        width=config.game.grid_width,
        height=config.game.grid_height
    )

    # Create and load agent
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        device=device_manager.get_device()
    )
    agent.load(model_path)
    agent.epsilon = 0  # No exploration during evaluation

    scores = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            score = info.get("score", 0)

        scores.append(score)

        if not quiet and (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{episodes}: Score {score}")

    # Calculate statistics
    results = {
        "game": game_id,
        "model_path": model_path,
        "episodes": episodes,
        "scores": {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "min": min(scores),
            "max": max(scores),
        },
        "raw_scores": scores
    }

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Validate game
    available = GameRegistry.list_games()
    available_ids = [g.id for g in available if hasattr(g, 'id')]

    if args.game not in available_ids:
        if args.json:
            print(json.dumps({"error": f"Unknown game '{args.game}'", "available": available_ids}))
        else:
            print(f"Error: Unknown game '{args.game}'")
            print(f"Available games: {', '.join(available_ids)}")
        sys.exit(1)

    # Find model
    model_path = args.model or find_latest_model(args.game)

    if not model_path or not Path(model_path).exists():
        if args.json:
            print(json.dumps({"error": "No model found", "game": args.game}))
        else:
            print(f"Error: No model found for game '{args.game}'")
            print("Train a model first or specify --model path")
        sys.exit(1)

    # Initialize device
    device_manager = DeviceManager(force_cpu=args.cpu)

    # Get game metadata
    metadata = GameRegistry.get_metadata(args.game)
    game_name = metadata.name if metadata else args.game.title()

    if not args.quiet and not args.json:
        print("=" * 60)
        print(f"Novelty AI - {game_name} Model Evaluation")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Episodes: {args.episodes}")
        print("=" * 60)

    # Run evaluation
    results = evaluate_model(
        args.game, model_path, args.episodes,
        device_manager, quiet=args.quiet or args.json
    )

    # Output results
    if args.json:
        # Remove raw scores for cleaner JSON (can be large)
        output = {k: v for k, v in results.items() if k != 'raw_scores'}
        output["success"] = True
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Mean Score:   {results['scores']['mean']:.2f}")
        print(f"Median Score: {results['scores']['median']:.2f}")
        print(f"Std Dev:      {results['scores']['stdev']:.2f}")
        print(f"Min Score:    {results['scores']['min']}")
        print(f"Max Score:    {results['scores']['max']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
