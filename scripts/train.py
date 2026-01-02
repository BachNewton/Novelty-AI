#!/usr/bin/env python3
"""
Novelty AI - Training Script

Train AI on any registered game via command line.
Uses async training architecture for maximum speed.

Usage:
    python scripts/train.py --game snake              # Train with terminal UI (default)
    python scripts/train.py --game snake --visual     # Train with pygame visualization
    python scripts/train.py --game snake --headless   # No UI, basic logging only
    python scripts/train.py --game snake --episodes 5000 --load models/snake/checkpoint.pth
    python scripts/train.py --game snake --json       # Machine-readable output
"""
import sys
import json
import time
import argparse
import ctypes
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
from datetime import datetime
from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.device.device_manager import DeviceManager
from src.utils.config_loader import load_game_config
from src.training import TrainingConfig, get_default_num_envs
from src.training.vec_env import VectorizedEnv
from src.training.async_trainer import AsyncTrainer
from src.games.registry import GameRegistry
from src.algorithms.dqn.agent import DQNAgent


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
        description="Novelty AI - Train AI on games (async architecture for maximum speed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --game snake              # Train with terminal UI
  python scripts/train.py --game snake --headless   # Headless (faster)
  python scripts/train.py --game snake --episodes 5000
  python scripts/train.py --game snake --resume     # Resume from checkpoint
  python scripts/train.py --game snake --json       # Machine-readable output
"""
    )

    parser.add_argument(
        "-g", "--game",
        type=str,
        required=True,
        metavar="GAME_ID",
        help="Game to train on (e.g., 'snake', 'space_invaders')"
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Show pygame visualization window (slower)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No terminal UI, just basic logging (for scripts/automation)"
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
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint (restores episode count and epsilon)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: CPU cores)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size (default: 128)"
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


def find_latest_checkpoint(game_id: str) -> Optional[str]:
    """Find the latest checkpoint for a game.

    Prefers final_model.pth if it exists, otherwise uses the highest episode checkpoint.
    """
    import re

    checkpoint_dir = Path(f"models/{game_id}")
    if not checkpoint_dir.exists():
        return None

    # Prefer final_model.pth - it's always the most recent completed state
    final = checkpoint_dir / "final_model.pth"
    if final.exists():
        return str(final)

    # Fall back to highest episode checkpoint
    checkpoints = list(checkpoint_dir.glob("model_ep*.pth"))
    if not checkpoints:
        return None

    def get_episode_num(path: Path) -> int:
        match = re.search(r'model_ep(\d+)\.pth', path.name)
        return int(match.group(1)) if match else 0

    latest_checkpoint = max(checkpoints, key=get_episode_num)
    return str(latest_checkpoint)


def get_checkpoint_episode(checkpoint_path: str) -> int:
    """Get the episode number from a checkpoint file without fully loading it."""
    import torch
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        return checkpoint.get("episode", 0)
    except Exception:
        return 0


def create_agent_and_env(config, device, num_envs: int, batch_size: int, load_path: Optional[str] = None):
    """Create vectorized environment and agent with fast buffer."""
    reward_config = config.get_reward_config()

    vec_env = VectorizedEnv(
        game_id=config.game_id,
        num_envs=num_envs,
        env_config={
            'width': config.grid_width,
            'height': config.grid_height,
            'reward_config': reward_config,
        }
    )

    agent_config = config.get_agent_config()
    agent_config['use_fast_buffer'] = True
    agent_config['batch_size'] = batch_size

    agent = DQNAgent(
        state_size=vec_env.state_size,
        action_size=vec_env.action_size,
        device=device,
        config=agent_config,
    )

    start_episode = 0
    if load_path and Path(load_path).exists():
        start_episode = agent.load(load_path)

    return vec_env, agent, start_episode


def train_headless(config, device_manager, game_id: str, load_path: Optional[str] = None,
                   num_envs: int = 1, batch_size: int = 128, quiet: bool = False):
    """Train without visualization using async training."""

    if not quiet:
        print("\n[Training] Starting headless training (async)...")
        print("[Training] Press Ctrl+C to stop training\n")

    vec_env, agent, start_episode = create_agent_and_env(
        config, device_manager.get_device(), num_envs, batch_size, load_path
    )

    if not quiet and load_path:
        print(f"[Training] Resuming from episode {start_episode}")

    # Initialize async trainer
    async_trainer = AsyncTrainer(agent, train_interval=0, trains_per_step=4)

    # Initialize TensorBoard writer
    log_dir = f"runs/{game_id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    if not quiet:
        print(f"[TensorBoard] Logging to {log_dir}")

    # Initialize state
    states = np.stack([env.reset(record=False) for env in vec_env.envs])
    episode_count = start_episode
    total_steps = 0
    high_score = 0
    recent_scores_50: deque = deque(maxlen=50)

    start_time = time.perf_counter()
    async_trainer.start()

    try:
        while episode_count < config.episodes:
            actions = agent.select_actions_batch(states, training=True)
            next_states, rewards, dones, infos = vec_env.step(actions)

            agent.store_transitions_batch(
                states, actions, rewards, next_states, dones.astype(np.float32)
            )

            total_steps += num_envs

            for i in range(num_envs):
                if dones[i]:
                    episode_count += 1
                    score = infos[i].get("score", 0)

                    recent_scores_50.append(score)

                    if score > high_score:
                        high_score = score
                        if not quiet:
                            print(f"\n[NEW HIGH SCORE] Episode {episode_count}: Score {score}!")

                    agent.on_episode_end()
                    next_states[i] = vec_env.envs[i].reset(record=False)

                    # Calculate rolling stats
                    avg_50 = sum(recent_scores_50) / len(recent_scores_50) if recent_scores_50 else 0
                    min_50 = min(recent_scores_50) if recent_scores_50 else 0
                    max_50 = max(recent_scores_50) if recent_scores_50 else 0
                    elapsed = time.perf_counter() - start_time
                    eps_per_sec = episode_count / elapsed if elapsed > 0 else 0
                    loss = async_trainer.last_loss or 0

                    # Log to TensorBoard
                    writer.add_scalar('Score/episode', score, episode_count)
                    writer.add_scalar('Score/avg_50', avg_50, episode_count)
                    writer.add_scalar('Score/min_50', min_50, episode_count)
                    writer.add_scalar('Score/max_50', max_50, episode_count)
                    writer.add_scalar('Score/high', high_score, episode_count)
                    writer.add_scalar('Training/epsilon', agent.epsilon, episode_count)
                    writer.add_scalar('Training/loss', loss, episode_count)
                    writer.add_scalar('Performance/episodes_per_sec', eps_per_sec, episode_count)

                    # Periodic console logging
                    if not quiet and episode_count % 10 == 0:
                        print(
                            f"Episode {episode_count}/{config.episodes} | "
                            f"Score: {score} | Avg: {avg_50:.1f} | "
                            f"High: {high_score} | Epsilon: {agent.epsilon:.4f}"
                        )

                    # Checkpoint saving
                    if episode_count % config.save_interval == 0:
                        checkpoint_path = f"{config.checkpoint_dir}/model_ep{episode_count}.pth"
                        agent.save(checkpoint_path, episode=episode_count)
                        if not quiet:
                            print(f"[Checkpoint] Saved at episode {episode_count}")

                    if episode_count >= config.episodes:
                        break

            states = next_states

    except KeyboardInterrupt:
        if not quiet:
            print("\n[Training] Interrupted by user")

    finally:
        async_trainer.stop()
        writer.close()
        final_path = f"{config.checkpoint_dir}/final_model.pth"
        agent.save(final_path, episode=episode_count)
        vec_env.close()

    return {
        "high_score": high_score,
        "model_path": final_path,
        "episodes_completed": episode_count - start_episode
    }


def train_with_visualization(config, device_manager, game_id: str, load_path: Optional[str] = None,
                              num_envs: int = 1, batch_size: int = 128, quiet: bool = False):
    """Train with real-time visualization dashboard using async training."""
    from src.visualization.dashboard import TrainingDashboard
    from src.training import Trainer

    if not quiet:
        print("\n[Training] Starting training with visualization...")
        print("[Training] Close the window or press ESC to stop")
        print("[Training] Press H to toggle headless mode\n")

    # Use the original Trainer for visual mode (it handles dashboard updates)
    dashboard = TrainingDashboard(
        window_width=1400,
        window_height=900,
        grid_width=config.grid_width,
        grid_height=config.grid_height,
        chart_update_interval=25,
        total_episodes=config.episodes,
        num_envs=num_envs,
        game_id=game_id,
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


def train_with_terminal_ui(config, device_manager, game_id: str, game_name: str,
                           load_path: Optional[str] = None, num_envs: int = 1,
                           batch_size: int = 128):
    """Train with rich terminal UI using async training."""
    from src.visualization.terminal_display import TerminalTrainingDisplay

    vec_env, agent, start_episode = create_agent_and_env(
        config, device_manager.get_device(), num_envs, batch_size, load_path
    )

    display = TerminalTrainingDisplay(
        total_episodes=config.episodes,
        game_name=game_name,
        num_envs=num_envs,
    )

    # Initialize async trainer
    async_trainer = AsyncTrainer(agent, train_interval=0, trains_per_step=4)

    # Initialize TensorBoard writer
    log_dir = f"runs/{game_id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    # Initialize state
    states = np.stack([env.reset(record=False) for env in vec_env.envs])
    episode_count = start_episode
    total_steps = 0
    high_score = 0
    start_time = time.perf_counter()

    display.start()
    display.add_message(f"[dim]TensorBoard: {log_dir}[/]")
    async_trainer.start()

    try:
        while episode_count < config.episodes:
            actions = agent.select_actions_batch(states, training=True)
            next_states, rewards, dones, infos = vec_env.step(actions)

            agent.store_transitions_batch(
                states, actions, rewards, next_states, dones.astype(np.float32)
            )

            total_steps += num_envs

            for i in range(num_envs):
                if dones[i]:
                    episode_count += 1
                    score = infos[i].get("score", 0)

                    # Update display
                    loss = async_trainer.last_loss or 0.0
                    display.update(
                        episode=episode_count,
                        score=score,
                        epsilon=agent.epsilon,
                        loss=loss,
                        steps=total_steps,
                        env_index=i,
                    )
                    display.record_episode(score)

                    if score > high_score:
                        high_score = score

                    # Calculate rolling stats from display's deque
                    avg_50 = sum(display.recent_scores_50) / len(display.recent_scores_50) if display.recent_scores_50 else 0
                    min_50 = min(display.recent_scores_50) if display.recent_scores_50 else 0
                    max_50 = max(display.recent_scores_50) if display.recent_scores_50 else 0
                    elapsed = time.perf_counter() - start_time
                    eps_per_sec = episode_count / elapsed if elapsed > 0 else 0

                    # Log to TensorBoard
                    writer.add_scalar('Score/episode', score, episode_count)
                    writer.add_scalar('Score/avg_50', avg_50, episode_count)
                    writer.add_scalar('Score/min_50', min_50, episode_count)
                    writer.add_scalar('Score/max_50', max_50, episode_count)
                    writer.add_scalar('Score/high', high_score, episode_count)
                    writer.add_scalar('Training/epsilon', agent.epsilon, episode_count)
                    writer.add_scalar('Training/loss', loss, episode_count)
                    writer.add_scalar('Performance/episodes_per_sec', eps_per_sec, episode_count)

                    agent.on_episode_end()
                    next_states[i] = vec_env.envs[i].reset(record=False)

                    # Checkpoint saving
                    if episode_count % config.save_interval == 0:
                        checkpoint_path = f"{config.checkpoint_dir}/model_ep{episode_count}.pth"
                        agent.save(checkpoint_path, episode=episode_count)
                        display.add_message(f"[green]Checkpoint saved at episode {episode_count}[/]")

                    if episode_count >= config.episodes:
                        break

            states = next_states

    except KeyboardInterrupt:
        display.add_message("[yellow]Training interrupted by user[/]")

    finally:
        async_trainer.stop()
        writer.close()
        display.stop()
        final_path = f"{config.checkpoint_dir}/final_model.pth"
        agent.save(final_path, episode=episode_count)
        vec_env.close()

    return {
        "high_score": high_score,
        "model_path": final_path,
        "episodes_completed": episode_count - start_episode
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

    # Set game ID and paths for game-specific directories
    config.game_id = args.game
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

    # Handle --resume flag
    load_path = args.load
    if args.resume:
        load_path = find_latest_checkpoint(args.game)
        if load_path:
            # Get episode count from checkpoint and add requested episodes
            start_episode = get_checkpoint_episode(load_path)
            requested_episodes = config.episodes  # The episodes requested for this run
            config.episodes = start_episode + requested_episodes  # New total
            if not args.quiet and not args.json:
                print(f"[Resume] Found checkpoint: {load_path}")
                print(f"[Resume] Starting from episode {start_episode}, training {requested_episodes} more (total: {config.episodes})")
        else:
            if args.json:
                print(json.dumps({"error": "No checkpoint found to resume from"}))
            else:
                print("Error: No checkpoint found to resume from")
                print(f"Train first or specify --load with a checkpoint path")
            sys.exit(1)

    # Determine UI mode
    use_visual = args.visual
    use_terminal_ui = not args.visual and not args.headless and not args.quiet and not args.json

    if not args.quiet and not args.json and not use_terminal_ui:
        print("=" * 60)
        print(f"Novelty AI - {game_name} Training (Async)")
        print("=" * 60)
        device_manager.print_device_info()
        print(f"Episodes: {config.episodes}")
        mode = "Visual" if use_visual else "Headless"
        print(f"Mode: {mode}")
        print(f"Parallel environments: {args.num_envs}")
        print(f"Batch size: {args.batch_size}")
        print("=" * 60)

    # Start training
    with prevent_sleep():
        if use_visual:
            result = train_with_visualization(
                config, device_manager, args.game,
                load_path=load_path, num_envs=args.num_envs,
                batch_size=args.batch_size, quiet=args.quiet,
            )
        elif use_terminal_ui:
            result = train_with_terminal_ui(
                config, device_manager, args.game, game_name,
                load_path=load_path, num_envs=args.num_envs,
                batch_size=args.batch_size,
            )
        else:
            result = train_headless(
                config, device_manager, args.game,
                load_path=load_path, num_envs=args.num_envs,
                batch_size=args.batch_size, quiet=args.quiet,
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
