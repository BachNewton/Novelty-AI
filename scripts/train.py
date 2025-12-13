#!/usr/bin/env python3
"""
Training Script - Main entry point for training the Snake AI.

Usage:
    python scripts/train.py                    # Train with visualization
    python scripts/train.py --headless         # Train without visualization (faster)
    python scripts/train.py --episodes 5000    # Custom episode count
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.game.snake_env import SnakeEnv
from src.ai.dqn_agent import DQNAgent
from src.device.device_manager import DeviceManager
from src.utils.config_loader import load_config
from src.visualization.replay_player import ReplayWindow


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

    return parser.parse_args()


def train_headless(config, device_manager, start_episode: int = 0, load_path: str = None):
    """
    Train without visualization (maximum speed).

    Opens a replay window when new high scores are achieved.
    """
    print("\n[Training] Starting headless training...")
    print("[Training] Replays will open in a separate window when new high scores are achieved")
    print("[Training] Press Ctrl+C to stop training\n")

    # Initialize environment and agent
    env = SnakeEnv(config.game.grid_width, config.game.grid_height)

    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        device=device_manager.get_device(),
        config={
            "gamma": config.training.gamma,
            "epsilon_start": config.training.epsilon_start,
            "epsilon_min": config.training.epsilon_min,
            "epsilon_decay": config.training.epsilon_decay,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "buffer_size": config.training.buffer_size,
            "target_update_freq": config.training.target_update_freq,
        }
    )

    # Load checkpoint if specified
    if load_path:
        print(f"[Training] Loading model from {load_path}")
        agent.load(load_path)

    # Initialize replay window (will open when needed)
    replay_window = ReplayWindow(
        grid_width=config.game.grid_width,
        grid_height=config.game.grid_height,
        fps=config.replay.playback_fps,
        max_replays=config.replay.max_replays
    )

    # Training stats
    high_score = 0
    recent_scores = []
    total_episodes = config.training.episodes
    step_count = 0

    try:
        for episode in range(start_episode + 1, total_episodes + 1):
            # Reset environment with recording
            state = env.reset(record=True)
            total_reward = 0
            done = False

            while not done:
                # Select and execute action
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)

                # Store transition
                agent.store_transition(state, action, reward, next_state, done)

                # Train every 4 steps
                step_count += 1
                if step_count % 4 == 0:
                    agent.train_step()

                state = next_state
                total_reward += reward

            # Episode complete
            score = info["score"]
            recent_scores.append(score)
            if len(recent_scores) > 100:
                recent_scores.pop(0)

            # Check for new high score
            if score > high_score:
                high_score = score
                print(f"\n[NEW HIGH SCORE] Episode {episode}: Score {score}!")

                # Queue replay for playback
                if config.replay.enabled:
                    replay_frames = env.get_replay()
                    replay_window.queue_replay(
                        frames=replay_frames,
                        score=score,
                        episode=episode,
                        save=True
                    )

            # Decay epsilon
            agent.decay_epsilon()

            # Progress output
            if episode % 10 == 0:
                avg_score = sum(recent_scores) / len(recent_scores)
                print(
                    f"Episode {episode}/{total_episodes} | "
                    f"Score: {score} | Avg: {avg_score:.1f} | "
                    f"High: {high_score} | Epsilon: {agent.epsilon:.4f}"
                )

            # Save checkpoint
            if episode % config.training.save_interval == 0:
                checkpoint_path = f"{config.training.checkpoint_dir}/model_ep{episode}.pth"
                agent.save(checkpoint_path)
                print(f"[Checkpoint] Saved model at episode {episode}")

    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user")

    finally:
        # Save final model
        final_path = f"{config.training.checkpoint_dir}/final_model.pth"
        agent.save(final_path)
        print(f"[Training] Saved final model to {final_path}")

        # Clean up replay window
        replay_window.stop()

    print(f"\n[Training] Complete! High score: {high_score}")


def train_with_visualization(config, device_manager, start_episode: int = 0, load_path: str = None):
    """Train with real-time visualization dashboard."""
    import pygame

    from src.visualization.dashboard import TrainingDashboard

    print("\n[Training] Starting training with visualization...")
    print("[Training] Close the window or press ESC to stop\n")

    # Initialize environment and agent
    env = SnakeEnv(config.game.grid_width, config.game.grid_height)

    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        device=device_manager.get_device(),
        config={
            "gamma": config.training.gamma,
            "epsilon_start": config.training.epsilon_start,
            "epsilon_min": config.training.epsilon_min,
            "epsilon_decay": config.training.epsilon_decay,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "buffer_size": config.training.buffer_size,
            "target_update_freq": config.training.target_update_freq,
        }
    )

    # Load checkpoint if specified
    if load_path:
        print(f"[Training] Loading model from {load_path}")
        agent.load(load_path)

    # Initialize visualization
    dashboard = TrainingDashboard(
        window_width=config.visualization.window_width,
        window_height=config.visualization.window_height,
        grid_width=config.game.grid_width,
        grid_height=config.game.grid_height,
        chart_update_interval=config.visualization.chart_update_interval,
    )

    # Training loop
    total_episodes = config.training.episodes
    running = True

    step_count = 0

    try:
        for episode in range(start_episode + 1, total_episodes + 1):
            if not running:
                break

            # Reset environment with recording
            state = env.reset(record=True)
            done = False
            loss = None

            while not done and running:
                # Select and execute action
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)

                # Store transition
                agent.store_transition(state, action, reward, next_state, done)

                # Train every 4 steps (reduces CPU load, keeps window responsive)
                step_count += 1
                if step_count % 4 == 0:
                    loss = agent.train_step()

                state = next_state

                # Update visualization
                game_state = env.get_game_state()
                running = dashboard.update(
                    game_state=game_state,
                    episode=episode,
                    score=info["score"],
                    epsilon=agent.epsilon,
                    loss=loss,
                    fps=config.visualization.render_fps
                )

            if not running:
                break

            # Episode complete
            score = info["score"]

            # Decay epsilon
            agent.decay_epsilon()

            # Console output
            if episode % 10 == 0:
                avg_score = dashboard.metrics.avg_scores[-1] if dashboard.metrics.avg_scores else 0
                print(
                    f"Episode {episode}/{total_episodes} | "
                    f"Score: {score} | Avg: {avg_score:.1f} | "
                    f"High: {dashboard.metrics.high_score} | Epsilon: {agent.epsilon:.4f}"
                )

            # Save checkpoint
            if episode % config.training.save_interval == 0:
                checkpoint_path = f"{config.training.checkpoint_dir}/model_ep{episode}.pth"
                agent.save(checkpoint_path)
                print(f"[Checkpoint] Saved model at episode {episode}")

    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user")

    finally:
        # Save final model
        final_path = f"{config.training.checkpoint_dir}/final_model.pth"
        agent.save(final_path)
        print(f"[Training] Saved final model to {final_path}")

        # Clean up
        dashboard.close()

    print(f"\n[Training] Complete! High score: {dashboard.metrics.high_score}")


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line args
    if args.episodes:
        config.training.episodes = args.episodes

    # Initialize device manager
    preferred = None
    if config.device.preferred != "auto":
        preferred = config.device.preferred

    device_manager = DeviceManager(
        preferred=preferred,
        force_cpu=args.cpu or config.device.force_cpu
    )

    print("=" * 60)
    print("Snake AI Training")
    print("=" * 60)
    device_manager.print_device_info()
    print(f"Episodes: {config.training.episodes}")
    print(f"Visualization: {'Disabled' if args.headless else 'Enabled'}")
    print("=" * 60)

    # Ensure directories exist
    Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.replay.save_dir).mkdir(parents=True, exist_ok=True)

    # Start training
    if args.headless:
        train_headless(config, device_manager, load_path=args.load)
    else:
        train_with_visualization(config, device_manager, load_path=args.load)


if __name__ == "__main__":
    main()
