"""
Shared Training Logic - Core training loop used by both CLI and UI.

This module provides a unified training implementation that can be used
with or without visualization, supporting parallel environments.
"""
import time
import multiprocessing
from pathlib import Path
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass

import numpy as np

from ..game.snake_env import SnakeEnv
from ..game.vec_env import VectorizedSnakeEnv
from ..ai.dqn_agent import DQNAgent
from ..visualization.replay_player import ReplayManager


def get_default_num_envs() -> int:
    """Get default number of parallel environments based on CPU cores."""
    cpu_count = multiprocessing.cpu_count()
    return max(1, cpu_count)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Grid settings
    grid_width: int = 20
    grid_height: int = 20

    # Training params (defaults match proven config.yaml)
    episodes: int = 10000
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    learning_rate: float = 0.001
    batch_size: int = 64
    buffer_size: int = 100000
    target_update_freq: int = 100

    # Algorithm options
    use_double_dqn: bool = True

    # Rewards (defaults disabled - see EXPERIMENTS.md for why)
    reward_food: float = 10.0
    reward_death: float = -10.0
    reward_step_penalty: float = -0.01
    reward_approach_food: float = 0.0
    reward_retreat_food: float = 0.0
    reward_length_bonus_factor: float = 0.0

    # Saving
    save_interval: int = 100
    checkpoint_dir: str = "models"

    # Replays
    replay_enabled: bool = True
    replay_dir: str = "replays"
    max_replays: int = 10

    @classmethod
    def from_config(cls, config) -> "TrainingConfig":
        """Create TrainingConfig from a loaded config object."""
        return cls(
            grid_width=config.game.grid_width,
            grid_height=config.game.grid_height,
            episodes=config.training.episodes,
            gamma=config.training.gamma,
            epsilon_start=config.training.epsilon_start,
            epsilon_min=config.training.epsilon_min,
            epsilon_decay=config.training.epsilon_decay,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            buffer_size=config.training.buffer_size,
            target_update_freq=config.training.target_update_freq,
            use_double_dqn=getattr(config.training, 'use_double_dqn', True),
            reward_food=getattr(config.rewards, 'food', 10.0),
            reward_death=getattr(config.rewards, 'death', -10.0),
            reward_step_penalty=getattr(config.rewards, 'step_penalty', -0.01),
            reward_approach_food=getattr(config.rewards, 'approach_food', 0.0),
            reward_retreat_food=getattr(config.rewards, 'retreat_food', 0.0),
            reward_length_bonus_factor=getattr(config.rewards, 'length_bonus_factor', 0.0),
            save_interval=config.training.save_interval,
            checkpoint_dir=config.training.checkpoint_dir,
            replay_enabled=config.replay.enabled,
            replay_dir=config.replay.save_dir,
            max_replays=config.replay.max_replays,
        )

    def get_reward_config(self) -> Dict[str, float]:
        """Get reward configuration dictionary."""
        return {
            "food": self.reward_food,
            "death": self.reward_death,
            "step_penalty": self.reward_step_penalty,
            "approach_food": self.reward_approach_food,
            "retreat_food": self.reward_retreat_food,
            "length_bonus_factor": self.reward_length_bonus_factor,
        }

    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration dictionary."""
        return {
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "target_update_freq": self.target_update_freq,
            "use_double_dqn": self.use_double_dqn,
        }


class Trainer:
    """
    Unified trainer for Snake AI.

    Supports both headless and visualization modes, with parallel environments.
    """

    def __init__(
        self,
        config: TrainingConfig,
        device,
        num_envs: Optional[int] = None,
        dashboard=None,
        on_high_score: Optional[Callable[[str, int, int], None]] = None,
        start_episode: int = 0,
        load_path: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            device: PyTorch device for training
            num_envs: Number of parallel environments (default: auto-detect)
            dashboard: Optional TrainingDashboard for visualization
            on_high_score: Callback(replay_path, score, episode) when high score achieved
            start_episode: Episode to start from (for resuming)
            load_path: Path to load model from
        """
        self.config = config
        self.device = device
        self.num_envs = num_envs or get_default_num_envs()
        self.dashboard = dashboard
        self.on_high_score = on_high_score
        self.start_episode = start_episode

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if config.replay_enabled:
            Path(config.replay_dir).mkdir(parents=True, exist_ok=True)

        # Initialize environments
        reward_config = config.get_reward_config()

        self.vec_env = VectorizedSnakeEnv(
            num_envs=self.num_envs,
            width=config.grid_width,
            height=config.grid_height,
            reward_config=reward_config,
        )

        # Display environment for visualization
        self.display_env = SnakeEnv(
            config.grid_width,
            config.grid_height,
            reward_config=reward_config,
        )

        # Initialize agent
        self.agent = DQNAgent(
            state_size=self.vec_env.state_size,
            action_size=self.vec_env.action_size,
            device=device,
            config=config.get_agent_config(),
        )

        # Load model if specified
        if load_path and Path(load_path).exists():
            print(f"[Training] Loading model from {load_path}")
            self.agent.load(load_path)

        # Replay manager
        self.replay_manager = None
        if config.replay_enabled:
            self.replay_manager = ReplayManager(
                save_dir=config.replay_dir,
                max_replays=config.max_replays,
            )

        # Training state
        self.running = True
        self.step_count = 0
        self.high_score = 0
        self.recent_scores = []

    def train(self, render_fps: int = 30) -> int:
        """
        Run the training loop.

        Args:
            render_fps: FPS for visualization mode

        Returns:
            High score achieved during training
        """
        total_episodes = self.config.episodes
        num_envs = self.num_envs

        # Initialize all environments with recording
        states = np.stack([e.reset(record=True) for e in self.vec_env.envs])
        display_state = self.display_env.reset(record=True)

        episode_count = self.start_episode
        loss = None
        episode_rewards = np.zeros(num_envs)

        print(f"[Training] Starting from episode {episode_count}")
        print(f"[Training] Using {num_envs} parallel environments")

        while episode_count < total_episodes and self.running:
            # Select actions for all environments (batched for GPU efficiency)
            actions = self.agent.select_actions_batch(states, training=True)

            # Step all environments
            next_states, rewards, dones, infos = self.vec_env.step(actions)

            # Store transitions
            for i in range(num_envs):
                self.agent.store_transition(
                    states[i], actions[i], rewards[i],
                    next_states[i], dones[i]
                )
                episode_rewards[i] += rewards[i]

            self.step_count += num_envs
            if self.step_count % 4 == 0:
                loss = self.agent.train_step()

            states = next_states

            # Handle completed episodes
            for i in range(num_envs):
                if dones[i]:
                    episode_count += 1
                    score = infos[i]["score"]

                    # Record metrics
                    if self.dashboard:
                        self.dashboard.record_episode(
                            episode_count, score, self.agent.epsilon, loss
                        )

                    self.recent_scores.append(score)
                    if len(self.recent_scores) > 100:
                        self.recent_scores.pop(0)

                    # High score handling
                    if score > self.high_score:
                        self.high_score = score
                        print(f"\n[NEW HIGH SCORE] Episode {episode_count}: Score {score}!")

                        if self.replay_manager:
                            replay_frames = self.vec_env.envs[i].get_replay()
                            replay_path = self.replay_manager.save_replay(
                                replay_frames, score, episode_count
                            )
                            if replay_path and self.on_high_score:
                                self.on_high_score(replay_path, score, episode_count)

                    self.agent.decay_epsilon()
                    episode_rewards[i] = 0

                    # Reset environment with recording
                    states[i] = self.vec_env.envs[i].reset(record=True)

                    # Periodic logging
                    if episode_count % 10 == 0:
                        avg_score = sum(self.recent_scores) / len(self.recent_scores) if self.recent_scores else 0
                        print(
                            f"Episode {episode_count}/{total_episodes} | "
                            f"Score: {score} | Avg: {avg_score:.1f} | "
                            f"High: {self.high_score} | Epsilon: {self.agent.epsilon:.4f}"
                        )

                    # Checkpoint saving
                    if episode_count % self.config.save_interval == 0:
                        checkpoint_path = f"{self.config.checkpoint_dir}/model_ep{episode_count}.pth"
                        self.agent.save(checkpoint_path)
                        print(f"[Checkpoint] Saved model at episode {episode_count}")

                    if episode_count >= total_episodes:
                        break

            # Update visualization
            if self.dashboard:
                if self.dashboard.show_game:
                    display_action = self.agent.select_action(display_state, training=False)
                    display_state, _, display_done, display_info = self.display_env.step(display_action)
                    if display_done:
                        display_state = self.display_env.reset(record=True)
                    game_state = self.display_env.get_game_state()
                    current_score = display_info["score"]
                else:
                    game_state = None
                    current_score = infos[0]["score"]

                result = self.dashboard.update(
                    game_state=game_state,
                    episode=episode_count,
                    score=current_score,
                    epsilon=self.agent.epsilon,
                    loss=loss,
                    steps=self.step_count,
                    fps=render_fps,
                )

                if result == False:
                    self.running = False
                elif result == 'switch_mode':
                    print(f"\n[Mode] Switched to {'visual' if self.dashboard.show_game else 'headless'} mode")
                    if self.dashboard.show_game:
                        display_state = self.display_env.reset(record=True)

        return self.high_score

    def save_final_model(self):
        """Save the final model."""
        final_path = f"{self.config.checkpoint_dir}/final_model.pth"
        self.agent.save(final_path)
        print(f"[Training] Saved final model to {final_path}")
        return final_path

    def stop(self):
        """Stop training."""
        self.running = False

    def close(self):
        """Clean up resources."""
        self.vec_env.close()
        if self.dashboard:
            self.dashboard.close()
