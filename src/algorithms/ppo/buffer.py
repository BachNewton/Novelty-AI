"""
PPO Rollout Buffer - Stores trajectories for PPO training.
"""

import numpy as np
import torch
from typing import Tuple, Generator, NamedTuple


class RolloutBufferSamples(NamedTuple):
    """Batch of rollout samples for PPO training."""
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """
    Rollout buffer for PPO that stores trajectories and computes advantages.

    Stores transitions collected during rollout phase and provides
    mini-batches for training with computed advantages using GAE.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_size: int,
        action_size: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_envs: int = 1
    ):
        """
        Initialize the rollout buffer.

        Args:
            buffer_size: Number of steps per rollout
            observation_size: Dimension of observations
            action_size: Number of possible actions
            device: PyTorch device for tensors
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            n_envs: Number of parallel environments
        """
        self.buffer_size = buffer_size
        self.observation_size = observation_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_envs = n_envs

        # Storage arrays
        self.observations = np.zeros(
            (buffer_size, n_envs, observation_size), dtype=np.float32
        )
        self.actions = np.zeros((buffer_size, n_envs), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)

        # Computed during finalization
        self.advantages = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, n_envs), dtype=np.float32)

        self.pos = 0
        self.full = False

    def reset(self) -> None:
        """Reset the buffer for new rollout."""
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            obs: Observations
            action: Actions taken
            reward: Rewards received
            done: Episode done flags
            value: Value estimates
            log_prob: Log probabilities of actions
        """
        if self.pos >= self.buffer_size:
            raise RuntimeError("Rollout buffer is full. Call reset() before adding more.")

        self.observations[self.pos] = obs.reshape(self.n_envs, -1)
        self.actions[self.pos] = action.flatten()
        self.rewards[self.pos] = reward.flatten()
        self.dones[self.pos] = done.flatten()
        self.values[self.pos] = value.flatten()
        self.log_probs[self.pos] = log_prob.flatten()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray) -> None:
        """
        Compute returns and advantages using GAE.

        Args:
            last_values: Value estimates for the last observations
            last_dones: Done flags for the last step
        """
        last_values = last_values.flatten()
        last_dones = last_dones.flatten()

        # GAE computation
        last_gae_lam = 0
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def get(self, batch_size: int) -> Generator[RolloutBufferSamples, None, None]:
        """
        Generate mini-batches for training.

        Args:
            batch_size: Size of each mini-batch

        Yields:
            RolloutBufferSamples for each mini-batch
        """
        indices = np.random.permutation(self.pos * self.n_envs)
        start_idx = 0

        while start_idx < len(indices):
            batch_indices = indices[start_idx:start_idx + batch_size]

            # Convert flat indices to (step, env) indices
            env_indices = batch_indices % self.n_envs
            step_indices = batch_indices // self.n_envs

            yield RolloutBufferSamples(
                observations=torch.tensor(
                    self.observations[step_indices, env_indices],
                    dtype=torch.float32,
                    device=self.device
                ),
                actions=torch.tensor(
                    self.actions[step_indices, env_indices],
                    dtype=torch.long,
                    device=self.device
                ),
                old_values=torch.tensor(
                    self.values[step_indices, env_indices],
                    dtype=torch.float32,
                    device=self.device
                ),
                old_log_probs=torch.tensor(
                    self.log_probs[step_indices, env_indices],
                    dtype=torch.float32,
                    device=self.device
                ),
                advantages=torch.tensor(
                    self.advantages[step_indices, env_indices],
                    dtype=torch.float32,
                    device=self.device
                ),
                returns=torch.tensor(
                    self.returns[step_indices, env_indices],
                    dtype=torch.float32,
                    device=self.device
                ),
            )

            start_idx += batch_size

    def __len__(self) -> int:
        """Return number of transitions in buffer."""
        return self.pos * self.n_envs
