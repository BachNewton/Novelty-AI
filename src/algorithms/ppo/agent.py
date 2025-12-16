"""
PPO Agent - Proximal Policy Optimization agent implementing AgentInterface.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from ...core.agent_interface import AgentInterface
from .buffer import RolloutBuffer


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO.

    Shared feature extractor with separate policy (actor) and value (critic) heads.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256
    ):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both policy logits and value.

        Args:
            x: Input state tensor

        Returns:
            Tuple of (policy_logits, value)
        """
        features = self.shared(x)
        return self.policy(features), self.value(features)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        features = self.shared(x)
        return self.value(features)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Args:
            x: Input state tensor
            action: Optional action to evaluate (for training)

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)


class PPOAgent(AgentInterface):
    """
    Proximal Policy Optimization Agent implementing AgentInterface.

    Features:
    - Actor-Critic architecture with shared features
    - Clipped surrogate objective for stable updates
    - GAE (Generalized Advantage Estimation) for variance reduction
    - Entropy bonus for exploration
    - Device-agnostic (CPU or GPU)
    """

    def __init__(
        self,
        state_size: int = 86,
        action_size: int = 7,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PPO agent.

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            device: PyTorch device (None = auto-detect)
            config: Configuration dictionary with hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self._training_mode = True

        # Device handling (default to CPU, allow override)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        # Hyperparameters with defaults
        config = config or {}
        self.learning_rate = config.get("learning_rate", 0.0003)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.n_steps = config.get("n_steps", 2048)
        self.batch_size = config.get("batch_size", 64)
        self.n_epochs = config.get("n_epochs", 10)

        # Network
        self.network = ActorCriticNetwork(
            state_size, action_size, config.get("hidden_size", 256)
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            observation_size=state_size,
            action_size=action_size,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        # Training statistics
        self.steps_done = 0
        self.updates_done = 0
        self.episode_count = 0
        self.training_losses: list = []
        self.policy_losses: list = []
        self.value_losses: list = []
        self.entropies: list = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action given current state.

        Args:
            state: Current observation
            training: If True, sample from policy; if False, take argmax

        Returns:
            Selected action index
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.network(state_tensor)
            probs = torch.softmax(logits, dim=-1)

            if training and self._training_mode:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            else:
                action = probs.argmax(dim=-1)

            return action.item()

    def select_actions_batch(self, states: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select actions for batch of states.

        Args:
            states: Batch of observations
            training: If True, sample from policy

        Returns:
            Array of action indices
        """
        with torch.no_grad():
            state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
            logits, _ = self.network(state_tensor)
            probs = torch.softmax(logits, dim=-1)

            if training and self._training_mode:
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
            else:
                actions = probs.argmax(dim=-1)

            return actions.cpu().numpy()

    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate for a state.

        Args:
            state: Current observation

        Returns:
            Value estimate
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            value = self.network.get_value(state_tensor)
            return value.item()

    def get_action_and_value(
        self,
        state: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        Get action, log probability, and value for a state.

        Args:
            state: Current observation

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, _, value = self.network.get_action_and_value(state_tensor)
            return int(action.item()), float(log_prob.item()), float(value.item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store transition in rollout buffer.

        Note: PPO uses rollout buffer, not replay buffer.
        For compatibility with trainer, this stores to buffer.
        """
        # Get value and log_prob for the state-action
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
            _, log_prob, _, value = self.network.get_action_and_value(state_tensor, action_tensor)

        self.buffer.add(
            obs=np.array([state]),
            action=np.array([action]),
            reward=np.array([reward]),
            done=np.array([done]),
            value=np.array([value.item()]),
            log_prob=np.array([log_prob.item()])
        )

        self.steps_done += 1

    def store_transitions_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        """
        Store batch of transitions efficiently with single forward pass.

        This is much faster than calling store_transition N times because
        it computes value and log_prob for all states in one batched forward pass.

        Args:
            states: Batch of states (num_envs, state_size)
            actions: Batch of actions (num_envs,)
            rewards: Batch of rewards (num_envs,)
            next_states: Batch of next states (num_envs, state_size)
            dones: Batch of done flags (num_envs,)
        """
        num_envs = states.shape[0]

        # Single batched forward pass for all transitions
        with torch.no_grad():
            state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
            _, log_probs, _, values = self.network.get_action_and_value(state_tensor, action_tensor)

            log_probs_np = log_probs.cpu().numpy()
            values_np = values.cpu().numpy()

        # Store all transitions
        for i in range(num_envs):
            self.buffer.add(
                obs=np.array([states[i]]),
                action=np.array([actions[i]]),
                reward=np.array([rewards[i]]),
                done=np.array([dones[i]]),
                value=np.array([values_np[i]]),
                log_prob=np.array([log_probs_np[i]])
            )
            self.steps_done += 1

    def train_step(self) -> Optional[float]:
        """
        Perform PPO update if buffer is full.

        Returns:
            Average loss if update performed, None otherwise
        """
        if not self.buffer.full:
            return None

        # Compute advantages
        # Get value of last state (we need to know if episode ended)
        # For simplicity, assume episode continues
        last_value = 0.0  # Will be overwritten if not done
        self.buffer.compute_returns_and_advantages(
            last_values=np.array([[last_value]]),
            last_dones=np.array([[0.0]])
        )

        # PPO update epochs
        total_loss = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.get(self.batch_size):
                # Get current policy outputs
                _, new_log_prob, entropy, new_value = self.network.get_action_and_value(
                    batch.observations, batch.actions
                )

                # Normalize advantages
                advantages = batch.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_prob - batch.old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_value, batch.returns)

                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

                # Track individual losses
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropies.append(-entropy_loss.item())

        # Reset buffer for next rollout
        self.buffer.reset()
        self.updates_done += 1

        avg_loss = total_loss / max(n_updates, 1)
        self.training_losses.append(avg_loss)

        return avg_loss

    def on_episode_end(self) -> None:
        """Called at end of episode."""
        self.episode_count += 1

    def set_training_mode(self, training: bool) -> None:
        """Set training mode."""
        self._training_mode = training
        if training:
            self.network.train()
        else:
            self.network.eval()

    def save(self, filepath: str) -> None:
        """Save model checkpoint."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "updates_done": self.updates_done,
            "episode_count": self.episode_count,
            "training_losses": self.training_losses[-1000:],
        }, filepath)

    def load(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)

        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps_done = checkpoint["steps_done"]
        self.updates_done = checkpoint.get("updates_done", 0)
        self.episode_count = checkpoint.get("episode_count", 0)

        if "training_losses" in checkpoint:
            self.training_losses = checkpoint["training_losses"]

    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        avg_loss = 0.0
        avg_policy_loss = 0.0
        avg_value_loss = 0.0
        avg_entropy = 0.0

        if self.training_losses:
            avg_loss = sum(self.training_losses[-100:]) / len(self.training_losses[-100:])
        if self.policy_losses:
            avg_policy_loss = sum(self.policy_losses[-100:]) / len(self.policy_losses[-100:])
        if self.value_losses:
            avg_value_loss = sum(self.value_losses[-100:]) / len(self.value_losses[-100:])
        if self.entropies:
            avg_entropy = sum(self.entropies[-100:]) / len(self.entropies[-100:])

        return {
            "steps_done": self.steps_done,
            "updates_done": self.updates_done,
            "episode_count": self.episode_count,
            "avg_loss": avg_loss,
            "avg_policy_loss": avg_policy_loss,
            "avg_value_loss": avg_value_loss,
            "avg_entropy": avg_entropy,
            "buffer_size": len(self.buffer),
        }
