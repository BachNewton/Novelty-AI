"""
DQN Agent - Deep Q-Learning agent with experience replay.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Optional, Dict, Any
from pathlib import Path

from .dqn_network import DQNNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent.

    Features:
    - Epsilon-greedy exploration with decay
    - Experience replay for stable learning
    - Target network for stable Q-value targets
    - Gradient clipping for stability
    """

    def __init__(
        self,
        state_size: int = 11,
        action_size: int = 3,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the DQN agent.

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            device: PyTorch device to use
            config: Configuration dictionary with hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or torch.device("cpu")

        # Hyperparameters with defaults
        config = config or {}
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 64)
        self.target_update_freq = config.get("target_update_freq", 100)
        buffer_size = config.get("buffer_size", 100000)

        # Networks
        self.policy_net = DQNNetwork(state_size, 256, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, 256, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Training statistics
        self.steps_done = 0
        self.training_losses: list = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: If True, use exploration; if False, use greedy policy

        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Loss value if training occurred, None if not enough samples
        """
        if not self.memory.is_ready(self.batch_size):
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q(s, a)
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Q target: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = self.criterion(q_values, q_targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1

        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_value = loss.item()
        self.training_losses.append(loss_value)

        return loss_value

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "training_losses": self.training_losses[-1000:],  # Keep last 1000
        }, filepath)

    def load(self, filepath: str):
        """
        Load model checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]

        if "training_losses" in checkpoint:
            self.training_losses = checkpoint["training_losses"]

    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        avg_loss = 0.0
        if self.training_losses:
            recent = self.training_losses[-100:]
            avg_loss = sum(recent) / len(recent)

        return {
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "memory_size": len(self.memory),
            "avg_loss": avg_loss,
        }
