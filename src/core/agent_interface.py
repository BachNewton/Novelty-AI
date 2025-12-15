"""
Abstract agent interface for Novelty AI.

All learning agents (DQN, PPO, etc.) must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np


class AgentInterface(ABC):
    """
    Abstract agent interface for all learning algorithms.

    Agents handle action selection, experience storage, and training.
    """

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action given the current state.

        Args:
            state: Current observation/state
            training: If True, may include exploration; if False, greedy

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def select_actions_batch(
        self, states: np.ndarray, training: bool = True
    ) -> np.ndarray:
        """
        Select actions for a batch of states (for vectorized environments).

        Args:
            states: Batch of observations, shape (batch_size, state_size)
            training: If True, may include exploration

        Returns:
            Array of selected actions, shape (batch_size,)
        """
        pass

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store an experience transition for later training.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode ended
        """
        pass

    @abstractmethod
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Training loss, or None if not enough data to train
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save agent checkpoint to file.

        Args:
            filepath: Path to save checkpoint
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load agent checkpoint from file.

        Args:
            filepath: Path to checkpoint file
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current training statistics.

        Returns:
            Dictionary with stats like epsilon, steps_done, etc.
        """
        pass

    def on_episode_end(self) -> None:
        """
        Called at the end of each episode.

        Use for epsilon decay, learning rate scheduling, etc.
        """
        pass

    def set_training_mode(self, training: bool) -> None:
        """
        Set the agent's training mode.

        Args:
            training: True for training mode, False for evaluation
        """
        pass


class AgentConfig:
    """
    Base configuration for agents.

    Subclasses should define algorithm-specific parameters.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 100000,
        **kwargs
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return vars(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create config from dictionary."""
        return cls(**data)
