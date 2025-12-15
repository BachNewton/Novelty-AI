"""
Abstract RL environment interface for Novelty AI.

Provides a Gym-like interface that all game environments must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class EnvInterface(ABC):
    """
    Abstract RL environment interface (Gym-like).

    Environments wrap games and provide observation encoding for agents.
    """

    @property
    @abstractmethod
    def state_size(self) -> int:
        """
        Dimension of the observation/state vector.

        Returns:
            Size of the state vector
        """
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        """
        Number of possible actions.

        Returns:
            Size of the action space
        """
        pass

    @abstractmethod
    def reset(self, record: bool = False) -> np.ndarray:
        """
        Reset the environment to initial state.

        Args:
            record: Whether to record this episode for replay

        Returns:
            Initial observation as numpy array
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: The action to take

        Returns:
            Tuple of (observation, reward, done, info)
            - observation: Current state as numpy array
            - reward: Reward for this step
            - done: Whether episode is finished
            - info: Additional information (score, etc.)
        """
        pass

    @abstractmethod
    def get_game_state(self) -> Dict[str, Any]:
        """
        Get the raw game state for visualization.

        Returns:
            Dictionary with game state for rendering
        """
        pass

    def get_replay(self) -> List[Dict[str, Any]]:
        """
        Get recorded frames if recording was enabled.

        Returns:
            List of frame dictionaries for replay
        """
        return []

    def render(self) -> None:
        """
        Render the current state (optional, for debugging).
        """
        pass

    def close(self) -> None:
        """
        Clean up any resources.
        """
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        pass
