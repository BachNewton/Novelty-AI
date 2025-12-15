"""
Abstract game interface for Novelty AI.

All games must implement GameInterface and provide GameMetadata.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional


@dataclass
class GameMetadata:
    """Metadata describing a game."""

    name: str                           # Display name (e.g., "Snake")
    id: str                             # Unique identifier (e.g., "snake")
    description: str                    # Brief description for UI
    version: str = "1.0.0"              # Game version
    min_players: int = 1                # Minimum players
    max_players: int = 1                # Maximum players
    supports_human: bool = True         # Can humans play?
    recommended_algorithms: List[str] = field(default_factory=lambda: ["dqn"])


class GameInterface(ABC):
    """
    Abstract base class for all games in Novelty AI.

    Games handle the core logic, rules, and state management.
    They are separate from the RL environment wrapper.
    """

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> GameMetadata:
        """
        Return metadata about this game.

        Returns:
            GameMetadata describing the game
        """
        pass

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Reset the game to initial state.

        Returns:
            Initial game state dictionary
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one game step with the given action.

        Args:
            action: The action to take (game-specific encoding)

        Returns:
            Tuple of (state, reward, done, info)
            - state: Current game state dictionary
            - reward: Reward for this step
            - done: Whether the game is over
            - info: Additional information dictionary
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current game state for rendering.

        Returns:
            Dictionary containing all state needed for rendering
        """
        pass

    @abstractmethod
    def is_valid_action(self, action: int) -> bool:
        """
        Check if an action is valid in the current state.

        Args:
            action: The action to check

        Returns:
            True if action is valid, False otherwise
        """
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """
        Number of possible actions in this game.

        Returns:
            Size of the action space
        """
        pass

    @property
    @abstractmethod
    def action_names(self) -> List[str]:
        """
        Human-readable names for each action.

        Returns:
            List of action names indexed by action number
        """
        pass

    # Optional recording support
    def start_recording(self) -> None:
        """Start recording game frames for replay."""
        pass

    def stop_recording(self) -> List[Dict[str, Any]]:
        """
        Stop recording and return recorded frames.

        Returns:
            List of frame dictionaries
        """
        return []

    def get_score(self) -> int:
        """
        Get the current score.

        Returns:
            Current game score
        """
        return 0
