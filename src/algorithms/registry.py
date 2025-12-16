"""
Algorithm registry for Novelty AI.

Central registry for discovering and instantiating learning algorithms.
"""

from typing import Dict, Type, List, Optional, Any
import torch

from ..core.agent_interface import AgentInterface


class AlgorithmRegistry:
    """
    Central registry for all available learning algorithms.

    Algorithms register themselves using AlgorithmRegistry.register().
    """

    _algorithms: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        algorithm_id: str,
        agent_class: Type[AgentInterface],
        config_class: Optional[Type] = None,
        description: str = ""
    ) -> None:
        """
        Register an algorithm with the registry.

        Args:
            algorithm_id: Unique identifier (e.g., "dqn", "ppo")
            agent_class: The agent implementation class
            config_class: Optional algorithm-specific config class
            description: Human-readable description
        """
        cls._algorithms[algorithm_id] = {
            'agent_class': agent_class,
            'config_class': config_class,
            'description': description,
        }

    @classmethod
    def get_algorithm(cls, algorithm_id: str) -> Optional[Dict[str, Any]]:
        """
        Get algorithm components by ID.

        Args:
            algorithm_id: The algorithm identifier

        Returns:
            Dictionary with agent class, or None if not found
        """
        return cls._algorithms.get(algorithm_id)

    @classmethod
    def list_algorithms(cls) -> List[str]:
        """
        List all registered algorithm IDs.

        Returns:
            List of algorithm identifiers
        """
        return list(cls._algorithms.keys())

    @classmethod
    def get_description(cls, algorithm_id: str) -> str:
        """
        Get description for an algorithm.

        Args:
            algorithm_id: The algorithm identifier

        Returns:
            Description string
        """
        algo = cls._algorithms.get(algorithm_id)
        return algo['description'] if algo else ""

    @classmethod
    def is_available(cls, algorithm_id: str) -> bool:
        """
        Check if an algorithm is registered.

        Args:
            algorithm_id: The algorithm identifier

        Returns:
            True if algorithm is available
        """
        return algorithm_id in cls._algorithms

    @classmethod
    def create_agent(
        cls,
        algorithm_id: str,
        env: Any,
        device: torch.device,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentInterface:
        """
        Create an agent instance for the given algorithm.

        Args:
            algorithm_id: The algorithm identifier
            env: Environment with state_size and action_size properties
            device: PyTorch device to use
            config: Optional configuration dictionary

        Returns:
            Agent instance

        Raises:
            ValueError: If algorithm is not registered
        """
        algo_data = cls._algorithms.get(algorithm_id)
        if not algo_data:
            raise ValueError(f"Unknown algorithm: {algorithm_id}")

        agent_class = algo_data['agent_class']
        return agent_class(
            state_size=env.state_size,
            action_size=env.action_size,
            device=device,
            config=config
        )

    @classmethod
    def get_config_class(cls, algorithm_id: str) -> Optional[Type]:
        """
        Get the config class for an algorithm.

        Args:
            algorithm_id: The algorithm identifier

        Returns:
            Config class or None
        """
        algo_data = cls._algorithms.get(algorithm_id)
        if algo_data:
            return algo_data.get('config_class')
        return None

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (useful for testing)."""
        cls._algorithms.clear()
