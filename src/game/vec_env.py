"""
Vectorized Snake Environment - Run multiple environments in parallel.
Enables faster experience collection by utilizing multiple CPU cores.
"""
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from .snake_env import SnakeEnv


class VectorizedSnakeEnv:
    """
    Vectorized wrapper for running multiple Snake environments in parallel.

    This allows collecting experiences from N environments simultaneously,
    significantly speeding up training when CPU cores are available.
    """

    def __init__(
        self,
        num_envs: int,
        width: int = 20,
        height: int = 20,
        reward_config: Optional[Dict[str, float]] = None,
        num_workers: Optional[int] = None
    ):
        """
        Initialize vectorized environment.

        Args:
            num_envs: Number of parallel environments
            width: Grid width in cells
            height: Grid height in cells
            reward_config: Optional reward configuration
            num_workers: Number of worker threads (default: num_envs)
        """
        self.num_envs = num_envs
        self.width = width
        self.height = height
        self.reward_config = reward_config

        # Create environments
        self.envs = [
            SnakeEnv(
                width=width,
                height=height,
                reward_config=reward_config
            )
            for _ in range(num_envs)
        ]

        # Thread pool for parallel execution
        self.num_workers = num_workers or num_envs
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

        # State and action sizes (same for all envs)
        self.state_size = self.envs[0].state_size
        self.action_size = self.envs[0].action_size

    def reset(self) -> np.ndarray:
        """
        Reset all environments.

        Returns:
            Stacked initial states, shape (num_envs, ...) where ... is state shape
        """
        # Submit all reset tasks
        futures = [self.executor.submit(env.reset) for env in self.envs]

        # Collect results
        states = [f.result() for f in futures]

        return np.stack(states)

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments with given actions.

        Args:
            actions: Array of actions, shape (num_envs,)

        Returns:
            Tuple of (states, rewards, dones, infos)
            - states: (num_envs, ...) state array
            - rewards: (num_envs,) reward array
            - dones: (num_envs,) done flags
            - infos: List of info dicts
        """
        # Submit all step tasks
        futures = [
            self.executor.submit(env.step, int(action))
            for env, action in zip(self.envs, actions)
        ]

        # Collect results
        results = [f.result() for f in futures]

        states = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=np.bool_)
        infos = [r[3] for r in results]

        return states, rewards, dones, infos

    def reset_done_envs(self, dones: np.ndarray) -> np.ndarray:
        """
        Reset environments that are done.

        Args:
            dones: Boolean array indicating which envs are done

        Returns:
            New states for reset environments (only for done envs)
        """
        new_states = []
        reset_indices = np.where(dones)[0]

        # Submit reset tasks for done environments
        futures = [(i, self.executor.submit(self.envs[i].reset)) for i in reset_indices]

        # Collect results
        for i, f in futures:
            new_states.append((i, f.result()))

        return new_states

    def get_scores(self) -> np.ndarray:
        """Get current scores for all environments."""
        return np.array([env.get_score() for env in self.envs])

    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass
