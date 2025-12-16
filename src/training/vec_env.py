"""
Vectorized Environment - Run multiple game environments in parallel.
Enables faster experience collection by utilizing multiple CPU cores.
Works with any game through the registry.

Uses multiprocessing to bypass Python GIL for true parallelism.
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Process
from typing import Tuple, Dict, Any, List, Optional

from ..core.env_interface import EnvInterface
from ..games.registry import GameRegistry


def _worker_process(
    conn,
    game_id: str,
    env_config: Dict[str, Any]
):
    """
    Worker process that owns and steps a single environment.

    Runs in a separate process to bypass GIL.
    Communicates with main process via Pipe.
    """
    from ..games.registry import GameRegistry

    # Create environment in this process
    env = GameRegistry.create_env(game_id, **env_config)

    while True:
        try:
            cmd, data = conn.recv()
        except (EOFError, ConnectionResetError):
            break

        try:
            if cmd == "step":
                state, reward, done, info = env.step(data)
                conn.send((state, reward, done, info))

            elif cmd == "reset":
                record = data.get("record", False) if isinstance(data, dict) else False
                state = env.reset(record=record)
                conn.send(state)

            elif cmd == "get_game_state":
                state = env.get_game_state()
                conn.send(state)

            elif cmd == "get_replay":
                replay = env.get_replay()
                conn.send(replay)

            elif cmd == "close":
                env.close()
                conn.close()
                break
        except Exception as e:
            conn.send(("error", str(e)))


class VectorizedEnv:
    """
    Vectorized wrapper for running multiple game environments in parallel.

    Uses multiprocessing for true parallelism, bypassing Python GIL.
    This allows collecting experiences from N environments simultaneously,
    significantly speeding up training when CPU cores are available.

    Works with any game registered in the GameRegistry.
    """

    def __init__(
        self,
        game_id: str,
        num_envs: int,
        env_config: Optional[Dict[str, Any]] = None,
        num_workers: Optional[int] = None
    ):
        """
        Initialize vectorized environment with multiprocessing.

        Args:
            game_id: ID of the game to create environments for
            num_envs: Number of parallel environments
            env_config: Configuration to pass to environment constructors
            num_workers: Ignored (kept for API compatibility)
        """
        self.game_id = game_id
        self.num_envs = num_envs
        self.env_config = env_config or {}

        # Create a dummy env to get state/action sizes
        dummy_env = GameRegistry.create_env(game_id, **self.env_config)
        self.state_size = dummy_env.state_size
        self.action_size = dummy_env.action_size
        dummy_env.close()

        # Start worker processes
        self.workers: List[Process] = []
        self.conns: List[Any] = []

        # Use spawn context for cross-platform compatibility (especially Windows)
        ctx = mp.get_context("spawn")

        for i in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            worker = ctx.Process(
                target=_worker_process,
                args=(child_conn, game_id, self.env_config),
                daemon=True
            )
            worker.start()
            child_conn.close()

            self.workers.append(worker)
            self.conns.append(parent_conn)

        # Create wrapper objects for direct env access (backwards compatibility)
        self.envs = [_EnvProxy(conn, i) for i, conn in enumerate(self.conns)]

        self._closed = False

    def reset(self) -> np.ndarray:
        """Reset all environments."""
        for conn in self.conns:
            conn.send(("reset", {"record": False}))

        states = [conn.recv() for conn in self.conns]
        return np.stack(states)

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step all environments with given actions."""
        for conn, action in zip(self.conns, actions):
            conn.send(("step", int(action)))

        results = [conn.recv() for conn in self.conns]

        states = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=np.bool_)
        infos = [r[3] for r in results]

        return states, rewards, dones, infos

    def reset_done_envs(self, dones: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Reset environments that are done."""
        reset_indices = np.where(dones)[0]

        for i in reset_indices:
            self.conns[i].send(("reset", {"record": False}))

        new_states = []
        for i in reset_indices:
            state = self.conns[i].recv()
            new_states.append((i, state))

        return new_states

    def get_game_states(self) -> List[Dict[str, Any]]:
        """Get raw game states for all environments."""
        for conn in self.conns:
            conn.send(("get_game_state", None))

        return [conn.recv() for conn in self.conns]

    def close(self):
        """Clean up resources."""
        if self._closed:
            return

        self._closed = True

        for conn in self.conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass

        for worker in self.workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()

        for conn in self.conns:
            try:
                conn.close()
            except Exception:
                pass

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass


class _EnvProxy:
    """Proxy object that mimics EnvInterface for backwards compatibility."""

    def __init__(self, conn, index: int):
        self.conn = conn
        self.index = index

    def reset(self, record: bool = False) -> np.ndarray:
        """Reset the environment."""
        self.conn.send(("reset", {"record": record}))
        return self.conn.recv()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step the environment."""
        self.conn.send(("step", action))
        return self.conn.recv()

    def get_game_state(self) -> Dict[str, Any]:
        """Get raw game state."""
        self.conn.send(("get_game_state", None))
        return self.conn.recv()

    def get_replay(self) -> List[Dict[str, Any]]:
        """Get recorded replay."""
        self.conn.send(("get_replay", None))
        return self.conn.recv()


# Backwards compatibility alias
VectorizedSnakeEnv = VectorizedEnv
