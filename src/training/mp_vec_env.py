"""
Multiprocessing Vectorized Environment - True parallel environment execution.

Uses multiprocessing with shared memory to escape Python's GIL and achieve
real parallelism for CPU-bound game simulations.
"""

import multiprocessing as mp
from multiprocessing import Process, Array, Queue, Event
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from ctypes import c_float, c_int, c_bool
import traceback


def _env_worker(
    worker_id: int,
    game_id: str,
    env_config: Dict[str, Any],
    state_size: int,
    action_size: int,
    # Shared memory arrays
    shared_states: Array,
    shared_rewards: Array,
    shared_dones: Array,
    shared_actions: Array,
    shared_scores: Array,
    # Synchronization
    cmd_queue: Queue,
    ready_event: Event,
    done_event: Event,
    error_queue: Queue,
):
    """
    Worker process function that runs a single environment.

    Each worker:
    1. Creates its own environment instance
    2. Waits for commands from main process
    3. Executes step/reset and writes results to shared memory
    4. Signals completion
    """
    try:
        # Import inside worker to avoid pygame init issues
        from ..games.registry import GameRegistry

        # Create environment instance
        env = GameRegistry.create_env(game_id, **env_config)

        # Signal ready
        ready_event.set()

        while True:
            # Wait for command
            cmd = cmd_queue.get()

            if cmd == 'step':
                # Read action from shared memory
                action = shared_actions[worker_id]

                # Execute step
                state, reward, done, info = env.step(action)

                # Write results to shared memory
                state_offset = worker_id * state_size
                for i, val in enumerate(state):
                    shared_states[state_offset + i] = val
                shared_rewards[worker_id] = reward
                shared_dones[worker_id] = done
                shared_scores[worker_id] = info.get('score', 0)

                # Signal completion
                done_event.set()

            elif cmd == 'reset':
                # Reset environment
                state = env.reset()

                # Write initial state to shared memory
                state_offset = worker_id * state_size
                for i, val in enumerate(state):
                    shared_states[state_offset + i] = val
                shared_dones[worker_id] = False
                shared_scores[worker_id] = 0

                # Signal completion
                done_event.set()

            elif cmd == 'close':
                env.close()
                break

    except Exception as e:
        error_queue.put((worker_id, str(e), traceback.format_exc()))
        ready_event.set()  # Don't block main process


class MultiprocessVecEnv:
    """
    Vectorized environment using true multiprocessing.

    Each environment runs in its own process, bypassing the GIL for
    CPU-bound game simulations. Uses shared memory for efficient
    data transfer between processes.
    """

    def __init__(
        self,
        game_id: str,
        num_envs: int,
        env_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize multiprocessing vectorized environment.

        Args:
            game_id: ID of the game to create environments for
            num_envs: Number of parallel environments
            env_config: Configuration to pass to environment constructors
        """
        self.game_id = game_id
        self.num_envs = num_envs
        self.env_config = env_config or {}

        # Create a temporary env to get state/action sizes
        from ..games.registry import GameRegistry
        temp_env = GameRegistry.create_env(game_id, **self.env_config)
        self.state_size = temp_env.state_size
        self.action_size = temp_env.action_size
        temp_env.close()

        # Create shared memory arrays
        # States: (num_envs * state_size) floats
        self.shared_states = Array(c_float, num_envs * self.state_size)
        # Rewards: (num_envs) floats
        self.shared_rewards = Array(c_float, num_envs)
        # Dones: (num_envs) bools
        self.shared_dones = Array(c_bool, num_envs)
        # Actions: (num_envs) ints
        self.shared_actions = Array(c_int, num_envs)
        # Scores: (num_envs) ints for info dict
        self.shared_scores = Array(c_int, num_envs)

        # Synchronization primitives
        self.cmd_queues: List[Queue] = []
        self.ready_events: List[Event] = []
        self.done_events: List[Event] = []
        self.error_queue = Queue()

        # Create and start worker processes
        self.workers: List[Process] = []

        for i in range(num_envs):
            cmd_queue = Queue()
            ready_event = Event()
            done_event = Event()

            self.cmd_queues.append(cmd_queue)
            self.ready_events.append(ready_event)
            self.done_events.append(done_event)

            worker = Process(
                target=_env_worker,
                args=(
                    i,
                    game_id,
                    self.env_config,
                    self.state_size,
                    self.action_size,
                    self.shared_states,
                    self.shared_rewards,
                    self.shared_dones,
                    self.shared_actions,
                    self.shared_scores,
                    cmd_queue,
                    ready_event,
                    done_event,
                    self.error_queue,
                ),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)

        # Wait for all workers to be ready
        for event in self.ready_events:
            event.wait(timeout=30)

        # Check for startup errors
        self._check_errors()

        # Placeholder for individual env access (for replay recording)
        # Workers handle their own envs, but we need a way to get replays
        self.envs = [_EnvProxy(i, self) for i in range(num_envs)]

    def _check_errors(self):
        """Check for and raise any worker errors."""
        while not self.error_queue.empty():
            worker_id, error_msg, tb = self.error_queue.get_nowait()
            raise RuntimeError(f"Worker {worker_id} error: {error_msg}\n{tb}")

    def reset(self) -> np.ndarray:
        """
        Reset all environments.

        Returns:
            Stacked initial states, shape (num_envs, state_size)
        """
        # Clear done events
        for event in self.done_events:
            event.clear()

        # Send reset command to all workers
        for queue in self.cmd_queues:
            queue.put('reset')

        # Wait for all workers to complete
        for event in self.done_events:
            event.wait()

        self._check_errors()

        # Read states from shared memory
        states = self._read_states()
        return states

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
        """
        # Write actions to shared memory
        for i, action in enumerate(actions):
            self.shared_actions[i] = int(action)

        # Clear done events
        for event in self.done_events:
            event.clear()

        # Send step command to all workers
        for queue in self.cmd_queues:
            queue.put('step')

        # Wait for all workers to complete
        for event in self.done_events:
            event.wait()

        self._check_errors()

        # Read results from shared memory
        states = self._read_states()
        rewards = np.array(self.shared_rewards[:], dtype=np.float32)
        dones = np.array(self.shared_dones[:], dtype=np.bool_)

        # Build info dicts (minimal, just score)
        infos = [{'score': self.shared_scores[i]} for i in range(self.num_envs)]

        return states, rewards, dones, infos

    def _read_states(self) -> np.ndarray:
        """Read states from shared memory into numpy array."""
        flat = np.array(self.shared_states[:], dtype=np.float32)
        return flat.reshape(self.num_envs, self.state_size)

    def close(self):
        """Clean up resources."""
        # Send close command to all workers
        for queue in self.cmd_queues:
            try:
                queue.put('close')
            except Exception:
                pass

        # Wait for workers to terminate
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        # Close queues
        for queue in self.cmd_queues:
            try:
                queue.close()
            except Exception:
                pass

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass


class _EnvProxy:
    """
    Proxy object that mimics env interface for compatibility.

    Used to maintain API compatibility with code that accesses
    vec_env.envs[i] directly (e.g., for replays).
    """

    def __init__(self, index: int, parent: MultiprocessVecEnv):
        self.index = index
        self.parent = parent
        self._recording = False

    def reset(self, record: bool = False) -> np.ndarray:
        """Reset this specific environment."""
        self._recording = record

        # Clear and send reset command to specific worker
        self.parent.done_events[self.index].clear()
        self.parent.cmd_queues[self.index].put('reset')
        self.parent.done_events[self.index].wait()

        # Read this env's state
        offset = self.index * self.parent.state_size
        state = np.array(
            self.parent.shared_states[offset:offset + self.parent.state_size],
            dtype=np.float32
        )
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step this specific environment."""
        self.parent.shared_actions[self.index] = action

        self.parent.done_events[self.index].clear()
        self.parent.cmd_queues[self.index].put('step')
        self.parent.done_events[self.index].wait()

        offset = self.index * self.parent.state_size
        state = np.array(
            self.parent.shared_states[offset:offset + self.parent.state_size],
            dtype=np.float32
        )
        reward = self.parent.shared_rewards[self.index]
        done = self.parent.shared_dones[self.index]
        info = {'score': self.parent.shared_scores[self.index]}

        return state, reward, done, info

    def get_replay(self) -> List[Dict[str, Any]]:
        """Get replay data - not supported in multiprocessing mode."""
        # Replays require accessing the game history, which lives in the worker
        # For now, return empty list. Full replay support would need IPC.
        return []

    def get_game_state(self) -> Dict[str, Any]:
        """Get game state - minimal support."""
        return {'score': self.parent.shared_scores[self.index]}

    def get_score(self) -> int:
        """Get current score."""
        return self.parent.shared_scores[self.index]

    def close(self):
        """No-op, parent handles cleanup."""
        pass
