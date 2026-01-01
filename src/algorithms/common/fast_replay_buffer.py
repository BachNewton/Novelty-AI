"""
Fast Replay Buffer - Optimized for high-throughput async training.

Key optimizations:
1. Pre-allocated numpy arrays (no per-push allocation)
2. Thread-safe with minimal locking
3. Vectorized sampling
4. Memory-efficient circular buffer
"""

import threading
import numpy as np
from typing import Tuple, Optional


class FastReplayBuffer:
    """
    High-performance replay buffer using pre-allocated numpy arrays.

    Thread-safe for concurrent push/sample operations, designed for
    async training where one thread pushes and another samples.
    """

    def __init__(
        self,
        capacity: int,
        state_size: int,
        seed: Optional[int] = None,
    ):
        """
        Initialize the replay buffer with pre-allocated arrays.

        Args:
            capacity: Maximum number of experiences to store
            state_size: Dimension of state vectors
            seed: Random seed for reproducible sampling
        """
        self.capacity = capacity
        self.state_size = state_size

        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Circular buffer state
        self._index = 0
        self._size = 0
        self._lock = threading.Lock()

        # Random state for sampling
        self._rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add an experience to the buffer (thread-safe).

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        with self._lock:
            idx = self._index

            self.states[idx] = state
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.next_states[idx] = next_state
            self.dones[idx] = float(done)

            self._index = (self._index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ):
        """
        Add a batch of experiences (thread-safe, more efficient than individual pushes).

        Args:
            states: Batch of states (N, state_size)
            actions: Batch of actions (N,)
            rewards: Batch of rewards (N,)
            next_states: Batch of next states (N, state_size)
            dones: Batch of done flags (N,)
        """
        batch_size = len(states)

        with self._lock:
            # Calculate indices for this batch
            indices = np.arange(self._index, self._index + batch_size) % self.capacity

            self.states[indices] = states
            self.actions[indices] = actions
            self.rewards[indices] = rewards
            self.next_states[indices] = next_states
            self.dones[indices] = dones.astype(np.float32)

            self._index = (self._index + batch_size) % self.capacity
            self._size = min(self._size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of experiences (thread-safe).

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) arrays
        """
        with self._lock:
            size = self._size
            if size < batch_size:
                raise ValueError(f"Not enough samples: {size} < {batch_size}")

            # Sample random indices
            indices = self._rng.integers(0, size, size=batch_size)

            # Return copies to avoid race conditions
            return (
                self.states[indices].copy(),
                self.actions[indices].copy(),
                self.rewards[indices].copy(),
                self.next_states[indices].copy(),
                self.dones[indices].copy(),
            )

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        with self._lock:
            return self._size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        with self._lock:
            return self._size >= batch_size


class LockFreeReplayBuffer:
    """
    Lock-free replay buffer for maximum throughput.

    Uses atomic operations and accepts slightly stale data during sampling
    to avoid locking overhead. Safe for single-producer single-consumer
    (one thread pushing, one thread sampling).
    """

    def __init__(
        self,
        capacity: int,
        state_size: int,
        seed: Optional[int] = None,
    ):
        """
        Initialize the lock-free replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            state_size: Dimension of state vectors
            seed: Random seed for reproducible sampling
        """
        self.capacity = capacity
        self.state_size = state_size

        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Use volatile-style access (Python int is atomic for read/write)
        self._index = 0
        self._size = 0

        # Random state
        self._rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add an experience (lock-free, single producer assumed)."""
        idx = self._index

        # Write data (numpy array assignment is atomic at element level)
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        # Update index and size (atomic int operations)
        self._index = (idx + 1) % self.capacity
        if self._size < self.capacity:
            self._size = self._size + 1

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch (may include very recent data)."""
        size = self._size
        if size < batch_size:
            raise ValueError(f"Not enough samples: {size} < {batch_size}")

        indices = self._rng.integers(0, size, size=batch_size)

        return (
            self.states[indices].copy(),
            self.actions[indices].copy(),
            self.rewards[indices].copy(),
            self.next_states[indices].copy(),
            self.dones[indices].copy(),
        )

    def __len__(self) -> int:
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size
