"""
Async Training - Decouple environment stepping from neural network training.

This module provides an AsyncTrainer that runs DQN training in a background
thread, allowing environment stepping and training to happen in parallel.

Key insight: PyTorch releases the GIL during tensor operations, so the training
thread can run truly parallel to the Python-based environment stepping.
"""

import threading
import time
from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class AsyncTrainingStats:
    """Statistics from async training."""
    train_steps: int = 0
    env_steps: int = 0
    trains_per_second: float = 0.0
    steps_per_second: float = 0.0


class AsyncTrainer:
    """
    Async trainer that runs DQN training in a background thread.

    This allows environment stepping to happen in the main thread while
    training happens continuously in the background, maximizing throughput.
    """

    def __init__(
        self,
        agent,
        train_interval: float = 0.0,
        min_buffer_size: Optional[int] = None,
        trains_per_step: int = 1,
    ):
        """
        Initialize async trainer.

        Args:
            agent: DQN agent with train_step() method
            train_interval: Minimum seconds between training steps (0 = as fast as possible)
            min_buffer_size: Minimum replay buffer size before training starts
            trains_per_step: Number of train_step() calls per iteration
        """
        self.agent = agent
        self.train_interval = train_interval
        self.min_buffer_size = min_buffer_size or agent.batch_size
        self.trains_per_step = trains_per_step

        # Threading primitives
        self._training_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused

        # Statistics
        self._train_steps = 0
        self._last_loss: Optional[float] = None
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()

    def start(self):
        """Start the background training thread."""
        if self._training_thread is not None and self._training_thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._pause_event.set()
        self._start_time = time.perf_counter()
        self._train_steps = 0

        self._training_thread = threading.Thread(
            target=self._training_loop,
            name="AsyncTrainer",
            daemon=True,
        )
        self._training_thread.start()

    def stop(self):
        """Stop the background training thread."""
        self._stop_event.set()
        self._pause_event.set()  # Unpause so thread can exit

        if self._training_thread is not None:
            self._training_thread.join(timeout=5.0)
            self._training_thread = None

    def pause(self):
        """Pause training (thread stays alive but doesn't train)."""
        self._pause_event.clear()

    def resume(self):
        """Resume training after pause."""
        self._pause_event.set()

    def _training_loop(self):
        """Background training loop."""
        while not self._stop_event.is_set():
            # Wait if paused
            self._pause_event.wait()

            if self._stop_event.is_set():
                break

            # Check if we have enough samples
            if len(self.agent.memory) < self.min_buffer_size:
                time.sleep(0.01)  # Wait for more samples
                continue

            # Perform training steps
            for _ in range(self.trains_per_step):
                loss = self.agent.train_step()
                if loss is not None:
                    with self._lock:
                        self._train_steps += 1
                        self._last_loss = loss

            # Optional sleep between training batches
            if self.train_interval > 0:
                time.sleep(self.train_interval)

    @property
    def train_steps(self) -> int:
        """Get number of training steps completed."""
        with self._lock:
            return self._train_steps

    @property
    def last_loss(self) -> Optional[float]:
        """Get the most recent training loss."""
        with self._lock:
            return self._last_loss

    def get_stats(self) -> AsyncTrainingStats:
        """Get training statistics."""
        with self._lock:
            train_steps = self._train_steps

        elapsed = time.perf_counter() - self._start_time if self._start_time else 1.0

        return AsyncTrainingStats(
            train_steps=train_steps,
            trains_per_second=train_steps / elapsed if elapsed > 0 else 0,
        )

    @property
    def is_running(self) -> bool:
        """Check if training thread is running."""
        return self._training_thread is not None and self._training_thread.is_alive()


class FastTrainingLoop:
    """
    Optimized training loop that maximizes throughput.

    Key optimizations:
    1. Async training in background thread
    2. Batched environment stepping (step N times, then sync)
    3. Minimal Python overhead in hot path
    4. Pre-allocated numpy arrays
    """

    def __init__(
        self,
        vec_env,
        agent,
        steps_before_sync: int = 4,
        trains_per_step: int = 1,
    ):
        """
        Initialize fast training loop.

        Args:
            vec_env: Vectorized environment
            agent: DQN agent
            steps_before_sync: Number of env steps before checking for episode ends
            trains_per_step: Training steps per env step batch
        """
        self.vec_env = vec_env
        self.agent = agent
        self.steps_before_sync = steps_before_sync

        # Pre-allocate arrays
        self.num_envs = vec_env.num_envs
        self.states = np.zeros((self.num_envs, vec_env.state_size), dtype=np.float32)
        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float32)

        # Async trainer
        self.async_trainer = AsyncTrainer(
            agent,
            train_interval=0,
            trains_per_step=trains_per_step,
        )

        # Statistics
        self.total_steps = 0
        self.episode_count = 0
        self.high_score = 0
        self.recent_scores: list = []

    def initialize(self, record: bool = True):
        """Initialize environments and start async training."""
        # Reset all environments
        for i, env in enumerate(self.vec_env.envs):
            self.states[i] = env.reset(record=record)

        self.async_trainer.start()

    def step_batch(self) -> list:
        """
        Step all environments and return completed episode infos.

        Returns:
            List of (env_index, score, steps) for completed episodes
        """
        completed = []

        # Get actions for all envs (batched forward pass)
        actions = self.agent.select_actions_batch(self.states, training=True)

        # Step all environments
        next_states, rewards, dones, infos = self.vec_env.step(actions)

        # Store transitions (this is the bottleneck we want to minimize)
        for i in range(self.num_envs):
            self.agent.store_transition(
                self.states[i], actions[i], rewards[i],
                next_states[i], dones[i]
            )
            self.episode_rewards[i] += rewards[i]

            if dones[i]:
                score = infos[i].get("score", 0)
                completed.append((i, score, self.total_steps))

                self.episode_count += 1
                self.recent_scores.append(score)
                if len(self.recent_scores) > 100:
                    self.recent_scores.pop(0)

                if score > self.high_score:
                    self.high_score = score

                # Reset this env
                self.episode_rewards[i] = 0
                next_states[i] = self.vec_env.envs[i].reset(record=True)

                # Decay epsilon
                self.agent.on_episode_end()

        self.states = next_states
        self.total_steps += self.num_envs

        return completed

    def close(self):
        """Stop async training and clean up."""
        self.async_trainer.stop()
