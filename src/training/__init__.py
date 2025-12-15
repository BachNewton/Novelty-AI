"""Training module - shared training logic for Novelty AI."""
from .trainer import Trainer, TrainingConfig, get_default_num_envs
from .vec_env import VectorizedEnv, VectorizedSnakeEnv

__all__ = [
    "Trainer",
    "TrainingConfig",
    "get_default_num_envs",
    "VectorizedEnv",
    "VectorizedSnakeEnv",  # Backwards compatibility
]
