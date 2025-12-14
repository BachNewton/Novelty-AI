"""Training module - shared training logic for Snake AI."""
from .trainer import Trainer, TrainingConfig, get_default_num_envs

__all__ = ["Trainer", "TrainingConfig", "get_default_num_envs"]
