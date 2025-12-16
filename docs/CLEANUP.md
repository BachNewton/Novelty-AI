# Cleanup Tasks

Technical debt and cleanup items to address when time permits. These are low-priority items that don't affect functionality but improve code quality.

## Backwards Compatibility Imports

These imports exist from before the registry pattern was implemented. They can be removed once we verify nothing external depends on them.

### `src/training/trainer.py` (lines 21-33)
```python
# Backwards compatibility
try:
    from ..games.snake.env import SnakeEnv
except ImportError:
    from ..game.snake_env import SnakeEnv

try:
    from ..algorithms.dqn.agent import DQNAgent
except ImportError:
    from ..ai.dqn_agent import DQNAgent

# Alias for backwards compatibility
VectorizedSnakeEnv = VectorizedEnv
```
**Status**: Can be removed - we now use GameRegistry and AlgorithmRegistry
**Risk**: Low - verify no external scripts import these aliases

### `src/training/__init__.py` (line 10)
```python
"VectorizedSnakeEnv",  # Backwards compatibility
```
**Status**: Can be removed along with trainer.py alias
**Risk**: Low

### `src/training/vec_env.py` (line 147)
```python
# Backwards compatibility alias
```
**Status**: Review and remove if unused
**Risk**: Low

## Unused Code

Items that may no longer be needed:

- [ ] Review if `src/game/` directory exists and can be deleted (old structure)
- [ ] Review if `src/ai/` directory exists and can be deleted (old structure)

## Code Quality Improvements

- [ ] Add type hints to any functions missing them
- [ ] Review and consolidate duplicate code patterns
- [ ] Add docstrings where missing

---

*Last updated: 2024-12-16*
