# Claude Code Instructions for Novelty AI

## Environment
- Running on Windows with Git Bash (MINGW)
- Use `/dev/null` instead of `nul` for redirecting output
- Shell is Unix-like, not CMD

## Project Overview

**Novelty AI** is a multi-game AI training hub. It provides:
- Unified interface for training AI on multiple games
- Registry pattern for games and algorithms (auto-discovery)
- Dual access: UI for humans, CLI for servers/AI automation
- Per-game configurations and experiment tracking

## Directory Structure

```
Novelty-AI/
├── README.md                     # Project overview
├── CLAUDE.md                     # This file - AI development guide
├── docs/games/                   # Per-game documentation
│   └── snake/
│       └── EXPERIMENTS.md        # Snake experiment notes
├── config/
│   ├── default.yaml              # Global settings
│   └── games/
│       └── snake.yaml            # Snake-specific config
├── src/
│   ├── core/                     # Abstract interfaces
│   │   ├── game_interface.py     # GameInterface ABC, GameMetadata
│   │   ├── env_interface.py      # EnvInterface ABC (gym-like)
│   │   ├── agent_interface.py    # AgentInterface ABC
│   │   └── renderer_interface.py # RendererInterface ABC
│   ├── device/                   # Hardware management
│   │   └── device_manager.py     # GPU/CPU device selection
│   ├── games/                    # Game implementations
│   │   ├── registry.py           # GameRegistry
│   │   └── snake/                # Snake game
│   ├── algorithms/               # AI algorithms
│   │   ├── registry.py           # AlgorithmRegistry
│   │   ├── common/               # Shared utilities (replay buffer)
│   │   └── dqn/                  # Deep Q-Network
│   ├── training/                 # Training infrastructure
│   │   ├── trainer.py            # Generic trainer
│   │   └── vec_env.py            # Vectorized environments
│   ├── utils/                    # Utility modules
│   │   └── config_loader.py      # Config loading utilities
│   └── visualization/            # UI components
│       ├── ui_components.py      # Button, Dropdown, Toggle widgets
│       ├── game_hub.py           # Game selection hub
│       ├── game_menu.py          # Per-game menu
│       ├── main_menu.py          # Main menu screen
│       ├── dashboard.py          # Training dashboard
│       ├── replay_player.py      # Replay viewing
│       └── hardware_monitor.py   # GPU/CPU stats display
├── scripts/
│   ├── ui.py                     # UI entry point
│   ├── train.py                  # CLI training
│   ├── play.py                   # CLI AI/human play
│   ├── evaluate.py               # Model evaluation
│   └── watch_replays.py          # CLI replay viewer
├── models/{game_id}/             # Per-game trained models
└── replays/{game_id}/            # Per-game replays
```

## Key Concepts

### Registry Pattern
Games and algorithms auto-register via imports:
- `src/games/snake/__init__.py` registers Snake with GameRegistry
- `src/algorithms/dqn/__init__.py` registers DQN with AlgorithmRegistry

### Interfaces
All games implement these interfaces:
- `EnvInterface` - Gym-like environment (reset, step, get_game_state)
- `RendererInterface` - Visual rendering (set_surface, render)
- `GameInterface` - Game logic (reset, step, get_state)

### Configuration
Hierarchical configs: `config/default.yaml` + `config/games/{game}.yaml`
Game configs override defaults.

## AI Iteration Workflow

Use these CLI commands to train, evaluate, and iterate on models:

### Training
```bash
# Train with visualization
python scripts/train.py --game snake

# Headless training (faster, for servers)
python scripts/train.py --game snake --headless

# Custom episode count
python scripts/train.py --game snake --episodes 5000 --headless

# Resume from checkpoint
python scripts/train.py --game snake --load models/snake/model_ep1000.pth
```

### Evaluation
```bash
# Evaluate model with statistics
python scripts/evaluate.py --game snake --episodes 100

# Machine-readable JSON output
python scripts/evaluate.py --game snake --json

# Output: mean, median, stdev, min, max scores
```

### Iteration Loop
1. **Train**: `python scripts/train.py --game snake --episodes 1000 --headless`
2. **Evaluate**: `python scripts/evaluate.py --game snake --json`
3. **Review** JSON output (mean score, consistency via stdev)
4. **If score < target**: Adjust hyperparameters in `config/games/snake.yaml`
5. **Repeat** until target score achieved
6. **Document** findings in `docs/games/snake/EXPERIMENTS.md`

### Quick Evaluation Script
```bash
# Train short run and evaluate
python scripts/train.py -g snake --episodes 500 --headless && \
python scripts/evaluate.py -g snake --episodes 50 --json
```

## Running Tests

**IMPORTANT**: Always run tests after making changes to verify nothing is broken.

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_screens.py -v

# Run tests matching a pattern
python -m pytest tests/ -v -k "GameMenu"
```

### Test Structure

```
tests/
├── conftest.py              # Pygame mock, shared fixtures
├── test_ui_components.py    # Button, Dropdown, Toggle tests
├── test_screens.py          # GameHub, GameMenu, Dashboard init tests
├── test_ui_integration.py   # Entry points, dropdown population
├── test_ui_click_paths.py   # UI navigation and click flow tests
└── test_infrastructure.py   # GameRegistry, config loading
```

### What Tests Catch

- Import errors (missing modules, circular imports)
- Initialization crashes (AttributeError, TypeError, KeyError)
- Path issues (like double paths: `models/snake/snake/`)
- Registry issues (missing methods, wrong keys)
- Config issues (missing files, invalid structure)

### When to Write Tests vs Rely on Mypy

**Mypy catches** (no test needed):
- Calling methods that don't exist
- Wrong number of arguments
- Wrong argument types
- Missing imports

**Tests catch** (write a test):
- Logic errors (wrong calculations, off-by-one)
- Integration issues (components don't work together)
- Runtime state issues (paths constructed incorrectly)
- Behavior verification (does clicking X do Y?)

**Rule**: If mypy would catch it at "compile time", don't write a test for it. Focus tests on runtime behavior and integration.

## Type Hints and Static Type Checking

This project uses type hints throughout. Type hints enable mypy to catch bugs at "compile time" instead of runtime.

### Writing Type Hints

**IMPORTANT**: Always add type hints when writing new code.

```python
# Function with type hints
def calculate_reward(score: int, multiplier: float = 1.0) -> float:
    return score * multiplier

# Optional parameters use Optional or | None
def load_model(path: str, device: Optional[str] = None) -> Model:
    ...

# Complex types
def get_scores(episodes: List[int]) -> Dict[str, float]:
    ...
```

### Running Mypy

**IMPORTANT**: Run mypy after making code changes to catch type errors before runtime.

```bash
# Check specific file
python -m mypy src/training/trainer.py --ignore-missing-imports

# Check entire src directory
python -m mypy src/ --ignore-missing-imports

# Check scripts too
python -m mypy src/ scripts/ --ignore-missing-imports
```

### What Mypy Catches

- Calling methods that don't exist (e.g., `agent.decay_epsilon()` when method is `on_episode_end()`)
- Missing required arguments (e.g., `renderer.render(state)` missing `surface` argument)
- Wrong argument types
- Attribute access on wrong types
- Returning wrong types from functions

### When to Run Mypy

Run mypy:
1. After writing new code
2. After changing method signatures
3. After renaming methods or attributes
4. Before committing changes

Mypy catches errors instantly that would otherwise only appear at runtime when you click through the UI.

## Planning New Features

- Before implementing training improvements, review `docs/games/snake/EXPERIMENTS.md`
- Document what worked and what didn't
- Helps avoid repeating failed approaches

## Adding a New Game

1. Create `src/games/{game_id}/` with:
   - `__init__.py` - Auto-registers game
   - `game.py` - Implements GameInterface
   - `env.py` - Implements EnvInterface
   - `renderer.py` - Implements RendererInterface

2. Create `config/games/{game_id}.yaml` with game-specific settings

3. Create `docs/games/{game_id}/EXPERIMENTS.md` for tracking experiments

4. Game auto-appears in UI via registry

## Adding a New Algorithm

1. Create `src/algorithms/{algo_id}/` with:
   - `__init__.py` - Auto-registers algorithm
   - `agent.py` - Implements AgentInterface

2. Register with `AlgorithmRegistry.register()`

## Key Files Reference

| Purpose | File |
|---------|------|
| Game registry | `src/games/registry.py` |
| Algorithm registry | `src/algorithms/registry.py` |
| Config loader | `src/utils/config_loader.py` |
| Device management | `src/device/device_manager.py` |
| Trainer | `src/training/trainer.py` |
| UI entry | `scripts/ui.py` |
| CLI training | `scripts/train.py` |
| Model evaluation | `scripts/evaluate.py` |
| Replay viewer | `scripts/watch_replays.py` |
| Snake experiments | `docs/games/snake/EXPERIMENTS.md` |

## CLI Quick Reference

```bash
# UI
python scripts/ui.py              # Game Hub
python scripts/ui.py -g snake     # Direct to Snake menu

# Training
python scripts/train.py -g snake                    # Visual training
python scripts/train.py -g snake --headless         # Headless
python scripts/train.py -g snake --json             # JSON output

# Playing
python scripts/play.py -g snake                     # Watch AI
python scripts/play.py -g snake --human             # Play yourself

# Evaluation
python scripts/evaluate.py -g snake                 # Stats
python scripts/evaluate.py -g snake --json          # JSON output

# Replays
python scripts/watch_replays.py                     # Watch all replays
python scripts/watch_replays.py --list              # List available replays
```
