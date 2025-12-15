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
│   ├── games/                    # Game implementations
│   │   ├── registry.py           # GameRegistry
│   │   └── snake/                # Snake game
│   ├── algorithms/               # AI algorithms
│   │   ├── registry.py           # AlgorithmRegistry
│   │   └── dqn/                  # Deep Q-Network
│   ├── training/                 # Training infrastructure
│   │   ├── trainer.py            # Generic trainer
│   │   └── vec_env.py            # Vectorized environments
│   └── visualization/            # UI components
│       ├── game_hub.py           # Game selection hub
│       ├── game_menu.py          # Per-game menu
│       └── dashboard.py          # Training dashboard
├── scripts/
│   ├── ui.py                     # UI entry point
│   ├── train.py                  # CLI training
│   ├── play.py                   # CLI AI/human play
│   └── evaluate.py               # Model evaluation
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
| Trainer | `src/training/trainer.py` |
| UI entry | `scripts/ui.py` |
| CLI training | `scripts/train.py` |
| Model evaluation | `scripts/evaluate.py` |
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
```
