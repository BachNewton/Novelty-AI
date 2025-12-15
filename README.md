# Novelty AI

A spin-off of the famous "Novelty Games", now we're doing something new. We'll have AIs make AIs!

**Novelty AI** is a multi-game AI training hub where you can watch AI learn to play classic games from scratch. Train, watch, and play alongside AI in a unified interface.

## Features

- **Multi-Game Hub** - Train AI on multiple games from one interface
- **Real-time Training Dashboard** - Watch AI learn with live charts and game preview
- **Dual Access** - Use the visual UI or command-line tools
- **Hardware Monitoring** - See GPU/CPU utilization during training
- **Replay System** - Save and replay high-scoring games

## Games

| Game | Status | Description |
|------|--------|-------------|
| Snake | Available | Classic snake game - teach AI to grow long! |
| Tetris | Coming Soon | Block-stacking puzzle |
| Pong | Coming Soon | Classic paddle game |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Game Hub
python scripts/ui.py

# Or go directly to Snake
python scripts/ui.py --game snake
```

## Command Line Tools

All features are available via CLI for automation and server use:

```bash
# Training
python scripts/train.py --game snake              # Visual training
python scripts/train.py --game snake --headless   # Headless (faster)

# Watch AI Play
python scripts/play.py --game snake               # Watch trained AI
python scripts/play.py --game snake --human       # Play yourself

# Evaluate Model
python scripts/evaluate.py --game snake --json    # Get performance stats
```

## How It Works

Novelty AI uses Deep Q-Learning (DQN) with:
- **20-feature state encoding** - Snake position, food direction, danger detection
- **Double DQN** - Reduces Q-value overestimation
- **Parallel Environments** - Train on multiple games simultaneously
- **Experience Replay** - Learn from past experiences

## Screenshots

The training dashboard shows:
- Live game preview (toggleable for faster training)
- Score over time chart
- Epsilon decay (exploration rate)
- Episode progress
- Hardware utilization

## Configuration

Settings are in `config/`:
- `config/default.yaml` - Global settings
- `config/games/snake.yaml` - Snake-specific settings

## Project Structure

```
Novelty-AI/
├── scripts/         # Entry points (ui.py, train.py, play.py)
├── src/
│   ├── core/        # Abstract interfaces
│   ├── games/       # Game implementations
│   ├── algorithms/  # AI algorithms (DQN)
│   ├── training/    # Training infrastructure
│   └── visualization/  # UI components
├── config/          # Configuration files
├── models/          # Trained models
└── replays/         # Saved replays
```

## Development

See [CLAUDE.md](CLAUDE.md) for development instructions, including:
- How to add new games
- How to add new algorithms
- AI iteration workflow
- CLI reference

## Experiment Notes

Learnings from training experiments are documented in `docs/games/snake/EXPERIMENTS.md`.

## License

MIT
