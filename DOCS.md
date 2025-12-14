# Novelty-AI: Snake AI

A Snake game with a Deep Q-Network (DQN) AI that learns to play. Watch the AI improve in real-time!

## Features

- **Snake Game**: Classic snake game with smooth rendering
- **DQN AI**: Deep Q-Network that learns through reinforcement learning
- **GPU Support**: Automatic detection for NVIDIA (CUDA) and AMD (DirectML) GPUs
- **Real-time Dashboard**: Watch training progress with live charts and hardware monitoring
- **Replay System**: Automatically records and plays back new high-score games
- **Human Play Mode**: Play the game yourself with keyboard controls

## Requirements

- Python 3.10+
- Windows (for full GPU support)
- NVIDIA or AMD GPU (optional, falls back to CPU)

## Installation

```bash
# Clone the repository
git clone https://github.com/BachNewton/Novelty-AI.git
cd Novelty-AI

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU-Specific Installation

**For NVIDIA GPU (CUDA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For AMD GPU (DirectML):**
```bash
pip install torch torch-directml
```

## Usage

### Unified UI (Recommended)

```bash
# Launch the main menu UI
python scripts/ui.py
```

From the menu you can:
- Start training (with option for headless mode)
- Watch AI play (select model from dropdown)
- Play the game yourself
- Watch saved replays

**Controls in UI:**
- ESC: Return to main menu
- H (during training): Toggle headless mode
- Window is resizable

### Train the AI

```bash
# Train with visualization (watch the AI learn)
python scripts/train.py

# Train headless for faster training (replays open on new high scores)
python scripts/train.py --headless

# Train for specific number of episodes
python scripts/train.py --episodes 5000

# Load from checkpoint
python scripts/train.py --load models/model_ep1000.pth
```

### Watch AI Play

```bash
# Watch the trained AI play
python scripts/play.py

# Watch specific model
python scripts/play.py --model models/final_model.pth

# Watch 10 games then stop
python scripts/play.py --games 10
```

### Play Yourself

```bash
python scripts/play_human.py
```

**Controls:**
- Arrow Keys / WASD: Move
- R: Restart
- ESC: Quit

### Watch Replays

```bash
# Watch all replays (best scores first)
python scripts/watch_replays.py

# List available replays
python scripts/watch_replays.py --list

# Watch specific replay
python scripts/watch_replays.py --replay replays/replay_ep500_score42.json

# Watch only the best replay
python scripts/watch_replays.py --best
```

**Controls:**
- SPACE: Pause/Resume
- LEFT/RIGHT: Skip frames
- +/-: Speed up/slow down
- N: Next replay
- ESC: Quit

## Configuration

Edit `config.yaml` to customize training parameters:

```yaml
game:
  grid_width: 20
  grid_height: 20

training:
  episodes: 10000
  batch_size: 64
  learning_rate: 0.001
  epsilon_decay: 0.995

device:
  preferred: "auto"  # "auto", "cuda", "directml", "cpu"
```

## Project Structure

```
Novelty-AI/
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
├── scripts/
│   ├── ui.py               # Unified UI entry point
│   ├── train.py            # Training script (CLI)
│   ├── play.py             # Watch AI play (CLI)
│   ├── play_human.py       # Human play mode (CLI)
│   └── watch_replays.py    # Replay viewer (CLI)
├── src/
│   ├── game/               # Snake game
│   ├── ai/                 # DQN agent
│   ├── device/             # GPU detection
│   ├── visualization/      # Dashboard, menus & effects
│   └── utils/              # Utilities
├── models/                  # Saved checkpoints
└── replays/                 # Recorded games
```

## How It Works

1. **State Encoding**: The game state is encoded as 11 features:
   - Danger detection (straight, right, left)
   - Current direction (one-hot)
   - Food location relative to head

2. **Deep Q-Network**: A neural network predicts Q-values for each action:
   - Input: 11 features
   - Hidden: 256 → 256 → 128 neurons
   - Output: 3 Q-values (straight, turn right, turn left)

3. **Experience Replay**: Stores past experiences and samples randomly for stable learning

4. **Target Network**: A separate network for computing targets, updated periodically

## Expected Results

- ~500 episodes: Agent learns basic wall avoidance
- ~2000 episodes: Agent starts eating food consistently
- ~5000 episodes: Agent scores 10-20 points regularly
- ~10000 episodes: Agent can reach 30-50+ points
