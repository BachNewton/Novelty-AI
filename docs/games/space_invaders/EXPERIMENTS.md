# Space Invaders Training Experiments

This document tracks experiments and findings for training AI on Space Invaders.

## Game Overview

Space Invaders is a classic fixed shooter where the player defends against descending alien invaders. Key mechanics:

- **Player**: Laser cannon at bottom, moves horizontally, fires upward
- **Invaders**: 5x11 formation (55 total), move as a group, drop when reaching edges
- **Bunkers**: 4 destructible shields for protection
- **Mystery Ship**: Bonus UFO that crosses top of screen
- **Win Condition**: Destroy all invaders
- **Lose Conditions**: Invaders reach bottom OR player loses all 3 lives

## Action Space

6 discrete actions combining movement and firing:
| Action | Movement | Fire |
|--------|----------|------|
| 0 | Stay | No |
| 1 | Stay | Yes |
| 2 | Left | No |
| 3 | Left | Yes |
| 4 | Right | No |
| 5 | Right | Yes |

## State Representation (32 features)

The environment provides a 32-dimensional feature vector:
- Player position and firing state
- Danger zones from enemy projectiles (left/center/right)
- Nearest projectile distances per zone
- Nearest invader positions per zone
- Formation state (alive ratio, position, direction, speed)
- Mystery ship state
- Bunker health ratios
- Strategic features (shooting gap, fire threat, etc.)

## Initial Configuration

```yaml
training:
  episodes: 10000
  batch_size: 64
  learning_rate: 0.0005
  gamma: 0.99
  epsilon_decay: 0.9995
  buffer_size: 200000

rewards:
  kill_bottom: 1.0
  kill_middle: 2.0
  kill_top: 3.0
  mystery_ship: 5.0
  wave_clear: 20.0
  death: -10.0
  game_over: -50.0
```

## Experiments

### Baseline (To Be Run)

**Configuration**: Default settings from `config/games/space_invaders.yaml`

**Expected Behavior**:
- Early episodes: Random movement, occasional lucky kills
- Mid training: Learn to dodge projectiles, target bottom invaders
- Late training: Strategic shooting, mystery ship timing

**Metrics to Track**:
- Average score per episode
- Waves cleared
- Survival time
- Invaders killed per episode

---

## Notes

### Design Decisions

1. **Invader Firing Rules**:
   - Only bottom-most invader per column can fire
   - Max 3 projectiles on screen
   - First projectile aims at player's X position
   - Fire rate increases as invaders decrease

2. **Reward Shaping**:
   - Higher rewards for harder-to-reach invaders (top row = 3x bottom row)
   - Edge invader bonus to encourage slowing formation advance
   - Small survival bonus to reward staying alive

3. **Feature Engineering**:
   - Zone-based danger detection (left/center/right thirds)
   - Normalized values for neural network stability
   - Strategic features like "shooting gap" and "fire threat"

### Potential Improvements

- [ ] Prioritized experience replay for rare events (wave clear, mystery ship)
- [ ] Curriculum learning (start with fewer invaders)
- [ ] Frame stacking for temporal information
- [ ] CNN-based observation of raw game state

---

## Quick Commands

```bash
# Train headless
python scripts/train.py -g space_invaders --headless

# Train with visualization
python scripts/train.py -g space_invaders

# Evaluate model
python scripts/evaluate.py -g space_invaders --json

# Watch AI play
python scripts/play.py -g space_invaders

# Human play
python scripts/play.py -g space_invaders --human
```
