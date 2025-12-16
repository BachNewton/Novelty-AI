# Tetris AI Experiments Log

This document records experiments tried during development and their outcomes, to inform future decisions.

---

## Initial Implementation (Current)

**Algorithm:** PPO (Proximal Policy Optimization)

**Rationale:** PPO was chosen over DQN for Tetris because:
- Better at handling delayed rewards (piece placement affects future line clears)
- More stable training on complex state spaces
- Works well with GPU acceleration

**State Representation:** 86-dimensional engineered feature vector:
- Column heights (10) - normalized by board height
- Height differences between adjacent columns (9) - detects bumpiness
- Holes per column (10) - empty cells below filled cells
- Current piece one-hot (7)
- Next pieces one-hot (5 x 7 = 35) - full preview queue
- Held piece one-hot + empty (8)
- Can hold flag (1)
- Piece position/rotation (3) - normalized
- Game progress: lines cleared, level, max height (3)

**Action Space:** 7 actions
1. Move Left
2. Move Right
3. Rotate Clockwise
4. Rotate Counter-clockwise
5. Soft Drop (move down 1 cell)
6. Hard Drop (instant placement)
7. Hold (swap with held piece)

**Reward Structure:**
- Single line: +100
- Double: +300
- Triple: +500
- Tetris (4 lines): +800
- Step survival: +0.01
- Height penalty: -0.1 × max_height
- Game over: -100

---

## Design Decisions

### Why Engineered Features Over Pixels

For Tetris, hand-crafted features significantly outperform pixel-based observations:

1. **Column heights and holes** directly encode what makes a good Tetris board
2. **Piece preview** is crucial for planning (pixel representation would need to learn this)
3. **State size** is 86 vs 200×100×3 for pixels - orders of magnitude smaller
4. **Training efficiency** - converges in hours vs days

### Why PPO Over DQN

1. **Delayed rewards** - Placing pieces optimally affects line clears 10+ moves later
2. **On-policy learning** - PPO's on-policy nature handles non-stationary targets better
3. **Stability** - PPO's clipped objective prevents catastrophic updates
4. **Parallelization** - Easy to scale with vectorized environments on GPU

### Why 7-Bag Randomizer

Modern Tetris uses 7-bag randomization:
- All 7 pieces shuffled, dealt out, repeat
- Prevents "droughts" (long waits for I-piece)
- More fair and learnable than pure random

---

## Hyperparameter Recommendations

### PPO Settings (config/games/tetris.yaml)

```yaml
learning_rate: 0.0003      # Standard PPO rate
gamma: 0.99               # High discount for long-term planning
gae_lambda: 0.95          # GAE for variance reduction
clip_epsilon: 0.2         # Standard PPO clip
entropy_coef: 0.01        # Encourage exploration
n_steps: 2048             # Steps before update
batch_size: 64            # Mini-batch size
n_epochs: 10              # PPO epochs per rollout
```

### Training Expectations

- **CPU training:** ~100 episodes/hour (good for testing)
- **GPU training (RTX 4070):** ~1000 episodes/hour
- **Convergence:** Expect meaningful learning by 10,000 episodes
- **Full training:** 50,000+ episodes for strong play

---

## Future Experiment Ideas

### Architectural Improvements

1. **Attention mechanism** - Attend to preview queue pieces differently based on current board state

2. **Hierarchical actions** - First select column, then rotation, reducing action space complexity

3. **Frame stacking** - Include last N board states for temporal awareness

4. **Tree search integration** - MCTS or beam search for look-ahead planning

### Reward Shaping Ideas

1. **T-spin detection** - Extra reward for T-spin clears (advanced technique)

2. **Back-to-back bonus** - Reward consecutive Tetrises/T-spins

3. **Combo counting** - Reward for consecutive line clears

4. **Well construction** - Small reward for maintaining a well for I-pieces

### State Representation Ideas

1. **Danger level** - Distance from top of board

2. **Clear potential** - How many filled cells in each row

3. **Surface roughness** - More nuanced bumpiness measure

4. **Reachable cells** - Similar to Snake's flood fill

---

## Known Limitations

1. **No T-spins** - Current AI won't learn T-spin techniques without specific reward

2. **No back-to-back** - Doesn't specifically optimize for consecutive specials

3. **Simple reward** - Quadratic line clear reward may not match competitive scoring

4. **Single piece lookahead** - Uses preview but doesn't explicitly plan sequences

---

## Benchmark Targets

| Metric | Beginner | Intermediate | Expert |
|--------|----------|--------------|--------|
| Lines/Game | 10 | 50 | 150+ |
| Avg Score | 1,000 | 10,000 | 50,000+ |
| Max Combo | 2 | 4 | 8+ |
| Tetris Rate | 10% | 30% | 50%+ |

Current AI target: Intermediate level within 50,000 training episodes.
