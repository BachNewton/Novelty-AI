# Experiments Log

This document records experiments tried during development and their outcomes, to inform future decisions.

---

## CNN Observation Mode (Removed)

**What:** Instead of hand-crafted 11-feature vector observations, used a 4-channel grid representation processed by a CNN.

**Implementation:**
- `CNNDQNetwork` class with convolutional layers
- Grid observation: 4 channels (snake head, snake body, food, walls)
- Config option: `observation_type: "grid"`

**Results:**
- Training was significantly slower due to larger network
- Required ~50,000 episodes to plateau vs ~1,000 with vector mode
- Final performance was worse (avg score ~15 vs ~20)
- CNN struggled to learn spatial relationships that hand-crafted features capture directly

**Conclusion:** For a simple game like Snake, hand-crafted features that encode domain knowledge (distance to food, danger detection, etc.) outperform learned representations. CNN approach is better suited for games where the optimal features aren't obvious.

**Files Removed:**
- `src/ai/cnn_network.py`
- Grid observation code in `snake_env.py`

---

## Dueling DQN Architecture (Removed)

**What:** Separate value and advantage streams in the network, combined as Q(s,a) = V(s) + A(s,a) - mean(A).

**Implementation:**
- `DuelingDQNNetwork` class
- Config option: `use_dueling: true`

**Results:**
- Marginal improvement in some runs, but inconsistent
- Added complexity without clear benefit for this problem size
- The small action space (4 directions) doesn't benefit much from advantage decomposition

**Conclusion:** Dueling architecture provides more benefit in environments with many actions where understanding state value separately from action advantages helps. Snake's 4-action space is too simple.

**Files Removed:**
- `DuelingDQNNetwork` class from `dqn_network.py`

---

## Prioritized Experience Replay (Removed)

**What:** Sample experiences based on TD error magnitude rather than uniformly, focusing learning on surprising transitions.

**Implementation:**
- `PrioritizedReplayBuffer` class with sum-tree data structure
- Importance sampling weights to correct bias
- Config options: `use_per: true`, `per_alpha`, `per_beta_start`

**Results:**
- Increased memory usage and computation overhead
- No clear improvement in final performance
- The simple replay buffer with uniform sampling worked well enough

**Conclusion:** PER helps most in sparse reward environments or when certain transitions are rare but important. Snake's reward structure (food = +10, death = -10, small step penalty) is dense enough that uniform sampling captures the important transitions.

**Files Removed:**
- `PrioritizedReplayBuffer` class from `replay_buffer.py`
- PER-related config options

---

## Reward Shaping Experiments (Disabled, Config Only)

**What:** Additional reward signals beyond food/death:
- `approach_food`: Positive reward for moving closer to food
- `retreat_food`: Negative reward for moving away from food
- `length_bonus_factor`: Bonus based on snake length

**Results:**
- Approach/retreat rewards caused oscillation behavior - snake would move back and forth near food
- Length bonus didn't improve learning
- Simple sparse rewards (food + death + small step penalty) worked best

**Conclusion:** For Snake, the natural reward structure is sufficient. Shaping rewards introduced unintended behaviors.

**Current Config:** All shaping rewards set to 0.0 (disabled but code remains for potential future experiments)

---

## Enhanced State Representation (Current)

**What:** Expanded from 11 to 20 features to address plateau at ~20 score due to self-collision in tight spaces.

**Problem:** Agent had no spatial awareness - only 1-step danger detection, leading to self-trapping at higher scores.

**New Features (9 additional):**
- Multi-step danger look-ahead (6 features): Danger 2 and 3 steps ahead in each relative direction
- Flood fill / reachable cells (3 features): BFS count of reachable cells from each possible move, normalized by grid size

**Hyperparameter Adjustments:**
- `epsilon_decay`: 0.995 -> 0.998 (slower exploration decay)
- `epsilon_min`: 0.01 -> 0.02 (slightly higher minimum exploration)
- Recommended training: 5000 episodes

**Rationale:** Flood fill directly solves the trap detection problem - agent can now see "if I go left, I can only reach 10 cells (trapped!) vs 50 cells if I go right"

**Results:**
- Previous plateau: ~20 avg score at ~1000 episodes
- New plateau: ~30 avg score at ~2000 episodes
- **50% improvement in average score**
- Agent now avoids obvious traps but still plateaus - likely needs more sophisticated planning

**Status:** Success - kept as current configuration. New plateau at ~30 suggests further improvements possible.

---

## Previous Proven Configuration

The previous configuration (before enhanced state):
- **Observation:** 11-feature hand-crafted vector (distances, danger flags, direction)
- **Network:** Simple 3-layer MLP (256 -> 256 -> 128 -> 3)
- **Algorithm:** Double DQN (reduces Q-value overestimation)
- **Replay:** Uniform sampling, 100k buffer
- **Exploration:** Fast epsilon decay (0.995) reaching low exploration by ~1000 episodes
- **Rewards:** Food (+10), Death (-10), Step penalty (-0.01)

This configuration reached avg score ~20 within ~1000 episodes but plateaued due to self-collision.

---

## Future Experiment Ideas

### To Address ~30 Score Plateau

1. **Frame stacking:** Stack last N states (e.g., 4) to give agent memory of recent positions. Helps detect movement patterns and avoid revisiting areas.

2. **LSTM/GRU network:** Replace or augment MLP with recurrent layers. Can learn temporal patterns like "I've been circling this area" without explicit feature engineering.

3. **Curriculum learning:** Start training on smaller grid (e.g., 10x10), then progressively increase to 20x20. Agent learns tight-space navigation faster on smaller grids.

4. **Hamiltonian path heuristics:** Add features that encourage space-filling patterns (e.g., distance to tail, whether path forms a cycle). Expert Snake play often follows Hamiltonian-like paths.

5. **Increased flood fill depth:** Current `max_depth=20`. Increasing to 30-50 could help with longer-term planning at the cost of slightly more computation.

6. **Tail-chasing feature:** Add feature indicating direction to own tail. Following tail is a safe fallback strategy when no food is nearby.

### Other Ideas (Lower Priority)

7. **Self-play / population training:** Multiple agents with different exploration strategies
8. **Intrinsic motivation:** Reward novel states to encourage exploration
9. **A* pathfinding features:** Add feature for shortest safe path length to food

Remember: Simple often beats complex for simple problems. Benchmark against the current working solution before committing to changes.
