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

## What Works Well

The current proven configuration:
- **Observation:** 11-feature hand-crafted vector (distances, danger flags, direction)
- **Network:** Simple 3-layer MLP (128 -> 128 -> 4)
- **Algorithm:** Double DQN (reduces Q-value overestimation)
- **Replay:** Uniform sampling, 100k buffer
- **Exploration:** Fast epsilon decay (0.995) reaching low exploration by ~1000 episodes
- **Rewards:** Food (+10), Death (-10), Step penalty (-0.01)

This configuration reaches avg score ~20 within ~1000 episodes.

---

## Future Experiment Ideas

If revisiting improvements:
1. **Curriculum learning:** Start with smaller grid, gradually increase
2. **Self-play / population training:** Multiple agents with different exploration strategies
3. **Intrinsic motivation:** Reward novel states to encourage exploration
4. **Frame stacking:** Give agent memory of recent positions
5. **LSTM/GRU:** Recurrent network for temporal patterns

Remember: Simple often beats complex for simple problems. Benchmark against the current working solution before committing to changes.
