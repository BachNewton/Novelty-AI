# Novelty AI: Space Invaders Deep Dive

Hey Landon! Kyle asked me to brief you on the project. Given your expertise, I'll skip the textbook stuff and get straight to the interesting implementation details.

---

## Novelty AI Hub (Brief)

Multi-game AI training platform. Registry pattern for games/algorithms—drop a new game in `src/games/`, it auto-registers. Same for algorithms in `src/algorithms/`. Vectorized environments, async training, CLI and UI interfaces. Clean separation of concerns.

**Current focus: Space Invaders.** That's where the interesting work is.

---

## Space Invaders: The State Encoding

This is where the domain engineering lives. 32-dimensional feature vector:

```
[0]     Player X (normalized)
[1]     Can fire (no active projectile)

[2-4]   Danger flags per zone (left/center/right)
        → 1.0 if invader projectile in bottom 30% of zone

[5-7]   Nearest projectile distance per zone
        → Normalized distance from screen bottom

[8-10]  Nearest invader X offset per zone
        → Relative to player, normalized to [-1, 1]

[11]    Invaders alive ratio (0-1)

[12]    Formation Y position
        → Proxy for threat level / time pressure

[13]    Formation direction (0=left, 1=right)

[14]    Formation speed (normalized)
        → Inverse of alive count — fewer aliens = faster = danger

[15-16] Mystery ship: active flag + X position

[17-20] Bunker health ratios (4 bunkers)

[21-23] Nearest invader Y distance per zone
        → How close is the front line?

[24]    Time since last shot (normalized, caps at 30 frames)
        → Encourages aggression, prevents camping

[25-26] Player projectile: active flag + Y position
        → Track your bullet's progress

[27]    Shooting gap score
        → Binary: is there a target within 20px of player X?

[28]    Fire threat score
        → Max threat from aligned projectiles, weighted by distance

[29-30] Formation edges (leftmost/rightmost invader X)
        → Formation boundaries for positioning

[31]    Bunker proximity
        → Distance to nearest bunker, inverted (1 = at bunker, 0 = far)
```

**Design decisions worth noting:**

The zone-based encoding (left/center/right) gives the network spatial awareness without position explosions. Each zone is `width/3` pixels.

`Formation speed` being inverse of alive count encodes an important game mechanic—the last few aliens are *fast*. The AI needs to understand this threat escalation.

`Shooting gap score` is a binary heuristic for "should I shoot?" rather than making the network learn column alignment from scratch. Accelerates learning significantly.

`Fire threat score` aggregates incoming danger into a single value. Calculated as:
```python
for proj in invader_projectiles:
    if abs(proj.x - player_x) < player_width:
        dist_ratio = (height - proj.y) / height
        threat = max(threat, 1.0 - dist_ratio)
```
Closer projectiles = higher threat. Only considers projectiles actually aimed at the player.

---

## Action Space

6 discrete actions (movement × firing):

| Action | Movement | Fire |
|--------|----------|------|
| 0 | Stay | No |
| 1 | Stay | Yes |
| 2 | Left | No |
| 3 | Left | Yes |
| 4 | Right | No |
| 5 | Right | Yes |

Single projectile limit (classic Space Invaders rules)—can't fire until current shot resolves. The `can_fire` feature tells the network when firing is even possible.

---

## Reward Structure

```python
rewards = {
    "hit_invader": 10,
    "hit_hard_invader": 20,      # Bottom rows, tougher enemies
    "hit_mystery": 50,           # Rare, high value
    "player_hit": -50,
    "life_lost": -100,
    "wave_complete": 200,
    "step_survival": 0.01,       # Tiny but important
    "game_over": -200,
}
```

The `step_survival` bonus is subtle but crucial. Without it, the network has no gradient signal during "boring" moments—just moving around. With it, survival itself has value, even before kills happen.

`wave_complete` bonus incentivizes actually finishing waves rather than farming easy kills and dying. We want wave completion, not score maximization.

---

## Network Architecture

```
Input:  32 → Dense(256) → ReLU → Dropout(0.1)
           → Dense(256) → ReLU → Dropout(0.1)
           → Dense(128) → ReLU
           → Dense(6)   → Q-values
```

Xavier initialization. Smooth L1 loss (Huber). Adam optimizer at lr=0.001 (fixed, no scheduling yet).

Double DQN with target network sync every 100 steps. Gradient clipping at 1.0.

Dropout at 10%—enough regularization to prevent overfitting to specific enemy patterns without killing capacity.

---

## The Async Training Architecture

This is where the systems engineering gets fun.

**Problem:** Environment stepping is Python-bound (game logic). Network training is PyTorch. Training blocks the main loop.

**Solution:** Background training thread that runs continuously:

```python
class AsyncTrainer:
    def _training_loop(self):
        while not self._stop_event.is_set():
            if len(self.agent.memory) < self.min_buffer_size:
                time.sleep(0.01)
                continue

            for _ in range(self.trains_per_step):  # Default: 4
                loss = self.agent.train_step()
```

**Why it works:** PyTorch releases the GIL during tensor operations. The training thread's CUDA/C++ backend runs truly parallel to Python's environment stepping. Not fake threading—actual parallelism.

**Buffer design matters here.** Pre-allocated numpy arrays, thread-safe with minimal locking:

```python
class FastReplayBuffer:
    def __init__(self, capacity, state_size):
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        # ... pre-allocated, no per-push allocation
```

Also have a `LockFreeReplayBuffer` for single-producer/single-consumer scenarios. Atomic int operations only, accepts slightly stale reads.

**Result:** ~1.8x throughput improvement over synchronous training.

---

## Vectorized Environments

16 parallel environments by default. Single batched forward pass for action selection:

```python
actions = agent.select_actions_batch(states, training=True)  # (16, 32) → (16,)
next_states, rewards, dones, infos = vec_env.step(actions)
agent.store_transitions_batch(states, actions, rewards, next_states, dones)
```

Epsilon decay happens per episode completion, not per step. With 16 envs completing episodes at different times, you get smoother decay curves and more stable exploration.

---

## Interesting Behaviors That Emerge

**Episodes 0-1000:** Random chaos, obviously.

**Episodes 1000-3000:** Learns to shoot. Learns that being under enemies = points. Still walks into projectiles.

**Episodes 3000-5000:** Dodging emerges. Starts tracking the formation's Y position. Uses bunkers as cover (not intentionally taught—emergent from bunker proximity feature + negative hit rewards).

**Episodes 5000+:** Prioritizes low invaders (higher threat). Tracks formation edges to predict movement. Actively hunts mystery ships when they appear.

**The "camping" problem:** Early training often produces a policy that hides in corners and rarely shoots. The `time_since_shot` feature + `step_survival` being tiny (0.01 vs 10 for kills) pushes past this.

**Formation speed awareness:** You can watch the AI's behavior change when invader count drops below ~10. It becomes more evasive, prioritizes survival, takes cleaner shots. The `formation_speed` feature is doing its job.

---

## Current Hyperparameters

```yaml
gamma: 0.99
epsilon_start: 1.0
epsilon_min: 0.01
epsilon_decay: 0.995        # Per episode
batch_size: 64
buffer_size: 100000
target_update_freq: 100     # Steps
learning_rate: 0.001        # Fixed
num_envs: 16
save_interval: 200          # Episodes
```

**What's not implemented yet:**
- Learning rate scheduling (would help late-stage refinement)
- Prioritized experience replay (would accelerate rare-event learning)
- Frame stacking / temporal features (would help with projectile trajectory prediction)

---

## File Map (Space Invaders Specific)

```
src/games/space_invaders/
├── game.py          # Core game logic, physics, collision
├── env.py           # EnvInterface wrapper, state encoding (the 32 features)
├── renderer.py      # Pygame rendering
├── config.py        # SpaceInvadersConfig dataclass
└── __init__.py      # Registry hook

config/games/space_invaders.yaml   # Hyperparameters, rewards, env settings
```

The state encoding lives in `env.py:_get_state()`. That's where most of the domain engineering happens. If you're going to improve something, that's probably where to look.

---

## Open Questions / Where You Could Help

1. **Feature engineering:** Is 32 dimensions overkill? Underkill? The zone-based approach works but feels hand-wavy.

2. **Temporal awareness:** Current state is memoryless. No velocity information for projectiles. Frame stacking? LSTM? Or is the current encoding sufficient?

3. **Reward shaping:** The survival bonus magnitude (0.01) was tuned by feel. Is there a principled way to set this?

4. **Exploration:** Pure epsilon-greedy. Would intrinsic curiosity help in a game this structured?

---

Landon, that's the state of things. The interesting bits are in the state encoding and the async training architecture. The rest is pretty standard DQN with modern improvements.

Let me know what you think. With your background, you'll probably spot inefficiencies I've missed.

— Kyle's codebase
