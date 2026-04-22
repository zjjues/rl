You previously helped me implement the "Intent-driven MAPPO (I-MAPPO)" algorithm for a continuous UAV scheduling environment. The code runs end-to-end, but my experiment logs show severe optimization instability:

- The `critic_loss` is extremely high (ranging from 80 to 270+).
- The `episode_return` is stuck in negative regimes (around -20 to -30) and does not show steady learning.
- Entropy varies wildly across seeds.
- Turning on `standardise_rewards` caused performance to drop further.

This strongly indicates that the Reward signals (both `R_env` and the potential difference `R_intent`) have extreme spikes/outliers that are exploding the Critic targets and destabilizing the Advantage calculation. Furthermore, to generate plots for my academic paper, I need to track specific metrics like "collision counts".

Please provide the updated Python/PyTorch code for the following **3 critical modifications**:

### 1. Reward Scaling and Clipping (In the Environment / Reward Logic)
Please rewrite the reward calculation function. 
- **Scale down Extrinsic Reward:** Ensure that the raw environment reward components (distance penalty, energy penalty, collision penalty) are scaled down. The total `R_env` per step should ideally fall within the `[-1.0, 1.0]` range.
- **Clamp the Potential Reward:** The potential-based shaping term `R_intent = gamma * Phi(s_{t+1}, I) - Phi(s_t, I)` is likely producing massive spikes. Please explicitly clamp this value before adding it to `R_env`. 
  *Code requirement:* `R_intent = torch.clamp(gamma * phi_next - phi_curr, min=-1.0, max=1.0)`
- **Total Reward:** `r_i = R_env + eta * R_intent`.

### 2. Metric Tracking for the Academic Paper (In the Environment step function)
I need to plot "Early-stage Collision Rates" for my paper. 
- In the environment's `step()` function, please add logic to detect if the distance between any two UAVs is less than `D_safe` (e.g., `D_safe = 0.1` in normalized coordinates).
- If a collision occurs, increment a `collision_count` counter for the current episode.
- Return this metric inside the `info` dictionary: `info = {'collision': True/False}`.
- In the main training loop, accumulate these collisions per episode and log `episode_collisions` alongside `episode_return`.

### 3. Stabilizing the PPO Update Loop (In the I-MAPPO Algorithm class)
Since the Critic is struggling to fit the value function, please adjust the PPO update logic to be less aggressive and more stable:
- **Reduce PPO Epochs:** Change `ppo_epochs` (or optimization iterations per rollout) from the default (e.g., 10) down to `4` or `5`.
- **Tighten PPO Clip:** Change the `eps_clip` parameter from `0.2` to `0.1` to prevent the Actor from changing too fast while the Critic is still confused.
- **Gradient Clipping:** Ensure that Gradient Clipping (e.g., `torch.nn.utils.clip_grad_norm_`) is applied to BOTH the Actor and the Critic with a `max_norm=0.5` before `optimizer.step()`.
- **Value Loss Clipping (Optional but recommended):** Implement Value Loss Clipping for the Critic (similar to standard PPO implementations) to restrict how much the value network can change per update.

Please output the specific, updated code snippets for:
1. The `step()` and reward calculation functions in `uav_scheduling_env.py`.
2. The PPO update function (specifically the Actor/Critic loss calculations and gradient clipping) in `imappo.py`.
3. The tracking logic in the main training loop.