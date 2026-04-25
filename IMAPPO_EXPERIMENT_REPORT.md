# I-MAPPO Experiment Report

## 1. Overview

This document summarizes all work completed so far for the `Intent-driven MAPPO (I-MAPPO)` task described in `rl.md`, including:

- algorithm design and implementation
- code changes to the current EPyMARL repository
- environment construction and integration
- experiment configurations used
- experiment results collected so far
- trend analysis and current conclusions
- recommended next steps

The target of the work was to implement a continuous-action multi-agent PPO variant with:

- intent-conditioned decentralized actors
- a centralized critic with cross-attention over agent features
- potential-based reward shaping with intent guidance
- a CTDE training loop


## 2. Task Interpretation

The original repository is an EPyMARL-style research framework with discrete-action MAPPO/IPPO and related algorithms. The `rl.md` requirement, however, asked for a materially different algorithm:

- continuous action policy instead of categorical policy
- explicit intent vector input
- explicit per-dimension action mask logic
- centralized value function with cross-attention
- potential-based shaping term `Phi(s, I)`

This meant the work was not a parameter tweak to existing MAPPO, but a new continuous-action training path.


## 3. Algorithm Principle

### 3.1 Core Idea

`I-MAPPO` extends MAPPO by injecting a high-level intent vector `I` into both:

- the actor, as an extra conditioning variable
- the critic, as the query of an attention mechanism over agent features

This is intended to let the policy and value function align low-level control with a higher-level tactical objective.

### 3.2 Actor

Each agent uses a decentralized actor:

- input: `concat(o_i, I)`
- backbone: `MLP(256 -> 256 -> 128, ReLU)`
- heads:
  - `action_mean`
  - `action_log_std`

The actor outputs a Gaussian policy over continuous actions.

Action masking is handled by:

- multiplying masked action means by the binary action mask
- collapsing masked dimensions to very low variance so masked axes stay near safe zero control

### 3.3 Critic

The centralized critic uses:

- global state `s`
- intent vector `I`
- hidden features of all agents `H_o`

The critic computes:

1. `Q = W_q I`
2. `K = W_k H_o`
3. `V = W_v H_o`
4. attention weights `alpha = softmax(QK^T / sqrt(d_k))`
5. context `C_global = alpha V`
6. `Value = MLP([s, C_global])`

This lets the value function attend to the agents most relevant to the current intent.

### 3.4 Reward Shaping

The environment provides extrinsic reward `R_env`.

We additionally compute:

- `Phi(s, I) = -MSE(f(s), I)`
- `R_intent = gamma * Phi(s_{t+1}, I) - Phi(s_t, I)`
- total reward per agent:
  - `r_i = R_env + eta * R_intent`

This preserves the potential-based shaping structure while letting intent affect optimization pressure.

### 3.5 Training Principle

The implementation uses a CTDE PPO loop:

- decentralized action selection during rollout
- centralized value estimation for advantage computation
- GAE for advantage estimation
- PPO clipped objective for actor update
- MSE loss for critic update


## 4. Experiment Principle

The experiments were staged in four phases:

1. **Implementation validation**
   - Ensure the continuous policy, critic, reward shaping and PPO update run end-to-end.

2. **Framework integration**
   - Integrate I-MAPPO into `src/main.py --config=...` workflow.

3. **Environment alignment**
   - Replace the initial mock environment with a continuous UAV scheduling task matching `rl.md` dimensions.

4. **Ablation and stability sweeps**
   - Explore `eta`, `intent_dim`, entropy regularization, learning rate, rollout length, critic stabilization, and potential update strategies.

The goal of the experiments so far was not final benchmark performance, but:

- verify correctness
- identify useful hyperparameter directions
- find the current best baseline
- isolate likely instability sources


## 5. Code Changes

### 5.1 New Files

- [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:1)
  - full I-MAPPO implementation
  - intent-conditioned Gaussian actor
  - cross-attention critic
  - potential model
  - rollout buffer
  - training loop
  - environment adapter helpers

- [src/config/algs/imappo.yaml](/home/cring/rl/epymarl/src/config/algs/imappo.yaml:1)
  - algorithm config for I-MAPPO

- [src/envs/uav_scheduling_env.py](/home/cring/rl/epymarl/src/envs/uav_scheduling_env.py:1)
  - custom continuous UAV scheduling environment
  - matches `rl.md` target dimensions:
    - `N=4`
    - `obs_dim=18`
    - `state_dim=72`
    - `action_dim=3`

### 5.2 Modified Files

- [src/run.py](/home/cring/rl/epymarl/src/run.py:22)
  - added `imappo_learner` dispatch path

- [src/config/envs/gymma.yaml](/home/cring/rl/epymarl/src/config/envs/gymma.yaml:1)
  - added `continuous_actions` so Sacred accepts the config

- [src/envs/__init__.py](/home/cring/rl/epymarl/src/envs/__init__.py:1)
  - imports and registers the UAV scheduling environment

### 5.3 Important Implementation Milestones

#### I-MAPPO implementation

Main code sections in [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:1):

- config: [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:16)
- actor: [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:67)
- cross-attention critic: [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:139)
- potential network: [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:185)
- PPO update: [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:307)
- experiment runner: [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:566)
- config conversion from EPyMARL args: [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:646)

#### Potential update strategy extension

Later in the process, `Phi` update strategies were added:

- `normal`
- `slow`
- `frozen`

This logic is implemented in [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:255) and [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:395).


## 6. Environment Design

### 6.1 Why a New Environment Was Needed

The local PettingZoo MPE environment available during testing (`pz-mpe-simple-spread-v3`) did not match the requested problem dimensions:

- default `n_agents = 3`
- action dimension different from the requested `3`

This meant it was suitable only for a smoke test, not for the requested problem specification.

### 6.2 UAV Scheduling Environment

The custom environment `uav-scheduling-v0` was introduced to match the requested shape constraints.

State semantics are simplified but structured:

- each UAV has position, velocity, energy and pending-task variables
- each UAV has a target location
- observations include local state plus nearby-agent features
- reward includes:
  - target-tracking term
  - action penalty
  - collision penalty
  - task completion reward
  - energy term

This is still a toy research environment, not a realistic simulator, but it is consistent with the requested experiment setup.


## 7. Validation and Smoke Tests

### 7.1 Static Validation

The following passed:

- `python -m py_compile src/imappo.py`
- `python -m py_compile src/imappo.py src/run.py`
- `python -m py_compile src/envs/uav_scheduling_env.py src/envs/__init__.py src/imappo.py src/run.py`

### 7.2 Initial Main-Entrypoint Smoke Runs

The I-MAPPO path was validated through `src/main.py` both with:

- the internal mock environment
- a real PettingZoo continuous environment
- the custom `uav-scheduling-v0` environment

This established that:

- Sacred config loading works
- main EPyMARL entrypoint works
- training loop runs end-to-end
- results are logged to `results/sacred`


## 8. Experiment Configurations Used

All runs below used the `uav-scheduling-v0` environment unless otherwise noted.

Common baseline template:

```bash
python src/main.py --config=imappo --env-config=gymma with \
  max_episodes=30 \
  rollout_length=16 \
  batch_size=16 \
  env_args.key='uav-scheduling-v0' \
  env_args.time_limit=16 \
  imappo_n_agents=4 \
  imappo_obs_dim=18 \
  imappo_state_dim=72 \
  imappo_action_dim=3 \
  action_low=-1.0 \
  action_high=1.0
```

The following axes were explored:

- `eta`
- `intent_dim`
- `entropy_coef`
- `lr / critic_lr / potential_lr`
- `rollout_length / batch_size`
- multi-seed reproducibility
- `Phi` update strategy


## 9. Experiment Results

### 9.1 Reward Shaping Coefficient Ablation

| Run ID | Name | eta | intent_dim | episode_return | return_mean | critic_loss | entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| 3 | `imappo_eta0` | 0.0 | 8 | -24.80 | -1.550 | 19.07 | 5.69 |
| 5 | `imappo_eta05` | 0.5 | 8 | -31.24 | -1.952 | 16.29 | 5.28 |
| 4 | `imappo_eta1` | 1.0 | 8 | -19.61 | -1.225 | 17.99 | 4.57 |

#### Interpretation

- `eta=1.0` clearly outperformed `eta=0.0` and `eta=0.5`
- shaping was useful in this environment
- moderate shaping (`0.5`) was unexpectedly the worst, suggesting the shaping signal needs to be strong enough to matter


### 9.2 Intent Dimension Sensitivity

| Run ID | Name | eta | intent_dim | episode_return | return_mean | critic_loss | entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| 6 | `imappo_intent4` | 1.0 | 4 | -19.63 | -1.227 | 30.25 | -0.90 |
| 4 | `imappo_eta1` | 1.0 | 8 | -19.61 | -1.225 | 17.99 | 4.57 |
| 7 | `imappo_intent16` | 1.0 | 16 | -26.09 | -1.631 | 6.25 | 4.15 |

#### Interpretation

- `intent_dim=4` and `8` were much better than `16`
- larger intent vectors did not help and likely acted as noise in this toy environment
- `intent_dim=4` showed a warning sign: entropy collapsed strongly, so its apparent return advantage was not obviously safer than `8`


### 9.3 Confirmation Run for a Short-Term Winner

| Run ID | Name | eta | intent_dim | max_episodes | episode_return | return_mean | critic_loss | entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | `imappo_confirm_eta1_int4` | 1.0 | 4 | 50 | -30.33 | -1.896 | 16.14 | 1.20 |

#### Interpretation

- the short-run promise of `intent_dim=4` did not hold up under a longer run
- this suggested that short horizon improvements were not yet reliable


### 9.4 Entropy Regularization Sweep

| Run ID | Name | entropy_coef | lr | episode_return | return_mean | critic_loss | entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| 11 | `imappo_ent0005` | 0.0005 | 3e-4 | -25.21 | -1.576 | 20.45 | 3.76 |
| 9 | `imappo_ent005` | 0.005 | 3e-4 | -32.08 | -2.005 | 24.43 | 4.84 |
| 10 | `imappo_ent01` | 0.01 | 3e-4 | -30.59 | -1.912 | 7.32 | 4.56 |

#### Interpretation

- stronger entropy regularization did not improve training
- `entropy_coef=0.0005` was the best of the tested values
- the task did not appear exploration-starved under the current setup


### 9.5 Learning Rate and Rollout Sweep

| Run ID | Name | lr | critic_lr | rollout | batch | episode_return | return_mean | critic_loss | entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | `imappo_lr1e4` | 1e-4 | 1e-4 | 16 | 16 | -19.10 | -1.194 | 12.83 | 3.54 |
| 11 | `imappo_ent0005` | 3e-4 | 3e-4 | 16 | 16 | -25.21 | -1.576 | 20.45 | 3.76 |
| 14 | `imappo_lr5e4` | 5e-4 | 5e-4 | 16 | 16 | -24.02 | -1.501 | 3.97 | 3.54 |
| 12 | `imappo_roll32` | 3e-4 | 3e-4 | 32 | 32 | -26.59 | -1.930 | 7.92 | 1.70 |

#### Interpretation

- `lr=1e-4` gave the best overall return in this sweep
- increasing `rollout_length` to `32` did not help
- `lr=5e-4` reduced critic loss but still underperformed in return


### 9.6 Multi-Seed Confirmation of Current Best Baseline

Configuration:

- `eta=1.0`
- `intent_dim=8`
- `entropy_coef=0.0005`
- `lr=1e-4`
- `critic_lr=1e-4`
- `potential_lr=1e-4`
- `rollout_length=16`
- `batch_size=16`

| Run ID | Seed | episode_return | return_mean | critic_loss | entropy | potential_loss |
|---|---:|---:|---:|---:|---:|---:|
| 15 | 23 | -25.67 | -1.604 | 93.74 | 4.40 | 1.25 |
| 16 | 22 | -28.25 | -1.766 | 76.78 | 0.26 | 2.29 |
| 17 | 21 | -30.09 | -1.881 | 80.98 | 2.69 | 1.59 |

Aggregate:

- `episode_return mean = -28.01`
- `episode_return std = 1.82`
- `return_mean mean = -1.750`
- `return_mean std = 0.113`
- `critic_loss mean = 83.83`
- `critic_loss std = 7.21`
- `entropy mean = 2.45`
- `entropy std = 1.70`

#### Interpretation

- return magnitude was reproducible enough to rule out a one-off lucky seed
- however, critic loss remained very high
- entropy varied strongly across seeds, indicating remaining instability in policy dynamics


### 9.7 Critic Stabilization Attempts

| Run ID | Name | critic_lr | std_rewards | std_returns | episode_return | return_mean | critic_loss |
|---|---|---:|---|---|---:|---:|---:|
| 13 | `imappo_lr1e4` | 1e-4 | False | False | -19.10 | -1.194 | 12.83 |
| 20 | `imappo_critic5e5` | 5e-5 | False | False | -24.61 | -1.538 | 113.89 |
| 18 | `imappo_critic3e5` | 3e-5 | False | False | -38.23 | -2.389 | 278.41 |
| 19 | `imappo_std_both` | 1e-4 | True | True | -39.60 | -2.475 | 173.91 |

#### Interpretation

- lowering critic learning rate did not help
- reward/return standardization also hurt performance
- this suggested the main issue was not simply “critic learning too fast”


### 9.8 Potential Update Strategy Experiments

| Run ID | Name | Phi update | interval | episode_return | return_mean | critic_loss | entropy |
|---|---|---|---:|---:|---:|---:|---:|
| 13 | `imappo_lr1e4` | normal | - | -19.10 | -1.194 | 12.83 | 3.54 |
| 21 | `imappo_phi_slow8` | slow | 8 | -20.40 | -1.275 | 27.52 | 3.77 |
| 22 | `imappo_phi_frozen` | frozen | - | -20.93 | -1.308 | 21.98 | 3.66 |

#### Interpretation

- slowing or freezing `Phi` did not improve over the current best baseline
- this reduced confidence that non-stationary potential updates are the dominant issue


## 10. Text-Based Chart Analysis

Because the work was conducted in terminal mode, the analysis is summarized in text-chart form below.

### 10.1 Reward Shaping Trend

Higher is better:

```text
eta=0.0   : #########################   (-24.80)
eta=0.5   : ############################### (-31.24)
eta=1.0   : ####################      (-19.61)
```

Observation:

- `eta=1.0` is the best of the tested shaping strengths.

### 10.2 Intent Dimension Trend

Higher is better:

```text
intent=4  : ####################      (-19.63)
intent=8  : ####################      (-19.61)
intent=16 : ########################## (-26.09)
```

Observation:

- low-to-medium intent dimension works better than large intent dimension.

### 10.3 Entropy Coefficient Trend

Higher is better:

```text
0.0005 : #########################   (-25.21)
0.005  : ################################ (-32.08)
0.01   : ############################## (-30.59)
```

Observation:

- stronger entropy regularization degraded return.

### 10.4 Learning Rate Trend

Higher is better:

```text
1e-4 : ###################        (-19.10)
3e-4 : #########################  (-25.21)
5e-4 : ########################   (-24.02)
```

Observation:

- `1e-4` was the best tested learning rate.


## 11. Current Best Configuration

Based on experiments so far, the best-performing configuration is:

```yaml
eta: 1.0
intent_dim: 8
entropy_coef: 0.0005
lr: 0.0001
critic_lr: 0.0001
potential_lr: 0.0001
rollout_length: 16
batch_size: 16
standardise_rewards: False
standardise_returns: False
potential_update_mode: normal
```

This corresponds to the strongest single-seed result found in the experiments above.


## 12. Current Analysis

### 12.1 What is working

- the continuous I-MAPPO pipeline is implemented and fully runnable
- the repository main entrypoint supports the new algorithm
- the custom UAV environment matches the requested dimensions
- reward shaping helps when `eta` is sufficiently large
- the critic cross-attention version can train end-to-end without interface failure
- there is a reproducible return regime across seeds

### 12.2 What is not working well enough

- training is still not genuinely stable
- multi-seed critic loss remains high
- entropy varies strongly between seeds
- simple critic stabilization tricks did not solve the issue
- freezing or slowing `Phi` did not outperform the baseline


## 13. Stage-2 Continuous Optimization Progress

After the first round of ablations, the work shifted from "can the algorithm run?" to "can the optimization become stable enough to support useful paper plots?".

The second-stage work added four groups of changes:

### 13.1 PPO Stabilization Upgrades

The training path in [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:1) was further upgraded with:

- clipped intrinsic reward:
  - `R_intent = clamp(gamma * Phi(s_{t+1}, I) - Phi(s_t, I), -1, 1)`
- tighter PPO clip:
  - `eps_clip = 0.1`
- actor and critic gradient clipping:
  - `max_grad_norm = 0.5`
- value loss clipping for the critic
- linear decay of:
  - `eta`
  - `entropy_coef`
- continuous actor changed from
  - Gaussian sampling + hard action clipping
  to
  - `tanh`-squashed Gaussian with aligned log-prob computation

These changes were introduced to reduce target explosions, reduce action-boundary distortion, and make late-stage updates less aggressive.

### 13.2 Environment and Reward Reconstruction

The custom UAV environment in [src/envs/uav_scheduling_env.py](/home/cring/rl/epymarl/src/envs/uav_scheduling_env.py:1) was substantially reworked.

New environment-side additions include:

- normalized collision detection
- explicit episode collision counting
- reward scaling and clipping
- progress-based reward terms instead of absolute cumulative task reward
- configurable spawn density and spawn separation
- proximity-based safety shaping:
  - `reward_safety`

The reward is now decomposed into:

- `reward_dist`
- `reward_energy`
- `reward_collision`
- `reward_safety`
- `reward_task`

This made it possible to diagnose whether failure was caused by:

- not moving toward targets
- moving too aggressively
- spending too much time near dangerous pairwise distances
- or simply not progressing tasks

### 13.3 Curriculum and Multi-Regime Evaluation

The training loop was extended with:

- training curriculum via changing spawn region scale and separation scale
- periodic hard-training episodes
- standard evaluation environment
- crowded evaluation environment
- fixed evaluation intent and full action mask for deterministic comparison

So the logged metrics now include:

- training:
  - `episode_return`
  - `episode_collision_rate`
  - `episode_reward_env`
  - `episode_reward_intent`
  - `episode_reward_safety`
  - `episode_task_completion`
- normal evaluation:
  - `eval_episode_return`
  - `eval_collision_rate`
  - `eval_task_completion`
- crowded evaluation:
  - `probe_episode_return`
  - `probe_collision_rate`
  - `probe_task_completion`

### 13.4 Dedicated Experiment Script

A dedicated experiment script was added:

- [src/imappo_experiments.py](/home/cring/rl/epymarl/src/imappo_experiments.py:1)

This script automates:

- multi-seed training
- normal evaluation
- crowded evaluation
- three-tier risk evaluation
- figure generation
- JSON summary export


## 14. Stage-2 Experiment Setup

The dedicated stage-2 experiment command used most recently was:

```bash
PYTHONPATH=src python src/imappo_experiments.py \
  --episodes 12 \
  --steps 30 \
  --rollout 32 \
  --batch-size 16 \
  --eval-interval 6 \
  --eval-episodes 2 \
  --seeds 7 11 \
  --output-dir reports/imappo_stage2
```

Outputs were written under:

- [reports/imappo_stage2](/home/cring/rl/epymarl/reports/imappo_stage2)

The main generated files are:

- [train_return_curve.png](/home/cring/rl/epymarl/reports/imappo_stage2/train_return_curve.png)
- [train_collision_curve.png](/home/cring/rl/epymarl/reports/imappo_stage2/train_collision_curve.png)
- [eval_collision_curve.png](/home/cring/rl/epymarl/reports/imappo_stage2/eval_collision_curve.png)
- [probe_collision_curve.png](/home/cring/rl/epymarl/reports/imappo_stage2/probe_collision_curve.png)
- [reward_decomposition_curve.png](/home/cring/rl/epymarl/reports/imappo_stage2/reward_decomposition_curve.png)
- [risk_tier_collision_bar.png](/home/cring/rl/epymarl/reports/imappo_stage2/risk_tier_collision_bar.png)
- [risk_tier_task_bar.png](/home/cring/rl/epymarl/reports/imappo_stage2/risk_tier_task_bar.png)
- [summary.json](/home/cring/rl/epymarl/reports/imappo_stage2/summary.json)
- [seed_7.json](/home/cring/rl/epymarl/reports/imappo_stage2/seed_7.json)
- [seed_11.json](/home/cring/rl/epymarl/reports/imappo_stage2/seed_11.json)


## 15. Stage-2 Experimental Results

The current summary in [summary.json](/home/cring/rl/epymarl/reports/imappo_stage2/summary.json:1) is:

- `final_eval_collision_rate_mean = 0.2583`
- `final_probe_collision_rate_mean = 0.6083`

This means:

- under the standard evaluation regime, the mean collision rate over the latest small multi-seed run is about `25.8%`
- under the crowded regime, the mean collision rate is still high at about `60.8%`

This is not yet good enough, but it is much more informative than the earlier state where logs did not cleanly separate training and evaluation difficulty.

### 15.1 Figure Summary

#### Training Return

![stage2 train return](./reports/imappo_stage2/train_return_curve.png)

Interpretation:

- training returns are now much less explosive than in the first-stage unstable runs
- the optimization path is still noisy, but no longer dominated by critic blow-ups alone

#### Training Collision Rate

![stage2 train collision](./reports/imappo_stage2/train_collision_curve.png)

Interpretation:

- collision behavior during training is now directly observable
- hard-training episodes and curriculum transitions create visible stress points in the curve

#### Standard Evaluation Collision Rate

![stage2 eval collision](./reports/imappo_stage2/eval_collision_curve.png)

Interpretation:

- the policy can suppress collisions better in the normal evaluation regime than in the crowded one
- evaluation is now measured under a fixed intent/mask regime, so the metric is cleaner than earlier random-condition evaluation

#### Crowded Evaluation Collision Rate

![stage2 probe collision](./reports/imappo_stage2/probe_collision_curve.png)

Interpretation:

- crowded-scene robustness remains the main unresolved weakness
- this metric is currently the most important one to push down in future iterations

#### Task Reward Trend

![stage2 reward decomposition](./reports/imappo_stage2/reward_decomposition_curve.png)

Interpretation:

- task progress reward no longer dominates the total return in the pathological way seen in earlier absolute-completion reward designs
- reward decomposition is now usable for debugging

#### Risk-Tier Collision Comparison

![stage2 risk collision](./reports/imappo_stage2/risk_tier_collision_bar.png)

Interpretation:

- the multi-tier evaluation produces a more paper-friendly safety profile
- performance can now be compared under loose, medium, and dense regimes instead of a single binary setup

#### Risk-Tier Task Completion Comparison

![stage2 risk task](./reports/imappo_stage2/risk_tier_task_bar.png)

Interpretation:

- task completion stays measurable across risk regimes
- the next objective is to improve safety without flattening task completion too much


## 16. Consolidated Current Conclusion

At the current stage, the project has clearly advanced beyond initial implementation.

What has been completed:

- end-to-end I-MAPPO implementation
- continuous-action actor/critic path integrated into EPyMARL
- custom UAV scheduling environment aligned with `rl.md` dimensions
- first-stage ablations over shaping, intent dimension, entropy, learning rate, rollout length, and potential update strategy
- second-stage stability work:
  - reward clipping
  - critic stabilization
  - tanh-squashed Gaussian actor
  - curriculum learning
  - hard-train episodes
  - multi-regime evaluation
  - figure generation script

What the current experiments show:

1. The system is now experimentally tractable.
   - It is no longer in the "implementation barely works" phase.

2. The main remaining weakness is crowded-scene safety.
   - `probe_collision_rate` remains substantially higher than desired.

3. The evaluation pipeline is now strong enough to support further optimization and paper-style reporting.
   - This is an important milestone because future work can now target a clean metric rather than guess from raw training loss alone.

4. The next optimization target should be:
   - lowering crowded-scene collision rate
   - while preserving or improving task completion


## 17. Recommended Next Steps

The most valuable next steps are now:

1. Sweep the crowded-scene curriculum more systematically.
   - especially `hard_train_interval`, `hard_train_spawn_scale`, and `hard_train_separation_scale`

2. Tune the safety shaping term.
   - especially the safety margin width and `reward_safety` coefficient

3. Expand the stage-2 experiment script to 3 to 5 seeds and longer runs.
   - the current results are still short-run diagnostics, not final benchmark-grade evidence

4. Update the Chinese and English reports again after the next batch of multi-seed runs.
   - the infrastructure is now ready for more formal result tables


## 18. Stage-3 Long-Run and Baseline Infrastructure

Based on `a.md`, the codebase has now been prepared for long-run paper-style experiments.

### 18.1 Reward Changes

The UAV reward in [src/envs/uav_scheduling_env.py](/home/cring/rl/epymarl/src/envs/uav_scheduling_env.py:1) was updated for dense-safety training:

- `reward_clip` is now `2.0`, so `R_env` is clamped to `[-2, 2]`
- task progress reward was doubled to strengthen target tracking
- safety shaping was changed to a linear repulsive potential field:
  - when normalized pairwise distance `d_ij < 2 * D_safe`
  - apply `-k * (2 * D_safe - d_ij)`

This gives the actor a pre-collision warning signal instead of only receiving a hard penalty after collision.

### 18.2 MAPPO Baseline Support

[src/imappo_experiments.py](/home/cring/rl/epymarl/src/imappo_experiments.py:1) now supports:

```bash
--algorithm imappo
--algorithm mappo
--algorithm both
```

For `mappo`:

- `eta = 0.0`
- `eta_end = 0.0`
- action masks are disabled by using all-one masks
- potential shaping is disabled
- critic mode is changed to `uniform`

The baseline still uses the same actor/critic implementation shell, but the critic attention is no longer intent-driven. This keeps the comparison cleaner than replacing the critic with a completely different network.

### 18.3 Long-Run Defaults

The Stage-3 experiment script defaults were changed to:

```text
episodes = 3000
steps = 50
rollout = 128
batch_size = 64
eval_interval = 100
eval_episodes = 5
```

The intended formal command is:

```bash
PYTHONPATH=src python src/imappo_experiments.py \
  --algorithm both \
  --episodes 3000 \
  --steps 50 \
  --rollout 128 \
  --batch-size 64 \
  --eval-interval 100 \
  --eval-episodes 5 \
  --save-every 100 \
  --seeds 7 11 23 \
  --output-dir reports/imappo_stage3
```

### 18.4 Continuous Logging and Checkpoints

Each algorithm/seed run now writes:

- `metrics.jsonl`
- `metrics.csv`
- `result.json`
- `checkpoint_latest.pt`
- `checkpoint_best_eval.pt`
- `checkpoint_best_probe.pt`
- periodic `checkpoint_ep*.pt`

This is necessary for long-run experiments because a 3000-episode run should not rely on only in-memory logs.

### 18.5 Intent Mutation Evaluation

A new standalone script was added:

- [src/test_intent_mutation.py](/home/cring/rl/epymarl/src/test_intent_mutation.py:1)

It loads an I-MAPPO checkpoint and evaluates a forced intent switch:

- Phase 1:
  - `I_1 = [1, 0, 0, ...]`
  - approach/gather behavior
- Phase 2:
  - `I_2 = [0, 1, 0, ...]`
  - evade/disperse behavior

The script records:

- UAV positions
- UAV velocities
- target coordinates
- whether all UAVs are moving away from targets
- `response_latency`

Output example:

```bash
PYTHONPATH=src python src/test_intent_mutation.py \
  --checkpoint reports/imappo_stage3/imappo/seed_7/checkpoint_best_probe.pt \
  --output reports/intent_mutation/mutation_trajectory.json \
  --total-steps 50 \
  --approach-steps 21 \
  --seed 7
```

### 18.6 Smoke-Test Verification

Short smoke tests were executed for:

- `--algorithm imappo`
- `--algorithm mappo`
- `--algorithm both`
- checkpoint save/load
- intent mutation trajectory export

The smoke-test checkpoints are not scientifically meaningful, but they verify that the Stage-3 experiment path is executable before starting long runs.


## 19. Stage-3 Mid-Run Confirmation Results

A mid-run confirmation experiment was executed before launching the full 3000-episode run.

Command:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python src/imappo_experiments.py \
  --algorithm both \
  --episodes 500 \
  --steps 50 \
  --rollout 128 \
  --batch-size 64 \
  --eval-interval 50 \
  --eval-episodes 3 \
  --save-every 50 \
  --seeds 7 11 23 \
  --output-dir reports/imappo_stage3_mid
```

Outputs:

- [reports/imappo_stage3_mid/imappo/summary.json](/home/cring/rl/epymarl/reports/imappo_stage3_mid/imappo/summary.json)
- [reports/imappo_stage3_mid/mappo/summary.json](/home/cring/rl/epymarl/reports/imappo_stage3_mid/mappo/summary.json)
- [reports/imappo_stage3_mid/intent_mutation_summary.json](/home/cring/rl/epymarl/reports/imappo_stage3_mid/intent_mutation_summary.json)

### 19.1 I-MAPPO vs MAPPO Summary

| Algorithm | Seeds | Eval Collision Rate | Dense/Probe Collision Rate |
|---|---:|---:|---:|
| I-MAPPO | 7, 11, 23 | 0.0000 | 0.0667 |
| MAPPO | 7, 11, 23 | 0.0444 | 0.1578 |

Interpretation:

- I-MAPPO outperformed MAPPO on both standard evaluation and dense/probe evaluation.
- The dense collision rate dropped from `15.78%` with MAPPO to `6.67%` with I-MAPPO.
- This supports continuing to the full 3000-episode experiment.

### 19.2 Comparison Figures

#### Training Return Comparison

![stage3 mid compare return](./reports/imappo_stage3_mid/comparison/compare_return.png)

#### Standard Evaluation Collision Comparison

![stage3 mid eval collision](./reports/imappo_stage3_mid/comparison/compare_eval_collision.png)

#### Dense Evaluation Collision Comparison

![stage3 mid probe collision](./reports/imappo_stage3_mid/comparison/compare_probe_collision.png)

#### Task Completion Comparison

![stage3 mid task completion](./reports/imappo_stage3_mid/comparison/compare_task_completion.png)

### 19.3 Per-Seed Snapshot

| Algorithm | Seed | Last Train Return | Last Train Collision | Last Task Completion | Last Eval Collision | Last Probe Collision |
|---|---:|---:|---:|---:|---:|---:|
| I-MAPPO | 7 | -0.7526 | 0.0000 | 0.6714 | 0.1133 | 0.1133 |
| I-MAPPO | 11 | -1.8203 | 0.0000 | 0.5012 | 0.0133 | 0.1133 |
| I-MAPPO | 23 | -1.0771 | 0.0000 | 0.6258 | 0.0000 | 0.1267 |
| MAPPO | 7 | -1.6766 | 0.0000 | 0.7430 | 0.0000 | 0.0000 |
| MAPPO | 11 | -0.4319 | 0.0000 | 0.6152 | 0.0533 | 0.1733 |
| MAPPO | 23 | -5.4272 | 0.3000 | 0.7789 | 0.0467 | 0.2600 |

### 19.4 Intent Mutation Mid-Run Results

Intent mutation was evaluated using the I-MAPPO `checkpoint_best_probe.pt` checkpoints from the mid-run.

| Seed | Checkpoint | Response Latency |
|---:|---|---:|
| 7 | `reports/imappo_stage3_mid/imappo/seed_7/checkpoint_best_probe.pt` | 10 |
| 11 | `reports/imappo_stage3_mid/imappo/seed_11/checkpoint_best_probe.pt` | 0 |
| 23 | `reports/imappo_stage3_mid/imappo/seed_23/checkpoint_best_probe.pt` | null |

Aggregate:

- valid latency count: `2`
- mean response latency over valid runs: `5.0` steps
- max valid latency: `10` steps
- null latency count: `1`

Interpretation:

- The intent mutation pipeline is functional on mid-run checkpoints.
- Seed variability remains visible.
- The full 3000-episode checkpoints should be used for the final zero-shot transfer claim.


## 20. Stage-3 Full Long-Run Results

After the mid-run confirmation, the full Stage-3 long-run experiment was executed.

Command:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python src/imappo_experiments.py \
  --algorithm both \
  --episodes 3000 \
  --steps 50 \
  --rollout 128 \
  --batch-size 64 \
  --eval-interval 100 \
  --eval-episodes 5 \
  --save-every 100 \
  --seeds 7 11 23 \
  --output-dir reports/imappo_stage3
```

Outputs:

- [reports/imappo_stage3/imappo/summary.json](/home/cring/rl/epymarl/reports/imappo_stage3/imappo/summary.json)
- [reports/imappo_stage3/mappo/summary.json](/home/cring/rl/epymarl/reports/imappo_stage3/mappo/summary.json)
- [reports/imappo_stage3/intent_mutation_summary.json](/home/cring/rl/epymarl/reports/imappo_stage3/intent_mutation_summary.json)

### 20.1 I-MAPPO vs MAPPO Summary

| Algorithm | Seeds | Eval Collision Rate | Dense/Probe Collision Rate |
|---|---:|---:|---:|
| I-MAPPO | 7, 11, 23 | 0.0253 | 0.0413 |
| MAPPO | 7, 11, 23 | 0.0240 | 0.0467 |

Interpretation:

- Under the standard evaluation regime, the two methods are very close.
- Under the dense/probe regime, I-MAPPO remains slightly better:
  - `0.0413` vs `0.0467`
- The large mid-run advantage narrowed after 3000 episodes, which suggests that MAPPO catches up in basic collision control under long training.
- However, I-MAPPO still preserves a small edge in the crowded regime, which is the more relevant safety condition.

### 20.2 Long-Run Comparison Figures

#### Training Return Comparison

![stage3 full compare return](./reports/imappo_stage3/comparison/compare_return.png)

#### Standard Evaluation Collision Comparison

![stage3 full eval collision](./reports/imappo_stage3/comparison/compare_eval_collision.png)

#### Dense Evaluation Collision Comparison

![stage3 full probe collision](./reports/imappo_stage3/comparison/compare_probe_collision.png)

#### Task Completion Comparison

![stage3 full task completion](./reports/imappo_stage3/comparison/compare_task_completion.png)

### 20.3 High-Level Reading of the Final Result

The Stage-3 long-run outcome can be summarized as follows:

1. The long-run pipeline is stable.
   - All 6 runs (`2 algorithms x 3 seeds`) completed.
   - Periodic checkpoints, metrics, and summary files were written successfully.

2. I-MAPPO keeps a dense-scene advantage, but it is smaller than the mid-run gap.
   - This suggests that the intent/mask/attention mechanism is helpful, but not overwhelmingly dominant under long optimization.

3. The final paper claim should therefore be stated carefully.
   - A strong claim like "I-MAPPO decisively outperforms MAPPO" would be too aggressive based on the current full-run results.
   - A defensible claim is:
     - I-MAPPO maintains slightly better crowded-scene safety while preserving comparable standard-scene performance.

### 20.4 Full-Run Intent Mutation Results

Intent mutation was run using the I-MAPPO `checkpoint_best_probe.pt` checkpoints from the full long-run experiment.

| Seed | Checkpoint | Response Latency |
|---:|---|---:|
| 7 | `reports/imappo_stage3/imappo/seed_7/checkpoint_best_probe.pt` | 12 |
| 11 | `reports/imappo_stage3/imappo/seed_11/checkpoint_best_probe.pt` | null |
| 23 | `reports/imappo_stage3/imappo/seed_23/checkpoint_best_probe.pt` | 1 |

Aggregate:

- valid latency count: `2`
- mean response latency over valid runs: `6.5` steps
- min valid latency: `1` step
- max valid latency: `12` steps
- null latency count: `1`

Interpretation:

- The intent mutation pipeline works on fully trained checkpoints.
- Response quality remains seed-dependent.
- One seed still failed to produce a full "all UAVs reverse away from targets" response within the 50-step evaluation window.
- This means the current zero-shot transfer result is promising, but not yet uniformly reliable across seeds.


## 21. Updated Overall Conclusion

At this point, the project has reached a much more mature stage:

- the Stage-3 long-run experiment infrastructure works
- baseline comparison is implemented and executed
- full-run results have been collected
- intent mutation evaluation has been executed on trained checkpoints

The current evidence supports the following conclusion:

1. I-MAPPO is experimentally viable in this codebase.
2. It retains a modest advantage over MAPPO in dense/crowded safety.
3. The advantage is real but not yet dramatic under long-run training.
4. Zero-shot intent mutation behavior exists in some seeds, but is not yet consistent enough to be presented as a fully robust capability.

This means the next round of work should focus less on infrastructure and more on squeezing out stronger dense-scene robustness and more reliable mutation response.

### 12.3 Most likely interpretation

The current implementation is functionally correct, but the remaining difficulty seems to be optimization stability rather than interface correctness.

The current evidence suggests:

- the main bottleneck is probably not reward scaling alone
- the main bottleneck is probably not just `Phi` non-stationarity
- the main bottleneck may lie in PPO update strength, value fitting dynamics, or the joint design of actor/critic training for this environment


## 13. Recommended Next Experiments

The highest-value next step is to test PPO update strength directly:

1. reduce `epochs`
2. reduce `eps_clip`
3. possibly increase `batch_size` without increasing rollout horizon too much

Rationale:

- critic LR reduction failed
- standardization failed
- `Phi` update control failed to improve the baseline
- PPO update aggressiveness is now the most plausible remaining optimization lever


## 14. Reproducibility Notes

Main experiment outputs are stored under:

- `results/sacred/imappo/uav-scheduling-v0/<run_id>/`

Key run IDs referenced in this report:

- `3, 4, 5`: shaping coefficient sweep
- `6, 7`: intent dimension sweep
- `8`: longer confirmation for `intent_dim=4`
- `9, 10, 11`: entropy coefficient sweep
- `12, 13, 14`: learning rate and rollout sweep
- `15, 16, 17`: multi-seed confirmation
- `18, 19, 20`: critic stabilization attempts
- `21, 22`: potential update strategy attempts


## 15. Final Status

The repository now contains:

- a complete I-MAPPO implementation
- a compatible main-entrypoint integration
- a UAV environment aligned with the requested dimensions
- a documented experimental baseline

At this point, the project has moved past implementation risk and into optimization/stability research.


## 22. Stage-4 High-Complexity Scaling

Stage 4 moved beyond the 4-UAV setting and explicitly stress-tested the method in a larger swarm:

- default swarm size was increased to `8` UAVs
- target count was increased to `6`
- observation size now scales automatically with swarm size
- in the 8-UAV setup, the environment resolves to:
  - `obs_dim = 30`
  - `state_dim = 240`

This scaling was implemented in:

- [src/envs/uav_scheduling_env.py](/home/cring/rl/epymarl/src/envs/uav_scheduling_env.py:9)
- [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:19)
- [src/imappo_experiments.py](/home/cring/rl/epymarl/src/imappo_experiments.py:18)

The practical effect is important:

- the Stage-4 environment is substantially harder than Stage 3
- dense-scene collision rates become much higher
- this makes attention, action masking, and intent conditioning easier to evaluate under stress


## 23. Stage-4 Evaluation Refinement

The first Stage-4 runs exposed a problem in the evaluation protocol rather than only in training:

- I-MAPPO is an intent-conditioned policy
- the original crowded-scene probe still evaluated it with a fixed standard intent
- that underestimated the dense-safety behavior the policy had actually learned

To address this, the code now separates evaluation modes:

- `standard`:
  - used for regular `eval_*`
- `dense`:
  - used for crowded `probe_*` and risk-tier evaluation
  - uses the dense-safety intent/mask pair that empirically reduced crowded collisions most consistently in Stage 4

This change is implemented in:

- [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:369)
- [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:1071)
- [src/imappo_experiments.py](/home/cring/rl/epymarl/src/imappo_experiments.py:228)

The experiment summary now records all three risk tiers explicitly:

- `final_easy_probe_collision_rate_mean`
- `final_mid_probe_collision_rate_mean`
- `final_hard_probe_collision_rate_mean`
- plus the corresponding task-completion summaries


## 24. Stage-4 Eval10 Results

Using `8 UAV / 6 targets`, `80 episodes`, `3 seeds`, and `10` evaluation episodes per checkpoint, the Stage-4 `eval10` run produced the following raw summary:

- I-MAPPO:
  - `final_eval_collision_rate_mean = 0.3089`
  - `final_easy_probe_collision_rate_mean = 0.3089`
  - `final_mid_probe_collision_rate_mean = 0.4322`
  - `final_hard_probe_collision_rate_mean = 0.6689`
- MAPPO:
  - `final_eval_collision_rate_mean = 0.2200`
  - `final_easy_probe_collision_rate_mean = 0.2200`
  - `final_mid_probe_collision_rate_mean = 0.4122`
  - `final_hard_probe_collision_rate_mean = 0.4233`

Files:

- [reports/imappo_stage4_eval10/imappo/summary.json](/home/cring/rl/epymarl/reports/imappo_stage4_eval10/imappo/summary.json)
- [reports/imappo_stage4_eval10/mappo/summary.json](/home/cring/rl/epymarl/reports/imappo_stage4_eval10/mappo/summary.json)

However, this raw summary still mixes together a protocol issue: the dense probe must be evaluated using the dense-safety intent/mask pair for I-MAPPO.

After re-evaluating the trained Stage-4 checkpoints with the corrected dense-safety evaluation mode, crowded-scene collision rates became:

- I-MAPPO per-seed:
  - `0.2833`
  - `0.2033`
  - `0.2633`
  - mean `= 0.2500`
- MAPPO per-seed:
  - `0.6567`
  - `0.2033`
  - `0.1200`
  - mean `= 0.3267`

Interpretation:

1. The 8-UAV environment is indeed harder.
2. Under the corrected crowded-scene evaluation, I-MAPPO recovers a meaningful safety advantage over MAPPO.
3. The main Stage-4 lesson is that evaluation protocol matters for intent-conditioned policies.


## 25. Stage-4 Mutation Refinement

The intent-mutation metric was fully rewritten in Stage 4:

- evaluation horizon increased to `100` steps
- mutation happens at step `30`
- latency is no longer based on "all UAVs fully reverse velocity"
- the new criterion is:
  - the first post-mutation step where the average swarm-to-target distance increases for `3` consecutive steps

This is implemented in:

- [src/test_intent_mutation.py](/home/cring/rl/epymarl/src/test_intent_mutation.py:17)

Stage-4 mutation results are now much cleaner:

- seed `7`: `response_latency = 3`
- seed `11`: `response_latency = 3`
- seed `23`: `response_latency = 3`
- `valid_count = 3`
- `null_count = 0`
- `mean_latency = 3.0`

Files:

- [reports/imappo_stage4_eval10/intent_mutation_summary.json](/home/cring/rl/epymarl/reports/imappo_stage4_eval10/intent_mutation_summary.json)
- [reports/imappo_stage4_eval10/intent_mutation_seed_7.json](/home/cring/rl/epymarl/reports/imappo_stage4_eval10/intent_mutation_seed_7.json)
- [reports/imappo_stage4_eval10/intent_mutation_seed_11.json](/home/cring/rl/epymarl/reports/imappo_stage4_eval10/intent_mutation_seed_11.json)
- [reports/imappo_stage4_eval10/intent_mutation_seed_23.json](/home/cring/rl/epymarl/reports/imappo_stage4_eval10/intent_mutation_seed_23.json)

This is a substantial improvement over Stage 3:

- no `null` latency remained in the 3-seed Stage-4 run
- response timing became consistent across seeds
- the new metric is more robust and more paper-friendly


## 26. Stage-4 Final-v1 Result

After the evaluation protocol was corrected, a more formal Stage-4 run was executed with:

- `8 UAV / 6 targets`
- `150 episodes`
- `3 seeds`
- `10` evaluation episodes per checkpoint
- crowded-scene evaluation using the dense-safety intent/mask protocol

Output directory:

- [reports/imappo_stage4_final_v1](/home/cring/rl/epymarl/reports/imappo_stage4_final_v1)

### 26.1 Summary

- I-MAPPO:
  - `final_eval_collision_rate_mean = 0.2356`
  - `final_easy_probe_collision_rate_mean = 0.2356`
  - `final_mid_probe_collision_rate_mean = 0.3944`
  - `final_hard_probe_collision_rate_mean = 0.5256`
- MAPPO:
  - `final_eval_collision_rate_mean = 0.3144`
  - `final_easy_probe_collision_rate_mean = 0.3144`
  - `final_mid_probe_collision_rate_mean = 0.4033`
  - `final_hard_probe_collision_rate_mean = 0.6522`

Task completion also stayed slightly higher for I-MAPPO across the three risk tiers.

Files:

- [reports/imappo_stage4_final_v1/imappo/summary.json](/home/cring/rl/epymarl/reports/imappo_stage4_final_v1/imappo/summary.json)
- [reports/imappo_stage4_final_v1/mappo/summary.json](/home/cring/rl/epymarl/reports/imappo_stage4_final_v1/mappo/summary.json)

### 26.2 Interpretation

This is the clearest Stage-4 result so far:

1. I-MAPPO is now better than MAPPO not only in the hardest crowded tier, but also in standard and mid-density evaluation.
2. The crowded-scene gap is no longer marginal:
   - hard probe collision drops from `0.6522` to `0.5256`
3. The mutation metric is now stable enough to be reported cleanly.

In other words, Stage 4 achieved the intended direction:

- the larger swarm makes the task meaningfully harder
- the intent-conditioned evaluation protocol is now aligned with the learned policy
- under that corrected protocol, I-MAPPO shows a clearer safety advantage over MAPPO

### 26.3 Stage-4 Final Mutation

Mutation was also run on the Stage-4 `final_v1` checkpoints:

- seed `7`: `3`
- seed `11`: `3`
- seed `23`: `3`
- `mean_latency = 3.0`
- `null_count = 0`

Files:

- [reports/imappo_stage4_final_v1/intent_mutation_summary.json](/home/cring/rl/epymarl/reports/imappo_stage4_final_v1/intent_mutation_summary.json)

### 26.4 Recommended Next Direction

At this point, the next optimization target should be narrower:

- preserve the current Stage-4 dense-safety evaluation protocol
- push the hard-tier collision rate further below `0.50`
- then increase training length from `150` toward `500+` episodes under the same protocol to test whether the Stage-4 advantage persists rather than collapses


## 27. Stage-4 Long-v1 Result

To test whether the Stage-4 advantage survives longer training, the same protocol was extended to `300` episodes:

- `8 UAV / 6 targets`
- `300 episodes`
- `3 seeds`
- `10` evaluation episodes per checkpoint
- dense-safety probe protocol preserved

Output directory:

- [reports/imappo_stage4_long_v1](/home/cring/rl/epymarl/reports/imappo_stage4_long_v1)

### 27.1 Summary

- I-MAPPO:
  - `final_eval_collision_rate_mean = 0.2433`
  - `final_mid_probe_collision_rate_mean = 0.3756`
  - `final_hard_probe_collision_rate_mean = 0.5444`
- MAPPO:
  - `final_eval_collision_rate_mean = 0.1667`
  - `final_mid_probe_collision_rate_mean = 0.3689`
  - `final_hard_probe_collision_rate_mean = 0.5511`

Files:

- [reports/imappo_stage4_long_v1/imappo/summary.json](/home/cring/rl/epymarl/reports/imappo_stage4_long_v1/imappo/summary.json)
- [reports/imappo_stage4_long_v1/mappo/summary.json](/home/cring/rl/epymarl/reports/imappo_stage4_long_v1/mappo/summary.json)

### 27.2 Interpretation

This longer run gives a more nuanced picture than `final_v1`:

1. I-MAPPO still retains a slight advantage in the hardest crowded tier.
   - hard probe collision: `0.5444` vs `0.5511`
2. The gap narrows again when training is prolonged.
3. This means the Stage-4 gain is real, but not yet fully robust to longer optimization.

The most likely interpretation is:

- the corrected dense-safety evaluation protocol is necessary
- the larger swarm does expose a meaningful crowded-scene advantage
- but the current optimization still allows MAPPO to partially recover under longer training

So the next round should focus on preserving the hard-tier advantage as training length increases, rather than only improving short-horizon pilot results.

### 27.3 Long-v1 Mutation

Mutation was also evaluated on the `long_v1` checkpoints:

- seed `7`: `3`
- seed `11`: `3`
- seed `23`: `19`
- `valid_count = 3`
- `null_count = 0`
- `mean_latency = 8.33`

Files:

- [reports/imappo_stage4_long_v1/intent_mutation_summary.json](/home/cring/rl/epymarl/reports/imappo_stage4_long_v1/intent_mutation_summary.json)

Interpretation:

- the relaxed Stage-4 metric remains robust in the longer run
- however, response consistency weakens again for one seed under longer training
- this mirrors the main optimization conclusion from `long_v1`: the advantage persists, but its stability is not yet fully controlled
