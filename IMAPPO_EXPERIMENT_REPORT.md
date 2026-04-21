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
