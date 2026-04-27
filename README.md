# Extended Python MARL framework - EPyMARL

EPyMARL is an extension of [PyMARL](https://github.com/oxwhirl/pymarl) for multi-agent reinforcement learning research. This repository keeps the original EPyMARL codebase and additionally includes a project-specific I-MAPPO + UAV scheduling experiment pipeline.

For the original framework, see the brief English overview below and the standard installation / run instructions in the later sections.

For the project-specific additions and experiment results, see the Chinese sections:

- [项目快速摘要](#项目快速摘要)
- [本仓库新增重点](#本仓库新增重点)
- [I-MAPPO 与 UAV 调度环境](#i-mappo-与-uav-调度环境)

Core EPyMARL features include:
- **New!** Support for training in environments with individual rewards for all agents (for all algorithms that support such settings)
- **New!** Updated EPyMARL to use maintained [Gymnasium](https://gymnasium.farama.org/index.html) library instead of deprecated OpenAI Gym version 0.21.
- **New!** Support for new environments: native integration of [PettingZoo](https://pettingzoo.farama.org/), [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator), [matrix games](https://github.com/uoe-agents/matrix-games), [SMACv2](https://github.com/oxwhirl/smacv2), and [SMAClite](https://github.com/uoe-agents/smaclite)
- **New!** Support for logging to [weights and biases (W&B)](https://wandb.ai/)
- **New!** We added a simple plotting script to visualise run data
- Additional algorithms (IA2C, IPPO, MADDPG, MAA2C and MAPPO)
- Option for no-parameter sharing between agents (original PyMARL only allowed for parameter sharing)
- Flexibility with extra implementation details (e.g. hard/soft updates, reward standarization, and more)
- Consistency of implementations between different algorithms (fair comparisons)

See our blog post here: https://agents.inf.ed.ac.uk/blog/epymarl/

## Project-specific additions (中文)

## 项目快速摘要

- 基座仍然是 EPyMARL，用于多智能体强化学习训练与对比实验。
- 本仓库新增了一套连续动作 **I-MAPPO** 算法实现，以及 `uav-scheduling-v0` 无人机调度环境。
- 已补齐从训练、评估、baseline 对比、checkpoint、intent mutation 到图表和中英文报告的完整实验链路。
- 当前正式长程实验结果已落库到 `reports/imappo_stage3/`，可直接复现与查阅。

## 本仓库新增重点

### 算法与环境

- 新增 [src/imappo.py](./src/imappo.py)：
  连续动作 I-MAPPO，支持 I-MAPPO / MAPPO baseline、attention 或 uniform critic、checkpoint 保存与加载。
- 新增 [src/envs/uav_scheduling_env.py](./src/envs/uav_scheduling_env.py)：
  `uav-scheduling-v0` 连续 UAV 调度环境，包含碰撞检测、任务推进奖励、安全缓冲惩罚和课程学习相关配置。
- 新增 [src/test_intent_mutation.py](./src/test_intent_mutation.py)：
  用于 zero-shot intent mutation 响应测试。

### 实验能力

- 新增 [src/imappo_experiments.py](./src/imappo_experiments.py)：
  一键运行多 seed 的 I-MAPPO / MAPPO 训练、常规评估、拥挤评估、风险分档统计和图表汇总。
- 已提供中程确认实验 `reports/imappo_stage3_mid/` 和正式长程实验 `reports/imappo_stage3/`。
- 已提供中英文完整报告：
  [IMAPPO_EXPERIMENT_REPORT.md](./IMAPPO_EXPERIMENT_REPORT.md) /
  [IMAPPO_EXPERIMENT_REPORT_ZH.md](./IMAPPO_EXPERIMENT_REPORT_ZH.md)

### 当前正式结果

- I-MAPPO:
  `final_eval_collision_rate_mean = 0.0253`
  `final_probe_collision_rate_mean = 0.0413`
- MAPPO:
  `final_eval_collision_rate_mean = 0.0240`
  `final_probe_collision_rate_mean = 0.0467`
- 结论：
  常规评估下两者接近；在更拥挤的 `probe` 场景中，I-MAPPO 仍略优于 MAPPO。

## Upstream update summary

- **July 2024**: EPyMARL migrated from legacy OpenAI Gym to maintained [Gymnasium](https://gymnasium.farama.org/index.html), expanded environment support, and improved compatibility for newer wrappers and benchmarks.
- **General-sum support**: algorithms such as `IA2C`, `IPPO`, `MAA2C`, `MAPPO`, `IQL`, and `PAC` can train with individual rewards by setting `common_reward=False`.
- **Tooling**: the upstream project also supports W&B logging and simple result plotting.
- **July 2023**: upstream added the Pareto Actor-Critic (PAC) algorithm. Paper: https://arxiv.org/abs/2209.14344

For detailed upstream usage, see the later installation / configuration sections in this README.

# Table of Contents
- [Extended Python MARL framework - EPyMARL](#extended-python-marl-framework---epymarl)
- [项目快速摘要](#项目快速摘要)
- [本仓库新增重点](#本仓库新增重点)
- [Table of Contents](#table-of-contents)
- [Installation & Run instructions](#installation--run-instructions)
  - [Installing Dependencies](#installing-dependencies)
  - [Benchmark Paper Experiments](#benchmark-paper-experiments)
  - [Experiments in SMACv2 and SMAClite](#experiments-in-smacv2-and-smaclite)
  - [Experiments in PettingZoo and VMAS](#experiments-in-pettingzoo-and-vmas)
  - [I-MAPPO 与 UAV 调度环境](#i-mappo-与-uav-调度环境)
  - [Registering and Running Experiments in Custom Environments](#registering-and-running-experiments-in-custom-environments)
- [Experiment Configurations](#experiment-configurations)
- [Run a hyperparameter search](#run-a-hyperparameter-search)
- [Logging](#logging)
  - [Weights and Biases](#weights-and-biases)
- [Saving and loading learnt models](#saving-and-loading-learnt-models)
  - [Saving models](#saving-models)
  - [Loading models](#loading-models)
- [Plotting](#plotting)
- [Citing PyMARL and EPyMARL](#citing-pymarl-and-epymarl)
- [License](#license)

# Installation & Run instructions

## Installing Dependencies

To install the dependencies for the codebase, clone this repo and run:
```sh
pip install -r requirements.txt
```

To install a set of supported environments, you can use the provided `env_requirements.txt`:
```sh
pip install -r env_requirements.txt
```
which will install the following environments:
- [Level Based Foraging](https://github.com/uoe-agents/lb-foraging)
- [Multi-Robot Warehouse](https://github.com/uoe-agents/robotic-warehouse)
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) (used for the multi-agent particle environment)
- [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator)
- [Matrix games](https://github.com/uoe-agents/matrix-games)
- [SMAC](https://github.com/oxwhirl/smac)
- [SMACv2](https://github.com/oxwhirl/smacv2)
- [SMAClite](https://github.com/uoe-agents/smaclite)

To install these environments individually, please see instructions in the respective repositories. We note that in particular SMAC and SMACv2 require a StarCraft II installation with specific map files. See their documentation for more details.

Note that the [PAC algorithm](#update-as-of-15th-july-2023) introduces separate dependencies. To install these dependencies, use the provided requirements file:
```sh
pip install -r pac_requirements.txt
```

## Benchmark Paper Experiments

In ["Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks"](https://arxiv.org/abs/2006.07869) we introduce the Level-Based Foraging (LBF) and Multi-Robot Warehouse (RWARE) environments, and additionally evaluate in SMAC, Multi-agent Particle environments, and a set of matrix games. After installing these environments (see instructions above), we can run experiments in these environments as follows:

Matrix games:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="matrixgames:penalty-100-nostate-v0"
```

LBF:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3"
```

RWARE:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v2"
```

MPE:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3"
```
Note that for the MPE environments tag (predator-prey) and adversary, we provide pre-trained prey and adversary policies. These can be used to control the respective agents to make these tasks fully cooperative (used in the paper) by setting `env_args.pretrained_wrapper="PretrainedTag"` or `env_args.pretrained_wrapper="PretrainedAdversary"`.

SMAC:
```sh
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name="3s5z"
```

Below, we provide the base environment and key / map name for all the environments evaluated in the "Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks":

- Matrix games: all with `--env-config=gymma with env_args.time_limit=25 env_args.key="..."`
  - Climbing: `matrixgames:climbing-nostate-v0`
  - Penalty $k=0$: `matrixgames:penalty-0-nostate-v0`
  - Penalty $k=-25$: `matrixgames:penalty-25-nostate-v0`
  - Penalty $k=-50$: `matrixgames:penalty-50-nostate-v0`
  - Penalty $k=-75$: `matrixgames:penalty-75-nostate-v0`
  - Penalty $k=-100$: `matrixgames:penalty-100-nostate-v0`
- LBF: all with `--env-config=gymma with env_args.time_limit=50 env_args.key="..."`
  - 8x8-2p-2f-coop: `lbforaging:Foraging-8x8-2p-2f-coop-v3`
  - 8x8-2p-2f-2s-coop: `lbforaging:Foraging-2s-8x8-2p-2f-coop-v3`
  - 10x10-3p-3f: `lbforaging:Foraging-10x10-3p-3f-v3`
  - 10x10-3p-3f-2s: `lbforaging:Foraging-2s-10x10-3p-3f-v3`
  - 15x15-3p-5f: `lbforaging:Foraging-15x15-3p-5f-v3`
  - 15x15-4p-3f: `lbforaging:Foraging-15x15-4p-3f-v3`
  - 15x15-4p-5f: `lbforaging:Foraging-15x15-4p-5f-v3`
- RWARE: all with `--env-config=gymma with env_args.time_limit=500 env_args.key="..."`
  - tiny 2p: `rware:rware-tiny-2ag-v2`
  - tiny 4p: `rware:rware-tiny-4ag-v2`
  - small 4p: `rware:rware-small-4ag-v2`
- MPE: all with `--env-config=gymma with env_args.time_limit=25 env_args.key="..."`
  - simple speaker listener: `pz-mpe-simple-speaker-listener-v4`
  - simple spread: `pz-mpe-simple-spread-v3`
  - simple adversary: `pz-mpe-simple-adversary-v3` with additional `env_args.pretrained_wrapper="PretrainedAdversary"`
  - simple tag: `pz-mpe-simple-tag-v3` with additional `env_args.pretrained_wrapper="PretrainedTag"`
- SMAC: all with `--env-config=sc2 with env_args.map_name="..."`
  - 2s_vs_1sc: `2s_vs_1sc`
  - 3s5z: `3s5z`
  - corridor: `corridor`
  - MMM2: `MMM2`
  - 3s_vs_5z: `3s_vs_5z`
  
## Experiments in SMACv2 and SMAClite

EPyMARL now supports the new SMACv2 and SMAClite environments. We provide wrappers to integrate these environments into the Gymnasium interface of EPyMARL. To run experiments in these environments, you can use the following exemplary commands:

SMACv2:
```sh
python src/main.py --config=qmix --env-config=sc2v2 with env_args.map_name="protoss_5_vs_5"
```
We provide prepared configs for a range of SMACv2 scenarios, as described in the [SMACv2 repository](https://github.com/oxwhirl/smacv2), under `src/config/envs/smacv2_configs`. These can be run by providing the name of the config file as the `env_args.map_name` argument. To define a new scenario, you can create a new config file in the same format as the provided ones and provide its name as the `env_args.map_name` argument.

SMAClite:
```sh
python src/main.py --config=qmix --env-config=smaclite with env_args.time_limit=150 env_args.map_name="MMM"
```
By default, SMAClite uses a numpy implementation of the RVO2 library for collision avoidance. To instead use a faster optimised C++ RVO2 library, follow the instructions of [this repo](https://github.com/micadam/SMAClite-Python-RVO2) and provide the additional argument `env_args.use_cpp_rvo2=True`.

## Experiments in PettingZoo and VMAS

EPyMARL supports the PettingZoo and VMAS libraries for multi-agent environments using wrappers. To run experiments in these environments, you can use the following exemplary commands:

PettingZoo:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3"
```

VMAS:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=150 env_args.key="vmas-balance"
```

## I-MAPPO 与 UAV 调度环境

这一节只保留和当前项目直接相关的入口信息。更完整的背景、实验过程、图表与分析，请直接看：

- [IMAPPO_EXPERIMENT_REPORT.md](./IMAPPO_EXPERIMENT_REPORT.md)
- [IMAPPO_EXPERIMENT_REPORT_ZH.md](./IMAPPO_EXPERIMENT_REPORT_ZH.md)
- [reports/imappo_stage3](./reports/imappo_stage3)

### 新增内容

- 在 `src/imappo.py` 中新增连续动作版 I-MAPPO 实现
- 在 `src/config/algs/imappo.yaml` 中新增算法配置
- 新增一个注册为 `uav-scheduling-v0` 的 Gymnasium 环境
- 新增 `src/imappo_experiments.py`，用于 Stage 2 / Stage 3 专项实验
- 新增 `src/test_intent_mutation.py`，用于 intent mutation 测试
- 新增中英文实验报告与图表，用于记录实现过程和实验结果

### I-MAPPO 简介

I-MAPPO 可以理解为在 MAPPO 基础上加入了 intent 引导机制，主要包括：

- 使用局部观测和 intent 向量作为输入的分布式 Gaussian actor
- 基于 agent 特征做 cross-attention 的 centralized critic
- 基于 `Phi(s, I)` 的 potential-based reward shaping
- 面向连续多智能体控制的 CTDE 训练流程

当前实现支持三种 `Phi` 更新模式：

- `normal`
- `slow`
- `frozen`

### UAV 调度环境

新增环境 `uav-scheduling-v0` 的维度与 `rl.md` 中的要求保持一致：

- `n_agents = 4`
- `obs_dim = 18`
- `state_dim = 72`
- `action_dim = 3`

这个环境是一个轻量级连续控制研究环境，主要用于算法开发、消融实验和稳定性分析，并不是面向真实部署的高保真仿真器。

### 示例命令

可以直接通过 EPyMARL 的主入口运行 I-MAPPO：

```sh
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
  action_high=1.0 \
```

也可以直接运行面向当前 UAV 任务的专用实验脚本：

```sh
PYTHONPATH=src python src/imappo_experiments.py \
  --algorithm imappo \
  --episodes 3000 \
  --steps 50 \
  --rollout 128 \
  --batch-size 64 \
  --eval-interval 100 \
  --eval-episodes 5 \
  --seeds 7 11 23
```

如果需要同时生成 I-MAPPO 与 MAPPO baseline 的对比曲线，可以运行：

```sh
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

该脚本会自动完成以下工作：

- 运行多 seed 的 I-MAPPO / MAPPO 训练
- 同时输出常规评估 `eval_*` 与拥挤场景评估 `probe_*`
- 额外统计宽松 / 中等拥挤 / 高拥挤三档风险下的碰撞率与任务完成度
- 连续写出 `metrics.jsonl` 和 `metrics.csv`
- 保存 `checkpoint_latest.pt`、`checkpoint_best_eval.pt`、`checkpoint_best_probe.pt`
- 将图表与 JSON 汇总结果保存到 `reports/imappo_stage3/`

Stage 3 还包含 intent mutation 评估脚本：

```sh
PYTHONPATH=src python src/test_intent_mutation.py \
  --checkpoint reports/imappo_stage3/imappo/seed_7/checkpoint_best_probe.pt \
  --output reports/intent_mutation/mutation_trajectory.json \
  --total-steps 50 \
  --approach-steps 21 \
  --seed 7
```

该脚本会在第 `approach_steps` 步将 intent 从 `[1, 0, 0, ...]` 切换为 `[0, 1, 0, ...]`，并保存 UAV 轨迹、速度和 `response_latency`。

如果需要运行修正后的 Stage 5 协议版本，可以使用：

```sh
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python src/imappo_experiments.py \
  --algorithm both \
  --episodes 300 \
  --steps 50 \
  --rollout 128 \
  --batch-size 64 \
  --eval-interval 25 \
  --eval-episodes 5 \
  --save-every 100 \
  --seeds 7 11 23 \
  --n-agents 8 \
  --n-targets 6 \
  --output-dir reports/imappo_stage5_v3
```

这一路径对应当前修正后的 Stage 5 规则：

- tactical posture 由环境侧外生协议驱动，而不是由 policy intent 直接决定
- 训练阶段按 episode 在 `attack` / `stealth` 间切换
- `standard` 评估使用 `attack`
- `dense` probe 评估使用 `stealth`
- threat penalty 默认参与最终 `reward_clip`

对应的 mutation 评估可以在训练完成后运行：

```sh
PYTHONPATH=src python src/test_intent_mutation.py \
  --checkpoint reports/imappo_stage5_v3/imappo/seed_7/checkpoint_latest.pt \
  --output reports/imappo_stage5_v3/comparison/mutation_seed7.json \
  --total-steps 50 \
  --approach-steps 30 \
  --seed 7
```

### 当前稳定化实现要点

围绕 `ai_studio_code.md` 和后续持续优化，当前实现已包含：

- 环境奖励缩放与裁剪，避免 `R_env` 爆炸
- `R_intent` 显式裁剪，抑制 potential difference 尖峰
- Actor / Critic 梯度裁剪与 critic 的 value loss clipping
- 训练期课程学习环境和周期性高风险训练 episode
- 常规评估与高拥挤评估分离，便于画论文图
- 近碰撞安全缓冲区惩罚 `reward_safety`
- 训练日志中可直接获取：
  - `episode_collision_rate`
  - `eval_collision_rate`
  - `probe_collision_rate`
  - `episode_reward_env`
  - `episode_reward_intent`
  - `episode_reward_safety`
  - `episode_task_completion`
  eta=1.0 \
  intent_dim=8 \
  entropy_coef=0.0005 \
  lr=0.0001 \
  critic_lr=0.0001 \
  potential_lr=0.0001
```

### 报告与图表

更详细的实现说明、实验配置、结果表格与分析见：

- [IMAPPO_EXPERIMENT_REPORT.md](./IMAPPO_EXPERIMENT_REPORT.md)
- [IMAPPO_EXPERIMENT_REPORT_ZH.md](./IMAPPO_EXPERIMENT_REPORT_ZH.md)

生成的图表位于：

- [reports/imappo](./reports/imappo)
- [reports/imappo_stage3_mid](./reports/imappo_stage3_mid)
- [reports/imappo_stage3](./reports/imappo_stage3)

## Registering and Running Experiments in Custom Environments

EPyMARL supports environments that have been registered with Gymnasium. If you would like to use any other Gymnasium environment, you can do so by using the `gymma` environment with the `env_args.key` argument being provided with the registration ID of the environment. Environments can either provide a single scalar reward to run common reward experiments (`common_reward=True`), or should provide one environment per agent to run experiments with individual rewards (`common_reward=False`) or with common rewards using some reward scalarisation (see [documentation](#support-for-training-in-environments-with-individual-rewards-for-all-agents) for more details). 

To register a custom environment with Gymnasium, use the template below:
```python
from gymnasium import register

register(
  id="my-environment-v1",                         # Environment ID.
  entry_point="myenv.environment:MyEnvironment",  # The entry point for the environment class
  kwargs={
            ...                                   # Arguments that go to MyEnvironment's __init__ function.
        },
    )
```

After, you can run an experiment in this environment using the following command:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="myenv:my-environment-v1"
```
assuming that the environment is registered with the ID `my-environment-v1` in the installed library `myenv`.

# Experiment Configurations

EPyMARL defines yaml configuration files for algorithms and environments under `src/config`. `src/config/default.yaml` defines default values for a range of configuration options, including experiment information (`t_max` for number of timesteps of training etc.) and algorithm hyperparameters.

Further environment configs (provided to the main script via `--env-config=...`) can be found in `src/config/envs`. Algorithm configs specifying algorithms and their hyperparameters (provided to the main script via `--config=...`) can be found in `src/config/algs`. To change hyperparameters or define a new algorithm, you can modify these yaml config files or create new ones.

# Run a hyperparameter search

We include a script named `search.py` which reads a search configuration file (e.g. the included `search.config.example.yaml`) and runs a hyperparameter search in one or more tasks. The script can be run using
```shell
python search.py run --config=search.config.example.yaml --seeds 5 locally
```
In a cluster environment where one run should go to a single process, it can also be called in a batch script like:
```shell
python search.py run --config=search.config.example.yaml --seeds 5 single 1
```
where the 1 is an index to the particular hyperparameter configuration and can take values from 1 to the number of different combinations.

# Logging

By default, EPyMARL will use sacred to log results and models to the `results` directory. These logs include configuration files, a json of all metrics, a txt file of all outputs and more. Additionally, EPyMARL can log data to tensorboard files by setting `use_tensorboard: True` in the yaml config. We also added support to log data to [weights and biases (W&B)](https://wandb.ai/) with instructions below.

## Weights and Biases

First, make sure to install W&B and follow their instructions to authenticate and setup your W&B library (see the [quickstart guide](https://docs.wandb.ai/quickstart) for more details).

To tell EPyMARL to log data to W&B, you then need to specify the following parameters in [your configuration](#experiment-configurations):
```yaml
use_wandb: True # Log results to W&B
wandb_team: null # W&B team name
wandb_project: null # W&B project name
```
to specify the team and project you wish to log to within your account, and set `use_wandb=True`. By default, we log all W&B runs in "offline" mode, i.e. the data will only be stored locally and can be uploaded to your W&B account via `wandb sync ...`. To directly log runs online, please specify `wandb_mode="online"` within the config.

We also support logging all stored models directly to W&B so you can download and inspect these from the W&B online dashboard. To do so, use the following config parameters:
```yaml
wandb_save_model: True # Save models to W&B (only done if use_wandb is True and save_model is True)
save_model: True # Save the models to disk
save_model_interval: 50000
```
Note that models are only saved in general if `save_model=True` and to further log them to W&B you need to specify `use_wandb`, `wandb_team`, `wandb_project`, and `wandb_save_model=True`.

# Saving and loading learnt models

## Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

## Loading models

Learnt models can be loaded using the `checkpoint_path` and `load_step` parameters. `checkpoint_path` should point to a directory stored for a run by epymarl as stated above. The pointed-to directory should contain sub-directories for various timesteps at which checkpoints were stored. If `load_step` is not provided (by default `load_step=0`) then the last checkpoint of the pointed-to run is loaded. Otherwise the checkpoint of the closest timestep to `load_step` will be loaded. After loading, the learning will proceed from the corresponding timestep.

To only evaluate loaded models without any training, set the `checkpoint_path` and `load_step` parameters accordingly for the loading, and additionally set `evaluate=True`. Then, the loaded checkpoint will be evaluated for `test_nepisode` episodes before terminating the run.

# Plotting

The plotting script provided as `plot_results.py` supports plotting of any logged metric, can apply simple window-smoothing, aggregates results across multiple runs of the same algorithm, and can filter which results to plot based on algorithm and environment names.

If multiple configs of the same algorithm exist within the loaded data and you only want to plot the best config per algorithm, then add the `--best_per_alg` argument! If this argument is not set, the script will visualise all configs of each (filtered) algorithm and show the values of the hyperparameter config that differ across all present configs in the legend.

# Citing EPyMARL and PyMARL

The Extended PyMARL (EPyMARL) codebase was used in [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869).

*Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, & Stefano V. Albrecht. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021*

In BibTeX format:

```tex
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   url = {http://arxiv.org/abs/2006.07869},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
   code = {https://github.com/uoe-agents/epymarl},
}
```

If you use the original PyMARL in your research, please cite the [SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

# License
All the source code that has been taken from the PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file).
Any new code is also licensed under the Apache License v2.0
