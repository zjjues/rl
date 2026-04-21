# I-MAPPO 实验报告（中文版）

## 1. 文档目的

本文档汇总截至目前为止围绕 `rl.md` 所做的全部工作，包含：

- 算法设计与实现思路
- 在当前 EPyMARL 工程中的代码改动
- 环境设计与接入方式
- 已执行的全部实验配置
- 实验结果表格与图表
- 基于当前结果的分析
- 现阶段可得出的结论与后续建议

对应英文版报告见 [IMAPPO_EXPERIMENT_REPORT.md](/home/cring/rl/epymarl/IMAPPO_EXPERIMENT_REPORT.md:1)。


## 2. 任务背景与目标

`rl.md` 提出的目标不是简单复现现有 MAPPO，而是实现一个新的连续动作多智能体强化学习算法：

- 名称：`Intent-driven MAPPO (I-MAPPO)`
- 特点：
  - 每个 agent 的 actor 接收局部观测 `o_i` 和高层意图向量 `I`
  - critic 是 centralized global critic，并带有 cross-attention
  - 训练采用 CTDE 框架
  - 奖励中加入基于 `Phi(s, I)` 的 potential-based reward shaping

由于当前仓库原生的 MAPPO 路径主要面向离散动作，因此本任务本质上是：

- 在现有框架内新增一条连续动作算法分支
- 再为该算法构建符合 `rl.md` 维度约束的实验环境
- 在此基础上完成初步实验与稳定性分析


## 3. 算法原理

### 3.1 整体思路

I-MAPPO 的核心思想是：让低层连续控制策略显式地受到一个高层“意图向量”驱动，并让 centralized critic 用 cross-attention 机制去判断当前意图应该重点关注哪些 agent 的状态特征。

简化理解如下：

- actor 负责“每个无人机现在该怎么动”
- intent 负责“这回合的整体任务倾向是什么”
- critic 负责“从全局角度评估当前状态在这个意图下有多好”


### 3.2 Actor 原理

每个 agent 使用一个分布式 actor：

- 输入：`concat(o_i, I)`
- 主干：三层 MLP，隐藏维度 `256 -> 256 -> 128`
- 输出：
  - `action_mean`
  - `action_log_std`

然后构造高斯策略：

- `pi(a|o_i, I) = Normal(mean, std)`

这是连续动作控制的标准做法。

本实现中还加入了动作 mask 逻辑：

- 如果某个动作维度被 mask 掉，则对应动作均值压到安全范围
- 同时把该维度的方差压低，使采样尽量贴近安全动作

这对应 `rl.md` 中要求的“binary spatial-temporal action mask”。


### 3.3 Cross-Attention Critic 原理

critic 输入三部分：

- 全局状态 `s`
- 意图向量 `I`
- 所有 agent 的观测特征 `H_o`

具体计算过程：

1. 用线性层把 `I` 投影成 query：`Q`
2. 用线性层把 agent hidden features 投影成 key：`K`
3. 用线性层把 agent hidden features 投影成 value：`V`
4. 计算：
   - `alpha = softmax(QK^T / sqrt(d_k))`
5. 得到意图引导下的全局上下文：
   - `C_global = alpha V`
6. 最后拼接：
   - `[s, C_global]`
7. 再经过 MLP 输出：
   - `V(s, I, H_o)`

这样 critic 可以在不同 intent 下对不同 agent 的状态赋予不同权重。


### 3.4 Reward Shaping 原理

环境原始回报记为 `R_env`。

额外定义：

- `Phi(s, I) = -MSE(f(s), I)`

也就是说，先把状态映射到一个与 intent 同维的 embedding 空间，再用负的均方误差表示“状态与意图的一致程度”。

Intrinsic reward 为：

- `R_intent = gamma * Phi(s_{t+1}, I) - Phi(s_t, I)`

总奖励为：

- `r_i = R_env + eta * R_intent`

这保持了 potential-based shaping 的结构，同时把 intent 融入奖励信号。


### 3.5 训练原理

训练流程采用 CTDE：

- 执行时：各 agent 仅用自己的 `obs + intent`
- 训练时：critic 使用全局状态与全部 agent 特征

优化方式：

- actor：PPO clipped objective
- critic：MSE value loss
- advantage：GAE


## 4. 实验原理

当前阶段实验的核心目标不是最终性能冲榜，而是回答下面几个工程和研究问题：

1. 算法是否实现正确，能否跑通
2. 连续动作路径能否和当前仓库主入口兼容
3. `eta`、`intent_dim` 等关键设计是否有效
4. 当前训练不稳定主要来自哪里
5. 能否找到一个“当前最优基线配置”

因此实验分成四类：

- 正确性验证
- 接口与环境集成
- 关键超参消融
- 稳定性与可重复性分析


## 5. 代码改动总结

### 5.1 新增文件

#### [src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:1)

新增了完整的 I-MAPPO 实现，包括：

- `IMAPPOConfig`
- `IntentConditionedActor`
- `CrossAttentionCritic`
- `StateIntentPotential`
- `RolloutBuffer`
- `IMAPPO` 主算法类
- `train_imappo` 训练函数
- 环境适配和配置转换逻辑

关键代码位置：

- 配置定义：[src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:16)
- Actor：[src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:67)
- Critic：[src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:139)
- Potential 网络：[src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:185)
- PPO 更新：[src/imappo.py](/home/cring/rl/epymarl/src/imappo.py:307)

#### [src/config/algs/imappo.yaml](/home/cring/rl/epymarl/src/config/algs/imappo.yaml:1)

新增 I-MAPPO 算法配置。

#### [src/envs/uav_scheduling_env.py](/home/cring/rl/epymarl/src/envs/uav_scheduling_env.py:1)

新增自定义连续 UAV 调度环境：

- `n_agents = 4`
- `obs_dim = 18`
- `state_dim = 72`
- `action_dim = 3`


### 5.2 修改文件

#### [src/run.py](/home/cring/rl/epymarl/src/run.py:22)

增加 `imappo_learner` 分支，让算法可以通过：

```bash
python src/main.py --config=imappo ...
```

直接从主入口启动。

#### [src/config/envs/gymma.yaml](/home/cring/rl/epymarl/src/config/envs/gymma.yaml:1)

增加 `continuous_actions` 字段，避免 Sacred 因为未知配置键报错。

#### [src/envs/__init__.py](/home/cring/rl/epymarl/src/envs/__init__.py:1)

引入并注册新的 UAV 环境。


## 6. 环境设计说明

### 6.1 为什么需要新环境

本地可直接调用的 PettingZoo MPE 环境只适合做 smoke test，不满足 `rl.md` 指定的维度：

- 可用环境默认是 `3` 个 agent
- 动作维度也与要求不一致

因此无法作为“严格对齐需求”的正式实验环境。


### 6.2 UAV Scheduling 环境设计

新增环境 `uav-scheduling-v0` 的目标不是构造一个真实工业级仿真器，而是构造一个：

- 连续动作
- 多无人机
- 任务导向
- 维度严格对齐 `rl.md`

的实验环境。

环境要素包括：

- 每个 UAV 的位置和速度
- 各自目标点
- 能量状态
- 剩余任务量
- 邻近 UAV 的局部信息

奖励包含：

- 接近目标的正向趋势
- 控制代价惩罚
- 碰撞惩罚
- 任务完成奖励
- 能量项


## 7. 图表

### 7.1 Reward Shaping 系数对比

![eta ablation](./reports/imappo/eta_ablation.png)

结论：

- `eta=1.0` 显著优于 `eta=0.0` 和 `eta=0.5`
- intent-based shaping 在这个任务上是有帮助的


### 7.2 Intent 维度对比

![intent dim](./reports/imappo/intent_dim_ablation.png)

结论：

- `intent_dim=4` 和 `8` 明显优于 `16`
- intent 维度不是越大越好


### 7.3 Entropy 系数扫描

![entropy sweep](./reports/imappo/entropy_sweep.png)

结论：

- 较大的 entropy regularization 没有改善训练
- 当前实验里 `entropy_coef=0.0005` 最优


### 7.4 学习率与 Rollout 长度对比

![lr rollout](./reports/imappo/lr_rollout_sweep.png)

结论：

- `lr=1e-4` 是当前最优学习率
- `rollout_length=32` 没带来好处


### 7.5 多 Seed 结果

![multiseed](./reports/imappo/multiseed_returns.png)

结论：

- 回报在多 seed 下处于同一量级
- 不是完全不可复现
- 但波动仍存在


### 7.6 Critic 稳定化尝试

![critic stability](./reports/imappo/critic_stability.png)

结论：

- 单纯降低 `critic_lr`
- 或同时打开 reward/return 标准化

都没有改善，甚至更差。


## 8. 实验配置与结果

### 8.1 Reward Shaping 系数实验

| Run ID | 名称 | eta | intent_dim | episode_return | return_mean | critic_loss | entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| 3 | imappo_eta0 | 0.0 | 8 | -24.80 | -1.550 | 19.07 | 5.69 |
| 5 | imappo_eta05 | 0.5 | 8 | -31.24 | -1.952 | 16.29 | 5.28 |
| 4 | imappo_eta1 | 1.0 | 8 | -19.61 | -1.225 | 17.99 | 4.57 |

分析：

- `eta=1.0` 是明显赢家
- 说明 shaping 信号在这个环境下有实际价值


### 8.2 Intent 维度实验

| Run ID | 名称 | eta | intent_dim | episode_return | return_mean | critic_loss | entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| 6 | imappo_intent4 | 1.0 | 4 | -19.63 | -1.227 | 30.25 | -0.90 |
| 4 | imappo_eta1 | 1.0 | 8 | -19.61 | -1.225 | 17.99 | 4.57 |
| 7 | imappo_intent16 | 1.0 | 16 | -26.09 | -1.631 | 6.25 | 4.15 |

分析：

- `4` 和 `8` 都明显好于 `16`
- `intent_dim=4` 虽然短程回报接近最好，但 entropy 已经出现塌缩迹象
- 因此从稳定性角度看，`intent_dim=8` 更保守


### 8.3 长一点的确认实验

| Run ID | 名称 | eta | intent_dim | max_episodes | episode_return | return_mean | critic_loss | entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | imappo_confirm_eta1_int4 | 1.0 | 4 | 50 | -30.33 | -1.896 | 16.14 | 1.20 |

分析：

- `intent_dim=4` 的短程优势没有延续到更长训练
- 因此没有继续作为主线配置


### 8.4 Entropy 系数实验

| Run ID | 名称 | entropy_coef | lr | episode_return | return_mean | critic_loss | entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| 11 | imappo_ent0005 | 0.0005 | 3e-4 | -25.21 | -1.576 | 20.45 | 3.76 |
| 9 | imappo_ent005 | 0.005 | 3e-4 | -32.08 | -2.005 | 24.43 | 4.84 |
| 10 | imappo_ent01 | 0.01 | 3e-4 | -30.59 | -1.912 | 7.32 | 4.56 |

分析：

- 更大的 entropy regularization 不好
- 当前任务不是“探索不够”的问题


### 8.5 学习率与 Rollout 长度实验

| Run ID | 名称 | lr | critic_lr | rollout | batch | episode_return | return_mean | critic_loss | entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | imappo_lr1e4 | 1e-4 | 1e-4 | 16 | 16 | -19.10 | -1.194 | 12.83 | 3.54 |
| 11 | imappo_ent0005 | 3e-4 | 3e-4 | 16 | 16 | -25.21 | -1.576 | 20.45 | 3.76 |
| 14 | imappo_lr5e4 | 5e-4 | 5e-4 | 16 | 16 | -24.02 | -1.501 | 3.97 | 3.54 |
| 12 | imappo_roll32 | 3e-4 | 3e-4 | 32 | 32 | -26.59 | -1.930 | 7.92 | 1.70 |

分析：

- `lr=1e-4` 给出了目前最好的单次结果
- `rollout_length=32` 没有帮助
- 因此当前主线配置固定为 `lr=1e-4, rollout=16`


### 8.6 多 Seed 确认实验

配置固定为：

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

聚合统计：

- `episode_return mean = -28.01`
- `episode_return std = 1.82`
- `return_mean mean = -1.750`
- `return_mean std = 0.113`
- `critic_loss mean = 83.83`
- `critic_loss std = 7.21`
- `entropy mean = 2.45`
- `entropy std = 1.70`

分析：

- 回报量级基本可复现
- 但 critic loss 偏高，entropy 波动仍明显
- 说明“不是完全随机”，但训练还不算稳定


### 8.7 Critic 稳定化实验

| Run ID | 名称 | critic_lr | std_rewards | std_returns | episode_return | return_mean | critic_loss |
|---|---|---:|---|---|---:|---:|---:|
| 13 | imappo_lr1e4 | 1e-4 | False | False | -19.10 | -1.194 | 12.83 |
| 20 | imappo_critic5e5 | 5e-5 | False | False | -24.61 | -1.538 | 113.89 |
| 18 | imappo_critic3e5 | 3e-5 | False | False | -38.23 | -2.389 | 278.41 |
| 19 | imappo_std_both | 1e-4 | True | True | -39.60 | -2.475 | 173.91 |

分析：

- 降低 critic_lr 没救训练
- 打开 reward/return 标准化也没救
- 说明当前问题不是简单的“critic 学太快”


### 8.8 Phi 更新策略实验

| Run ID | 名称 | Phi 更新方式 | interval | episode_return | return_mean | critic_loss | entropy |
|---|---|---|---:|---:|---:|---:|---:|
| 13 | imappo_lr1e4 | normal | - | -19.10 | -1.194 | 12.83 | 3.54 |
| 21 | imappo_phi_slow8 | slow | 8 | -20.40 | -1.275 | 27.52 | 3.77 |
| 22 | imappo_phi_frozen | frozen | - | -20.93 | -1.308 | 21.98 | 3.66 |

分析：

- 冻结 `Phi`
- 或者低频更新 `Phi`

都没有超过当前最佳基线 `run 13`。

这说明：

- `Phi` 的非平稳性不是当前首要瓶颈


## 9. 当前最优配置

截至目前，当前表现最好的配置是：

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

对应代表性 run：

- `run 13`


## 10. 基于当前实验结果的分析

### 10.1 已经可以确定的事情

1. `I-MAPPO` 代码路径是通的。
   - 算法实现、主入口、日志、环境、训练循环都已打通。

2. reward shaping 是有效的。
   - `eta=1.0` 明显优于不加 shaping。

3. intent 维度不宜过大。
   - `16` 明显差于 `4/8`。

4. 当前任务不需要更强 entropy 正则。
   - 更大的 `entropy_coef` 只会降低表现。

5. 当前最好的训练学习率是 `1e-4`。
   - 比 `3e-4` 和 `5e-4` 更稳。


### 10.2 当前最大问题

尽管已有一个当前最优基线，但训练仍未达到“稳定可靠”的程度，原因包括：

- 多 seed 下 critic loss 仍偏大
- entropy 波动明显
- 某些看似短程有效的配置在长程上会退化

也就是说，当前阶段已经从“能不能实现”进入“怎么让优化更稳定”的问题。


### 10.3 对原因的判断

根据已做实验，目前可以排除几类简单猜测：

- 不是因为 shaping 系数太小
- 不是因为 intent_dim 太小
- 不是因为 entropy 正则不够强
- 不是因为 critic_lr 太大
- 不是因为 `Phi` 非平稳更新

因此，剩下最值得怀疑的方向更集中在：

- PPO 更新强度本身
- actor/critic 联合优化方式
- critic 的结构归纳偏置


## 11. 当前结论

一句话总结：

> I-MAPPO 已经在当前工程中实现并完成实验闭环，且 reward shaping 和 intent 条件建模在该 toy UAV 环境上显示出明确价值；但当前训练尚未达到稳定收敛状态，后续重点应转向 PPO 更新强度与优化稳定性，而不是继续围绕 `Phi` 或简单 critic 学习率做文章。


## 12. 建议的下一步

基于当前所有结果，下一步最值得做的是：

1. 扫 `epochs`
2. 扫 `eps_clip`
3. 视情况调整 `batch_size`

原因：

- critic 稳定化的常规技巧已验证无效
- `Phi` 更新策略也不是主因
- 当前最可能的问题是 PPO 更新太激进或 advantage/value 协调不稳


## 13. 附录：图文件位置

本次新增的图表位于：

- [reports/imappo/eta_ablation.png](/home/cring/rl/epymarl/reports/imappo/eta_ablation.png)
- [reports/imappo/intent_dim_ablation.png](/home/cring/rl/epymarl/reports/imappo/intent_dim_ablation.png)
- [reports/imappo/entropy_sweep.png](/home/cring/rl/epymarl/reports/imappo/entropy_sweep.png)
- [reports/imappo/lr_rollout_sweep.png](/home/cring/rl/epymarl/reports/imappo/lr_rollout_sweep.png)
- [reports/imappo/multiseed_returns.png](/home/cring/rl/epymarl/reports/imappo/multiseed_returns.png)
- [reports/imappo/critic_stability.png](/home/cring/rl/epymarl/reports/imappo/critic_stability.png)
