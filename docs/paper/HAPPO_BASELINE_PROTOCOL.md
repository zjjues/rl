# HAPPO 强基线实现协议

## 固定参考

- 原论文：Kuba et al., *Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning*, ICLR 2022，https://arxiv.org/abs/2109.11251
- 官方 HARL 仓库：https://github.com/PKU-MARL/HARL
- 固定 commit：`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`
- 顺序 runner：`harl/runners/on_policy_ha_runner.py`
- HAPPO actor loss：`harl/algorithms/actors/happo.py`

## 本地不可变合同

1. 每个 agent 有独立 actor 与 optimizer；禁止参数共享。
2. 每次 rollout update 随机排列 agents。
3. 当前 agent 的 PPO surrogate 乘以此前已更新 agents 的 likelihood-ratio product factor。
4. 当前 agent 完成所有 PPO epochs 后，才更新 factor 并进入下一 agent。
5. centralized MLP critic 在全部 actors 后更新。
6. baseline 不接收 intent、action mask、规则先验或 safety filter。
7. result 必须记录 actor count、共享状态、更新方案和 critic 身份；validator 对 HAPPO 强制检查。

## 已验证证据

- `tests/test_happo_baseline.py`：独立参数、逐 actor 更新、有限 factor、agent count、checkpoint。
- `tests/test_research_protocol.py`：拒绝共享 actor 或不完整 HAPPO 声明。
- `tests/test_research_artifact.py`：逐 result 核验 actor count 与实现元数据。
- `experiments/smoke/uav_marl_architecture_v3_smoke/`：6/6 变体端到端完成。
- `docs/paper/audits/uav_marl_architecture_v3_smoke.json`：10 checksums、0 errors。

## 尚未满足

本地实现尚未在同一 UAV adapter 上与官方 HARL runner 做数值轨迹交叉核验；v3 也只有单 seed smoke。因此当前只能登记为“协议交叉检查并端到端验证的本地 HAPPO”，不能声称复现官方 HAPPO 性能或理论保证。
