# UAV I-MAPPO 架构先导实验统计报告

> **证据失效通知（2026-08-20）：** `mappo` 变体实际运行 `algorithm="imappo"`，未实现的 `critic_mode="concat"` 执行为 attention。本报告仅保留历史数值，所有 MAPPO/attention-vs-concat 方法解释撤回。最新审计：`docs/paper/audits/uav_imappo_main_semantic_protocol_audit.json`。

## Artifact 审计

- 状态：`valid`
- 变体/种子/结果：4 / 10 / 40
- checksum 条目：44
- 警告：study invocation used a dirty Git worktree

## 均值与 95% bootstrap CI

| Variant | Tier | Collision | Task completion |
|---|---|---:|---:|
| imappo | easy | 0.1191 [0.0981, 0.1406] | 0.5510 [0.5464, 0.5555] |
| imappo | medium | 0.2090 [0.1610, 0.2622] | 0.5556 [0.5498, 0.5613] |
| imappo | hard | 0.5478 [0.4610, 0.6346] | 0.5542 [0.5484, 0.5600] |
| mappo | easy | 0.1135 [0.0846, 0.1408] | 0.5481 [0.5420, 0.5538] |
| mappo | medium | 0.1777 [0.1496, 0.2039] | 0.5530 [0.5472, 0.5583] |
| mappo | hard | 0.4221 [0.3672, 0.4696] | 0.5518 [0.5450, 0.5595] |
| matd3 | easy | 0.3779 [0.3104, 0.4611] | 0.5807 [0.5754, 0.5852] |
| matd3 | medium | 0.4691 [0.4038, 0.5484] | 0.5876 [0.5799, 0.5953] |
| matd3 | hard | 0.5640 [0.4929, 0.6502] | 0.5864 [0.5813, 0.5914] |
| ippo | easy | 0.3030 [0.2657, 0.3363] | 0.6479 [0.6417, 0.6535] |
| ippo | medium | 0.4059 [0.3656, 0.4450] | 0.6533 [0.6481, 0.6582] |
| ippo | hard | 0.6249 [0.5919, 0.6627] | 0.6562 [0.6520, 0.6611] |

## I-MAPPO 配对主比较

| Baseline | Tier | Metric | Δ mean | 95% CI | exact p | Holm p | Reject |
|---|---|---|---:|---:|---:|---:|:---:|
| ippo | easy | collision_rate | -0.1838 | [-0.2265, -0.1358] | 0.001953 | 0.035156 | yes |
| ippo | easy | task_completion | -0.0970 | [-0.1005, -0.0939] | 0.001953 | 0.035156 | yes |
| ippo | medium | collision_rate | -0.1969 | [-0.2671, -0.1243] | 0.003906 | 0.035156 | yes |
| ippo | medium | task_completion | -0.0977 | [-0.1007, -0.0948] | 0.001953 | 0.035156 | yes |
| ippo | hard | collision_rate | -0.0770 | [-0.1619, 0.0180] | 0.150391 | 1.000000 | no |
| ippo | hard | task_completion | -0.1020 | [-0.1064, -0.0981] | 0.001953 | 0.035156 | yes |
| mappo | easy | collision_rate | 0.0056 | [-0.0321, 0.0395] | 0.781250 | 1.000000 | no |
| mappo | easy | task_completion | 0.0029 | [-0.0021, 0.0078] | 0.308594 | 1.000000 | no |
| mappo | medium | collision_rate | 0.0313 | [-0.0318, 0.1013] | 0.396484 | 1.000000 | no |
| mappo | medium | task_completion | 0.0026 | [-0.0023, 0.0075] | 0.341797 | 1.000000 | no |
| mappo | hard | collision_rate | 0.1258 | [0.0336, 0.2200] | 0.037109 | 0.296875 | no |
| mappo | hard | task_completion | 0.0024 | [-0.0047, 0.0096] | 0.542969 | 1.000000 | no |
| matd3 | easy | collision_rate | -0.2588 | [-0.3452, -0.1893] | 0.001953 | 0.035156 | yes |
| matd3 | easy | task_completion | -0.0297 | [-0.0342, -0.0254] | 0.001953 | 0.035156 | yes |
| matd3 | medium | collision_rate | -0.2601 | [-0.3611, -0.1710] | 0.001953 | 0.035156 | yes |
| matd3 | medium | task_completion | -0.0319 | [-0.0374, -0.0264] | 0.001953 | 0.035156 | yes |
| matd3 | hard | collision_rate | -0.0161 | [-0.1223, 0.0868] | 0.773438 | 1.000000 | no |
| matd3 | hard | task_completion | -0.0322 | [-0.0383, -0.0265] | 0.001953 | 0.035156 | yes |

## 可支持的结论

- I-MAPPO 相对 MAPPO 的 collision/task 主比较在 Holm 校正后均不显著，不能主张 attention critic + action mask 优于 MAPPO。
- I-MAPPO 相对 IPPO 在 easy/medium 碰撞率更低，但任务完成率也稳定更低，属于安全—任务权衡。
- I-MAPPO 相对 MATD3 在 easy/medium 碰撞率更低，但任务完成率更低；hard 碰撞差异不稳定。
- 本实验所有方法使用 one-hot intent 且关闭 intent reward，只能回答架构问题，不能证明自然语言/语义意图带来优势。

## 投稿限制

- 历史训练运行来自 dirty Git worktree；当前 artifact 可验证但不能升级为 frozen paper evidence。
- 每 seed/tier 仅 50 个评估回合，低于工程规定的 paper 门槛 100。
- 尚缺语义方法的因果消融、独立语言数据、跨场景部署和 HIL/实机证据。
