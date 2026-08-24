# UAV I-MAPPO 架构 smoke 统计报告

> 自动生成；证据等级为 smoke，仅验证执行与统计管线，禁止效果推断。

## Artifact 审计

- 状态：`valid`
- 变体/种子/结果：6 / 1 / 6
- checksum 条目：10
- 警告：study invocation used a dirty Git worktree

## 均值与 95% bootstrap CI

| Variant | Tier | Collision | Task completion |
|---|---|---:|---:|
| imappo | easy | 0.0000 [0.0000, 0.0000] | 0.6452 [0.6452, 0.6452] |
| imappo | medium | 0.3333 [0.3333, 0.3333] | 0.5358 [0.5358, 0.5358] |
| imappo | hard | 0.9467 [0.9467, 0.9467] | 0.5513 [0.5513, 0.5513] |
| imappo_no_mask | easy | 0.0000 [0.0000, 0.0000] | 0.6445 [0.6445, 0.6445] |
| imappo_no_mask | medium | 0.3333 [0.3333, 0.3333] | 0.5414 [0.5414, 0.5414] |
| imappo_no_mask | hard | 0.8067 [0.8067, 0.8067] | 0.5520 [0.5520, 0.5520] |
| mappo | easy | 0.0000 [0.0000, 0.0000] | 0.6443 [0.6443, 0.6443] |
| mappo | medium | 0.3333 [0.3333, 0.3333] | 0.5426 [0.5426, 0.5426] |
| mappo | hard | 0.8400 [0.8400, 0.8400] | 0.5532 [0.5532, 0.5532] |
| ippo | easy | 0.0000 [0.0000, 0.0000] | 0.6451 [0.6451, 0.6451] |
| ippo | medium | 0.3333 [0.3333, 0.3333] | 0.5370 [0.5370, 0.5370] |
| ippo | hard | 0.9133 [0.9133, 0.9133] | 0.5535 [0.5535, 0.5535] |
| happo | easy | 0.0933 [0.0933, 0.0933] | 0.6458 [0.6458, 0.6458] |
| happo | medium | 0.6667 [0.6667, 0.6667] | 0.5534 [0.5534, 0.5534] |
| happo | hard | 0.9867 [0.9867, 0.9867] | 0.5590 [0.5590, 0.5590] |
| matd3 | easy | 0.3200 [0.3200, 0.3200] | 0.6771 [0.6771, 0.6771] |
| matd3 | medium | 0.6267 [0.6267, 0.6267] | 0.5648 [0.5648, 0.5648] |
| matd3 | hard | 0.6800 [0.6800, 0.6800] | 0.5701 [0.5701, 0.5701] |

## I-MAPPO 配对主比较

| Baseline | Tier | Metric | Δ mean | 95% CI | exact p | Holm p | Reject |
|---|---|---|---:|---:|---:|---:|:---:|
| imappo_no_mask | easy | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| imappo_no_mask | easy | task_completion | 0.0007 | [0.0007, 0.0007] | 1.000000 | 1.000000 | no |
| imappo_no_mask | medium | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| imappo_no_mask | medium | task_completion | -0.0057 | [-0.0057, -0.0057] | 1.000000 | 1.000000 | no |
| imappo_no_mask | hard | collision_rate | 0.1400 | [0.1400, 0.1400] | 1.000000 | 1.000000 | no |
| imappo_no_mask | hard | task_completion | -0.0007 | [-0.0007, -0.0007] | 1.000000 | 1.000000 | no |
| mappo | easy | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| mappo | easy | task_completion | 0.0009 | [0.0009, 0.0009] | 1.000000 | 1.000000 | no |
| mappo | medium | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| mappo | medium | task_completion | -0.0068 | [-0.0068, -0.0068] | 1.000000 | 1.000000 | no |
| mappo | hard | collision_rate | 0.1067 | [0.1067, 0.1067] | 1.000000 | 1.000000 | no |
| mappo | hard | task_completion | -0.0019 | [-0.0019, -0.0019] | 1.000000 | 1.000000 | no |
| ippo | easy | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| ippo | easy | task_completion | 0.0001 | [0.0001, 0.0001] | 1.000000 | 1.000000 | no |
| ippo | medium | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| ippo | medium | task_completion | -0.0012 | [-0.0012, -0.0012] | 1.000000 | 1.000000 | no |
| ippo | hard | collision_rate | 0.0333 | [0.0333, 0.0333] | 1.000000 | 1.000000 | no |
| ippo | hard | task_completion | -0.0021 | [-0.0021, -0.0021] | 1.000000 | 1.000000 | no |
| happo | easy | collision_rate | -0.0933 | [-0.0933, -0.0933] | 1.000000 | 1.000000 | no |
| happo | easy | task_completion | -0.0006 | [-0.0006, -0.0006] | 1.000000 | 1.000000 | no |
| happo | medium | collision_rate | -0.3333 | [-0.3333, -0.3333] | 1.000000 | 1.000000 | no |
| happo | medium | task_completion | -0.0176 | [-0.0176, -0.0176] | 1.000000 | 1.000000 | no |
| happo | hard | collision_rate | -0.0400 | [-0.0400, -0.0400] | 1.000000 | 1.000000 | no |
| happo | hard | task_completion | -0.0077 | [-0.0077, -0.0077] | 1.000000 | 1.000000 | no |
| matd3 | easy | collision_rate | -0.3200 | [-0.3200, -0.3200] | 1.000000 | 1.000000 | no |
| matd3 | easy | task_completion | -0.0318 | [-0.0318, -0.0318] | 1.000000 | 1.000000 | no |
| matd3 | medium | collision_rate | -0.2933 | [-0.2933, -0.2933] | 1.000000 | 1.000000 | no |
| matd3 | medium | task_completion | -0.0290 | [-0.0290, -0.0290] | 1.000000 | 1.000000 | no |
| matd3 | hard | collision_rate | 0.2667 | [0.2667, 0.2667] | 1.000000 | 1.000000 | no |
| matd3 | hard | task_completion | -0.0188 | [-0.0188, -0.0188] | 1.000000 | 1.000000 | no |

## 解释边界

- 只有 1 个 seed；bootstrap CI 退化且随机化检验没有足够信息，任何点估计都不能解释为机制效果或等效性。
- 本报告只证明注册变体、配对统计、Holm family、图表和 checksum 管线可执行。
- 该架构协议不使用自然语言查询，不能证明自然语言或语义意图带来优势。

## 投稿限制

- 当前协议为 `smoke`，包含 1 seed(s)、每 seed/tier 3 个评估回合。
- dirty-worktree 警告、低 seed 数或低评估回合数存在时，不得升级为 frozen paper evidence。
- 尚缺独立语言数据、跨场景部署和 HIL/实机证据。
