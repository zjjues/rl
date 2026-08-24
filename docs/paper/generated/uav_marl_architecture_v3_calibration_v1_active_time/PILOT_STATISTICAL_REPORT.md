# UAV I-MAPPO 架构先导实验统计报告

> 自动生成；证据等级为 pilot，不是 frozen paper result。

## Artifact 审计

- 状态：`valid`
- 变体/种子/结果：6 / 1 / 6
- checksum 条目：10
- 警告：study invocation used a dirty Git worktree

## 均值与 95% bootstrap CI

| Variant | Tier | Collision | Task completion |
|---|---|---:|---:|
| imappo | easy | 0.1330 [0.1330, 0.1330] | 0.5528 [0.5528, 0.5528] |
| imappo | medium | 0.3410 [0.3410, 0.3410] | 0.5565 [0.5565, 0.5565] |
| imappo | hard | 0.7370 [0.7370, 0.7370] | 0.5439 [0.5439, 0.5439] |
| imappo_no_mask | easy | 0.0900 [0.0900, 0.0900] | 0.5488 [0.5488, 0.5488] |
| imappo_no_mask | medium | 0.3280 [0.3280, 0.3280] | 0.5466 [0.5466, 0.5466] |
| imappo_no_mask | hard | 0.8280 [0.8280, 0.8280] | 0.5364 [0.5364, 0.5364] |
| mappo | easy | 0.1240 [0.1240, 0.1240] | 0.5500 [0.5500, 0.5500] |
| mappo | medium | 0.1790 [0.1790, 0.1790] | 0.5502 [0.5502, 0.5502] |
| mappo | hard | 0.8090 [0.8090, 0.8090] | 0.5410 [0.5410, 0.5410] |
| ippo | easy | 0.1040 [0.1040, 0.1040] | 0.5535 [0.5535, 0.5535] |
| ippo | medium | 0.2900 [0.2900, 0.2900] | 0.5543 [0.5543, 0.5543] |
| ippo | hard | 0.6390 [0.6390, 0.6390] | 0.5463 [0.5463, 0.5463] |
| happo | easy | 0.2110 [0.2110, 0.2110] | 0.5596 [0.5596, 0.5596] |
| happo | medium | 0.4350 [0.4350, 0.4350] | 0.5585 [0.5585, 0.5585] |
| happo | hard | 0.7400 [0.7400, 0.7400] | 0.5531 [0.5531, 0.5531] |
| matd3 | easy | 0.2290 [0.2290, 0.2290] | 0.5490 [0.5490, 0.5490] |
| matd3 | medium | 0.4620 [0.4620, 0.4620] | 0.5549 [0.5549, 0.5549] |
| matd3 | hard | 0.5360 [0.5360, 0.5360] | 0.5462 [0.5462, 0.5462] |

## I-MAPPO 配对主比较

| Baseline | Tier | Metric | Δ mean | 95% CI | exact p | Holm p | Reject |
|---|---|---|---:|---:|---:|---:|:---:|
| imappo_no_mask | easy | collision_rate | 0.0430 | [0.0430, 0.0430] | 1.000000 | 1.000000 | no |
| imappo_no_mask | easy | task_completion | 0.0040 | [0.0040, 0.0040] | 1.000000 | 1.000000 | no |
| imappo_no_mask | medium | collision_rate | 0.0130 | [0.0130, 0.0130] | 1.000000 | 1.000000 | no |
| imappo_no_mask | medium | task_completion | 0.0099 | [0.0099, 0.0099] | 1.000000 | 1.000000 | no |
| imappo_no_mask | hard | collision_rate | -0.0910 | [-0.0910, -0.0910] | 1.000000 | 1.000000 | no |
| imappo_no_mask | hard | task_completion | 0.0075 | [0.0075, 0.0075] | 1.000000 | 1.000000 | no |
| mappo | easy | collision_rate | 0.0090 | [0.0090, 0.0090] | 1.000000 | 1.000000 | no |
| mappo | easy | task_completion | 0.0028 | [0.0028, 0.0028] | 1.000000 | 1.000000 | no |
| mappo | medium | collision_rate | 0.1620 | [0.1620, 0.1620] | 1.000000 | 1.000000 | no |
| mappo | medium | task_completion | 0.0063 | [0.0063, 0.0063] | 1.000000 | 1.000000 | no |
| mappo | hard | collision_rate | -0.0720 | [-0.0720, -0.0720] | 1.000000 | 1.000000 | no |
| mappo | hard | task_completion | 0.0029 | [0.0029, 0.0029] | 1.000000 | 1.000000 | no |
| ippo | easy | collision_rate | 0.0290 | [0.0290, 0.0290] | 1.000000 | 1.000000 | no |
| ippo | easy | task_completion | -0.0007 | [-0.0007, -0.0007] | 1.000000 | 1.000000 | no |
| ippo | medium | collision_rate | 0.0510 | [0.0510, 0.0510] | 1.000000 | 1.000000 | no |
| ippo | medium | task_completion | 0.0022 | [0.0022, 0.0022] | 1.000000 | 1.000000 | no |
| ippo | hard | collision_rate | 0.0980 | [0.0980, 0.0980] | 1.000000 | 1.000000 | no |
| ippo | hard | task_completion | -0.0024 | [-0.0024, -0.0024] | 1.000000 | 1.000000 | no |
| happo | easy | collision_rate | -0.0780 | [-0.0780, -0.0780] | 1.000000 | 1.000000 | no |
| happo | easy | task_completion | -0.0068 | [-0.0068, -0.0068] | 1.000000 | 1.000000 | no |
| happo | medium | collision_rate | -0.0940 | [-0.0940, -0.0940] | 1.000000 | 1.000000 | no |
| happo | medium | task_completion | -0.0020 | [-0.0020, -0.0020] | 1.000000 | 1.000000 | no |
| happo | hard | collision_rate | -0.0030 | [-0.0030, -0.0030] | 1.000000 | 1.000000 | no |
| happo | hard | task_completion | -0.0092 | [-0.0092, -0.0092] | 1.000000 | 1.000000 | no |
| matd3 | easy | collision_rate | -0.0960 | [-0.0960, -0.0960] | 1.000000 | 1.000000 | no |
| matd3 | easy | task_completion | 0.0038 | [0.0038, 0.0038] | 1.000000 | 1.000000 | no |
| matd3 | medium | collision_rate | -0.1210 | [-0.1210, -0.1210] | 1.000000 | 1.000000 | no |
| matd3 | medium | task_completion | 0.0016 | [0.0016, 0.0016] | 1.000000 | 1.000000 | no |
| matd3 | hard | collision_rate | 0.2010 | [0.2010, 0.2010] | 1.000000 | 1.000000 | no |
| matd3 | hard | task_completion | -0.0023 | [-0.0023, -0.0023] | 1.000000 | 1.000000 | no |

## 解释边界

- Holm FWER 0.05 下没有主比较被拒绝；不能主张架构优势，也不能据此主张等效。
- 该架构协议不使用自然语言查询，不能证明自然语言或语义意图带来优势。

## 投稿限制

- 当前协议为 `pilot`，包含 1 seed(s)、每 seed/tier 20 个评估回合。
- dirty-worktree 警告、低 seed 数或低评估回合数存在时，不得升级为 frozen paper evidence。
- 尚缺独立语言数据、跨场景部署和 HIL/实机证据。
