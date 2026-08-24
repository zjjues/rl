# UAV 语义与控制链式消融报告

> 自动生成；证据等级：`smoke`。 本结果仅验证管线，不作效果推断。

## Artifact 与预注册状态

- 审计状态：`valid`
- 变体/比较/种子：10 / 9 / 1
- checksum 条目：14
- 多重检验族：18 个预注册主假设
- 警告：study invocation used a dirty Git worktree

## 链式主比较

| Factor | Contrast | Metric | Δ | 95% CI | exact p | Holm p | Reject |
|---|---|---|---:|---:|---:|---:|:---:|
| action_mask | no_mask − imappo_full | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| action_mask | no_mask − imappo_full | task_completion | 0.0519 | [0.0519, 0.0519] | 1.000000 | 1.000000 | no |
| attention_critic | no_attention − imappo_full | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| attention_critic | no_attention − imappo_full | task_completion | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| intent_potential_shaping | no_intent_reward − imappo_full | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| intent_potential_shaping | no_intent_reward − imappo_full | task_completion | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| cbf_safety_filter | no_cbf − imappo_full | collision_rate | 0.0133 | [0.0133, 0.0133] | 1.000000 | 1.000000 | no |
| cbf_safety_filter | no_cbf − imappo_full | task_completion | 0.0006 | [0.0006, 0.0006] | 1.000000 | 1.000000 | no |
| nli_prototype_gate | no_nli_gate − imappo_full | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| nli_prototype_gate | no_nli_gate − imappo_full | task_completion | -0.0046 | [-0.0046, -0.0046] | 1.000000 | 1.000000 | no |
| learned_residual | prior_only − imappo_full | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| learned_residual | prior_only − imappo_full | task_completion | -0.0000 | [-0.0000, -0.0000] | 1.000000 | 1.000000 | no |
| semantic_rule_prior | no_profile_prior − imappo_full | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| semantic_rule_prior | no_profile_prior − imappo_full | task_completion | 0.0116 | [0.0116, 0.0116] | 1.000000 | 1.000000 | no |
| semantic_representation | identity_oracle − no_profile_prior | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| semantic_representation | identity_oracle − no_profile_prior | task_completion | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| intent_channel | no_intent − identity_oracle | collision_rate | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | no |
| intent_channel | no_intent − identity_oracle | task_completion | -0.0000 | [-0.0000, -0.0000] | 1.000000 | 1.000000 | no |

## 运行资源

| Variant | Seed | Wall s | Peak CUDA MiB | Text cache entries |
|---|---:|---:|---:|---:|
| imappo_full | 7 | 68.65 | 804.4 | 2 |
| no_mask | 7 | 14.25 | 812.2 | 2 |
| no_attention | 7 | 13.04 | 812.2 | 2 |
| no_intent_reward | 7 | 12.92 | 812.2 | 2 |
| no_cbf | 7 | 13.14 | 812.2 | 2 |
| no_nli_gate | 7 | 12.89 | 812.2 | 2 |
| prior_only | 7 | 13.62 | 812.2 | 2 |
| no_profile_prior | 7 | 12.57 | 812.2 | 2 |
| identity_oracle | 7 | 10.26 | 660.4 | 2 |
| no_intent | 7 | 10.36 | 660.3 | 2 |

## 解释边界

- 每条效应均为 variant minus 其契约中注册的 reference；链式比较不可改写成全部相对 full。
- `identity_oracle` 获得 canonical-label identity，不是自然语言理解基线。
- `no_intent` 仍保留同一任务标签、奖励画像与 posture-derived mask，但 actor/critic 输入为全零；报告必须披露该 mask 侧信道。
- CBF 是经验安全过滤器；碰撞改善不能表述为严格安全保证。
- smoke 的单 seed 与极短训练只证明实现和统计流水线连通。
