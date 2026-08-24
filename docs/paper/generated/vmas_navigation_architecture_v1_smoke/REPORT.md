# VMAS architecture-only 结果报告

> Smoke only：只验证执行、统计和产物管线，禁止算法排序。

## 机器合同

- 仅聚合 VMAS 场景原生 `episode_return`。
- 语言输入、偏好解码、UAV reward profile、动作掩码和 safety filter 全部关闭。
- 本结果不能证明语言泛化、偏好准确率、UAV 安全迁移或 UAV task completion。
- Artifact：`valid`；seeds=1；eval episodes/seed/tier=3。

## Episode return

| Variant | Tier | Mean [95% CI] | IQM |
|---|---|---:|---:|
| imappo_attention | canonical | 0.404719 [0.404719, 0.404719] | 0.404719 |
| mappo | canonical | 0.372961 [0.372961, 0.372961] | 0.372961 |
| ippo | canonical | 0.393917 [0.393917, 0.393917] | 0.393917 |
| happo | canonical | -0.110636 [-0.110636, -0.110636] | -0.110636 |
| matd3 | canonical | -1.631845 [-1.631845, -1.631845] | -1.631845 |

## 配对比较（treatment minus baseline）

| Baseline | Tier | Δ mean [95% CI] | exact p | Holm p | Reject |
|---|---|---:|---:|---:|:---:|
| mappo | canonical | 0.031758 [0.031758, 0.031758] | 1.000000 | 1.000000 | no |
| ippo | canonical | 0.010802 [0.010802, 0.010802] | 1.000000 | 1.000000 | no |
| happo | canonical | 0.515355 [0.515355, 0.515355] | 1.000000 | 1.000000 | no |
| matd3 | canonical | 2.036564 [2.036564, 2.036564] | 1.000000 | 1.000000 | no |

## 解释

- 单 seed 区间退化且随机化检验无充分信息，任何点值都不能解释为优势或等效。
