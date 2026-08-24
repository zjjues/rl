# VMAS architecture-only 结果报告

> Pilot calibration：用于估计成本与方差，不是冻结论文结论。

## 机器合同

- 仅聚合 VMAS 场景原生 `episode_return`。
- 语言输入、偏好解码、UAV reward profile、动作掩码和 safety filter 全部关闭。
- 本结果不能证明语言泛化、偏好准确率、UAV 安全迁移或 UAV task completion。
- Artifact：`valid`；seeds=1；eval episodes/seed/tier=20。

## Episode return

| Variant | Tier | Mean [95% CI] | IQM |
|---|---|---:|---:|
| imappo_attention | canonical | 0.050000 [0.050000, 0.050000] | 0.050000 |
| mappo | canonical | 0.016667 [0.016667, 0.016667] | 0.016667 |
| ippo | canonical | 0.033333 [0.033333, 0.033333] | 0.033333 |
| happo | canonical | 0.033333 [0.033333, 0.033333] | 0.033333 |
| matd3 | canonical | 0.133333 [0.133333, 0.133333] | 0.133333 |

## 配对比较（treatment minus baseline）

| Baseline | Tier | Δ mean [95% CI] | exact p | Holm p | Reject |
|---|---|---:|---:|---:|:---:|
| mappo | canonical | 0.033333 [0.033333, 0.033333] | 1.000000 | 1.000000 | no |
| ippo | canonical | 0.016667 [0.016667, 0.016667] | 1.000000 | 1.000000 | no |
| happo | canonical | 0.016667 [0.016667, 0.016667] | 1.000000 | 1.000000 | no |
| matd3 | canonical | -0.083333 [-0.083333, -0.083333] | 1.000000 | 1.000000 | no |

## 解释

- Holm FWER 0.05 下拒绝 0/4 条主比较。
- 未拒绝不能解释为等效；方向必须结合区间逐项报告。
