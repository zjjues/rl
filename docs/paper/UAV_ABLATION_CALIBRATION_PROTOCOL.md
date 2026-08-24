# UAV 链式消融正式注册与 calibration 协议

## 论文问题

本实验检验候选方法中的九个可分离机制是否分别改变 hard-risk UAV 场景的碰撞率或任务完成率：意图动作掩码、意图注意力 critic、势函数塑形、pairwise CBF、NLI prototype gate、学习残差、语义 rule prior、连续语义表示和完整意图通道。它是因果机制消融，不承担强算法基线比较；MAPPO、IPPO、HAPPO 与 MATD3 由独立 architecture study 负责。

## 正式注册

正式配置为 `configs/research/uav_imappo_ablation.paper.json`。十个变体构成一棵以 `imappo_full` 为根的九边链式对照图；每个非根变体恰有一个父对照，`changed_fields` 必须与实际变体字典的逐字段差异完全相等。正式协议固定十个共同随机数种子、每 seed 3000×200 training steps、每 100 episodes 一次 20-episode 训练监控及同规模 collision probe、hard tier 最终 100 episodes、每 50 episodes 原子 checkpoint。主要统计族仅包含注册的 `comparison × {collision_rate, task_completion} × hard`，使用配对检验、Holm 校正和 bootstrap 区间。

`no_nli_gate` 只检验 prototype gating 相对于 ungated NLI entailment 的机制作用，不能在 CityNav 已失败后被解释为具有开放域语言拒答能力。`identity_oracle` 是 canonical-label 身份上界，不是真实文本理解；`no_cbf` 检验软安全过滤器的行为贡献，不构成形式安全证明。

## Calibration 注册

`configs/research/uav_imappo_ablation.calibration.json` 从正式变体定义机械复制，只改变 evidence level、seed 数、训练/评估工作量和 checkpoint cadence：seed 7、100×100 training steps、一次 20-episode monitor、一次同规模 collision probe、hard tier 20 episodes。它只用于：

- 验证当前六轴语言安全合同下十条路径均能训练和落盘；
- 记录每变体 wall/process-CPU/CUDA audit；
- 按训练、周期监控、collision probe、最终评估的逐项 workload 外推正式预算；
- 识别异常计时并决定安全的 resume 分块。

单 seed calibration 的任何效果点值都禁止用于算法排序、显著性或论文主张。运行时预算器要求 calibration 与 paper 的变体字典逐字段完全相同；同名但实现漂移会硬失败。

## 运行前判据

1. 两份配置必须通过 `run_research_study.py --dry-run --allow-dirty`；正式运行仍要求 clean Git，不使用 `--allow-dirty`。
2. calibration 的 10 个 result、summary、manifest 与 checksums 必须通过 artifact audit；不得把 partial manifest 当完成。
3. 预算只称为 active CPU-hours 或 wall-hours；没有设备级计时就不得称 GPU-hours。
4. calibration 后仍不提升论文效果证据等级；只有 clean 10-seed paper artifact 才能关闭“算法新颖性与因果消融”门槛。

## 2026-08-24 执行记录

运行前注册提交为 `b974937`。十个变体均完成 seed 7 的 100×100-step 训练、20-episode monitor、20-episode collision probe 与 hard-tier 20-episode final evaluation。artifact audit 为 `valid`：10/10 results、14 checksum entries、9 条契约对照、0 errors、0 warnings；训练完成后无残留 checkpoint。

各变体 process CPU 为 40.16–58.23 s，wall 为 44.89–86.86 s。逐字段同构预算器把 16,000-step calibration workload 外推到每个正式 run 的 860,000 steps，100-run 正式计划为 **70.65–78.87 active CPU-hours**；这是排除宿主挂起的进程 CPU 规划量，不是 GPU device-hours。

单 seed 诊断中，full/no-CBF collision rate 为 0.003/0.018；但 no-attention、no-intent-reward、no-NLI-gate、prior-only 与 full 的多项点值接近。该现象只用于确认正式实验必须允许零效应或负结果，不能据此声称 CBF 有效或其余机制无效。审计、预算和自动报告分别位于 `docs/paper/audits/uav_imappo_ablation_calibration_v1_artifact_audit.json`、`docs/paper/audits/uav_imappo_ablation_calibrated_runtime_plan.json` 和 `docs/paper/generated/uav_imappo_ablation_calibration_v1_active_time/`。
