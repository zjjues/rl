# 论文工作区

本目录是工程与论文之间的唯一事实来源。代码中的重大方法、环境、基线、指标和实验协议变更，都必须同步更新这里的文档。

## 文档结构

- `RESEARCH_PLAN.md`：论文问题、目标贡献、阶段门槛和投稿路线。
- `METHODS_DRAFT.md`：可直接迁移到论文“方法”章节的持续草稿。
- `EXPERIMENT_PROTOCOL.md`：实验假设、基线、消融、统计方法和结果冻结规则。
- `CONTRIBUTION_LEDGER.md`：每项候选贡献所需的代码与证据。
- `RESULTS_LEDGER.md`：正式结果登记表；smoke 结果不得登记为论文结论。
- `RESEARCH_CHANGELOG.md`：重大工程变更及其论文影响。
- `PUBLICATION_READINESS.md`：投稿门槛、当前距离与最短推进路径。
- `SUBMISSION_READINESS_GATE.md`：12 项机器可审计的最终投稿门槛、阈值与 CI 用法。
- `UAV_SEMANTIC_GENERALIZATION_PROTOCOL.md`：六表示、无标签泄漏、seed-level exact/Holm 的正式泛化协议。
- `CURRENT_PROJECT_SUMMARY.md`：暂停时的单一项目状态入口、证据边界与恢复方法。
- `EXPERIMENT_ENVIRONMENT.md`：本地 Conda 环境、模型和重建方式。

## 强制规则

1. 任何声称“semantic”的主实验必须使用保留语义几何结构的编码器；hash 或随机向量只能作为对照组。
2. 任何基线必须运行其真实算法实现。兼容性替代模型必须以准确名称记录，不得借用其他算法名称。
3. 正式结果必须包含完整配置、代码提交、环境信息、所有 seed、置信区间和原始逐 seed 指标。
4. smoke、调试和中断运行必须与正式论文结果分目录保存。
5. 修改奖励、环境动力学、评价指标或训练更新逻辑时，必须在 `RESEARCH_CHANGELOG.md` 中记录。
