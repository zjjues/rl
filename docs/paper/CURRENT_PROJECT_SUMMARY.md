# 多 UAV 语义意图 MARL 工程：当前完整状态总结

> 冻结时间：2026-08-24 21:21（Asia/Shanghai）
> 分支：`testv1`
> 本次封存前基线提交：`c4a254163b1e1cc5ee710a95f5da63e987f2bf1e`
> 总体结论：**工程/实验协议准备度约 96%，论文证据准备度约 48%–52%；当前不是顶刊投稿 ready。**

本文件是暂停后恢复工作的单一入口。具体数值的原始依据仍以对应 JSON artifact、manifest、result 和专项文档为准；本文件不把 smoke、pilot、calibration 或作废结果升级为正式论文证据。

## 紧急停止点（最终封存）

- 已立即停止继续训练、功能扩展和全量回归；没有启动语义泛化 calibration 或 paper 运行。
- 最后一项代码改动仅冻结六表示语义泛化协议、统计口径和机器门禁：预计 `60` 个正式结果，当前 `0/60`；投稿门禁当前 `0/12`。
- 新增协议相关 focused tests 已分别通过（协议 7 项、统计 5 项；门禁/协议组合检查 15 项）。这些改动之后**未重新执行全量测试**；最近一次全量基线仍为 `191 passed`。
- 进程查询因当前 Windows 权限被拒绝；本轮没有发起任何训练进程，先前确认也无活跃训练。后续工作全部暂停，恢复时从本文件和 `CONTINUATION_STATE.md` 开始。

## 1. 当前实验已经结束运行

- 当前没有活跃的 `rl-test` Python 训练进程。
- 正在执行的正式消融单元已按用户要求中止：`uav_imappo_ablation_paper_v2 / imappo_full / seed=7`。
- 原子 checkpoint 位于 `experiments/paper/uav_imappo_ablation_paper_v2/imappo_full/seed_7/training_checkpoint.pt`。
- checkpoint：`next_episode=2050/3000`，大小 `6,847,602` bytes，SHA-256=`a0074a57914a2752209da085c170cf75d08a48edd3df58e4b01eab22276f05c1`。
- checkpoint schema=`episode_boundary_training_v1`；implementation SHA-256=`c4e55820d537dda168d8638dc6849f9b55e449b5f2c09cadb9ab88e944a40573`；registered result-protocol SHA-256=`ddb4f154a4f0564d06f6e8029fe8f14fe7228d3f4c3e96209addb4c7b23a5129`。
- `manifest.json` 保留从 clean commits `053c316...` 和 `bde3111...` 发起的两次 invocation；两次 implementation fingerprint 相同，第二次 `--resume` 身份检查已实际通过。
- manifest 的 `status=running` 表示研究单元尚未形成结果；这是运行器协议状态，不得手工改成 `partial` 或 `complete`。
- 当前没有 `result.json`、`summary.json` 或正式统计，所以活动 paper 消融有效进度仍是 **0/100**。`2050/3000` 只是一个可恢复的单运行训练进度。

## 2. 工程当前实现的功能

### 2.1 多 UAV 决策与语义意图

- 8-UAV/6-target 连续控制调度环境，支持安全、任务、能耗、时间、威胁等奖励分量及确定性 reset seed。
- 冻结 MiniLM 语义编码器和 NLI cross-encoder；模型 revision、投影种子、码本种子及 ridge adapter 参数均写入研究配置。
- 语义表示、随机稠密、legacy hash、真实 catalog one-hot 和 no-intent 对照路径已分离，禁止再把 hash 码本称为语义 embedding。
- 偏好合同固定为 distance、energy、safety、task、time、threat 六个可协商轴；collision 是不可由语言降低的独立安全约束。
- NLI prototype gate、objective-profile rule prior、动作 mask、potential shaping、learned residual 和 intent-attention critic 均有显式开关，可用于链式单因素消融。
- 支持 seen、paraphrase、unseen、反事实偏好和 episode 内动态切换诊断；表示检索指标与行为指标分开报告。

### 2.2 强基线与安全层

- 已实现并接入 IMAPPO、MAPPO、local IPPO、HAPPO 和 MATD3。
- HAPPO 使用独立 actor/optimizer、随机 agent 更新顺序、逐 actor 新旧 log-prob ratio 与前序乘积 factor、actor 后 centralized critic 更新；已经完成官方 HARL 源码协议级审计，但尚缺官方框架数值交叉复现。
- 控制策略支持 learned residual + objective-conditioned rule prior。
- 安全过滤支持 cyclic pairwise CBF、QP-CBF 和无过滤对照；诊断记录约束违例、可行性、最小间距和 fallback 情况。
- IMAPPO/HAPPO/MATD3 支持完整训练态 checkpoint：网络、优化器、rollout/replay、更新相位、累计日志、下一 episode、私有及全局 RNG。

### 2.3 跨场景与真实性路径

- VMAS `navigation` 和 `dispersion` 已接入五算法原生 episode return 协议。
- PyBullet Crazyflie 刚体/旋翼路径用于跨动力学、延迟、drag、ground effect、downwash 与鲁棒 QP 评估。
- Betaflight SITL 已有单机和双机链路 smoke；尚未形成多机冲突 policy-in-loop 证据。

### 2.4 研究治理与统计

- `smoke`、`pilot/calibration`、`paper`、`frozen` 四级证据目录隔离。
- paper 配置强制至少 10 seeds、每 seed/风险档至少 100 个最终评估 episode，并拒绝 dirty-worktree 新启动和结果覆盖。
- 训练、monitor、collision probe、final tier 和 query 使用注册的配对 seed 公式。
- 已实现 IQM、bootstrap 95% CI、paired differences、win rate、performance profile、exact paired tests 和 Holm 多重校正。
- artifact 支持 checksums、canonical protocol hash、implementation hash、严格 partial audit、跨实现 resume 拒绝和原子写入。
- 新增版本化 submission-readiness gate；正式研究会复用完整 artifact validator，人工偏好会重算 JSONL 审计，外部系统证据必须满足冻结字段、样本量和误差阈值。当前审计为 `not_ready`、0/12 final gates met。
- 最新完整代码回归为 **191 passed, 14 warnings, 8.14 s**；本轮新增独立 readiness 审计工具及测试，但不修改训练源码，训练 implementation fingerprint 仍与 checkpoint 一致。

## 3. 已完成且可保留的证据

| 证据 | 状态 | 可以支持什么 | 不能支持什么 |
|---|---|---|---|
| UAV 十变体消融 calibration | 10/10，artifact valid，0 errors/warnings | 路径连通、注册对照正确、运行时间预算 | 机制效果、排序、显著性、等效性 |
| UAV 六算法架构 calibration | 6/6 | IMAPPO/no-mask/MAPPO/IPPO/HAPPO/MATD3 可执行及预算 | 算法优越性 |
| VMAS navigation calibration | 5/5 | 五算法原生 return 路径和预算 | 单 seed 排名 |
| VMAS dispersion calibration | 5/5 | 第二公开场景复现路径和预算 | 单 seed 排名 |
| CBF 实现等价与性能基准 | CPU 约 10.08×、CUDA 约 66.24× 加速 | 工程实时性改善与数值等价 | 安全效果或形式化保证 |
| QP-CBF adversarial pilot | 5 seeds，碰撞率有经验改善 | 风险降低与不可行率量化 | 零碰撞保证 |
| Crazyflie/鲁棒动力学 pilot | 5 seeds，组合扰动下违规显著降低但未归零 | 经验鲁棒性与跨动力学风险降低 | HIL/实机或形式化鲁棒保证 |
| Betaflight SITL smoke | 单机/双机通信与控制链路通过 | 接口可运行 | 多机冲突闭环安全 |
| AerialVLN 外部语料审计 | source/hash/license/split 固定 | 真实导航语言 OOD 负对照 | 六维偏好真值 |
| CityNav one-shot final OOD | 32,637 条，冻结执行 | relevance gate 的真实负结果 | 重新调参后的第二次 final |

### 3.1 关键 calibration 点值

UAV 消融 hard-tier `(collision_rate, task_completion)`：full `(0.003, 0.608392)`、no-mask `(0.004, 0.672851)`、no-attention `(0.003, 0.608440)`、no-intent-reward `(0.003, 0.608409)`、no-CBF `(0.018, 0.608574)`、no-NLI-gate `(0.003, 0.609192)`、prior-only `(0.003, 0.608545)`、no-profile-prior `(0.007, 0.617855)`、identity-oracle `(0.007, 0.617827)`、no-intent `(0.007, 0.617782)`。这些全部是单 seed calibration 点值。

VMAS navigation 单 seed native return：attention-PPO `-2.711723`、MAPPO `-0.790685`、IPPO `-1.913966`、HAPPO `0.814603`、MATD3 `1.513628`。VMAS dispersion 对应点值为 `0.050000/0.016667/0.033333/0.033333/0.133333`。禁止据此排序。

### 3.2 关键负结果

- CityNav 冻结 one-shot final：accepted=`31,381/32,637`，false-accept rate=`0.961516`，Wilson 95% CI=`[0.959374, 0.963549]`，预注册 outcome=`fail`。该数据不得重跑、删改或用于继续调 gate 后再称 final。
- 多个动态偏好 pilot 表明 energy/collision 在线响应失败，safety 行为 CI 跨零，learned residual 尚未证明贡献。
- UAV 消融 calibration 中多条机制差异接近零，是正式实验可能产生负结果的风险信号，不是等效性证据。
- VMAS 当前任务不包含自然语言偏好真值，只能作为算法/环境外部有效性补充。
- CBF/QP 与鲁棒动力学实验均仍有约束违例或规划距离违规，不能声称严格安全。

## 4. 已作废但必须保留的实验

- `uav_imappo_ablation_paper_v2_superseded_pre_result_fingerprint_20260824`：seed 7 完成，但旧 result/manifest 没有实现指纹，严禁聚合。
- `uav_imappo_ablation_paper_v2_superseded_resume_objective_drift_20260824`：seed 7/11 fragment 因 resume 注入冗余 `objectives` 导致 protocol hash 漂移，strict validator 判 invalid，严禁聚合。
- 旧七轴语言奖励合同下的架构/消融 smoke 已 superseded；只能用于说明工程演进。
- legacy Stage7 的 hash-vs-one-hot 观察只能作为生成假设的历史 pilot，不能归因为语义理解。

作废目录和反证审计必须保留，不得迁移 result、手改哈希或重新解释为正式结果。

## 5. 当前正式 paper 实验协议

### 5.1 UAV 链式因果消融

- 10 variants × 10 seeds=`100` 个训练单元。
- 每单元 `3000×200` training，训练期 monitor 20 episodes，hard final 100 episodes。
- 9 条注册链式对照、collision/task 两个主指标，共 18 个 Holm 家族假设。
- 当前完成的有效 result=`0/100`；seed 7 仅训练到 episode 2050 checkpoint。
- 校正预算=`70.65–78.87 active CPU-hours`，不是 GPU device-hours。

### 5.2 UAV 架构对照

- IMAPPO、IMAPPO-no-mask、MAPPO、IPPO、HAPPO、MATD3 × 10 seeds=`60` 个训练单元。
- 校正预算=`70.76–96.04 active CPU-hours`。
- 仍缺 clean multi-seed 正式结果和官方 HARL 数值交叉复现。

### 5.3 VMAS 跨场景复现

- navigation 和 dispersion 各 5 algorithms × 10 seeds=`50` 单元。
- 校正预算分别为 `33.42–44.36` 与 `34.09–45.47 active CPU-hours`。
- 两场景合计 `67.52–89.83 active CPU-hours`；尚未运行 paper multi-seed。

## 6. 距离顶刊投稿仍缺什么

以下六项是不可由更多 smoke 或文档替代的硬门槛：

1. 独立招募、writer-disjoint 的人工偏好 train/dev/test，含 consent、伦理边界、独立 reviewer 和 adjudication 记录。
2. 在多来源训练数据上重新训练 relevance gate；用未访问的人类 preference test 和新的 OOD final 冻结验证。CityNav 不得复用为第二次 final。
3. clean snapshot 上完成 UAV 10-seed 架构对照和 10-seed/100-run 链式消融，生成配对 CI、Holm 结果、失败案例和冻结 artifact。
4. 完成 VMAS 两场景 10-seed 复现，并用官方 HARL 框架做数值交叉核验，而不只是源码协议审计。
5. 完成多 UAV 冲突场景的 policy-in-loop SITL，覆盖通信延迟、丢包、动力学与安全 fallback。
6. 至少完成 HIL 或受控实机验证，并使延迟/动力学边界与独立系统辨识覆盖率一致。

因此当前项目可以诚实定位为“具有严谨 provenance、强基线路径、负结果约束和多层安全验证的完整研究平台”，不能定位为“已证明语义 MARL 优势”或“顶刊投稿完成稿”。工程准备度约 96% 不等于论文准备度；真正缺口主要是昂贵的多种子证据、独立人工数据和真实系统验证。

机器审计将上述六类硬缺口细分为 12 个二值 final gates，当前 `0/12` 通过。语义泛化的完整 artifact 与专用 seed-level Holm 统计分别设门，避免“训练完成但统计口径错误”仍被标记 ready。该数字表示尚无最终门槛完整闭环，不是开发完成百分比；calibration、pilot 和已实现代码不会被错误折算为正式 evidence gate。

## 7. 当前允许与禁止的论文表述

允许：

- 研究平台支持冻结语义模型、强 MARL 基线、链式消融、跨场景复现、配对统计、严格 provenance 和精确 resume。
- calibration 证明所有主要路径可执行，并给出了 active CPU-time 预算。
- CityNav final 是 relevance gate 不能迁移到真实城市导航语言的明确负结果。
- CBF/QP/鲁棒动力学 pilot 显示经验风险降低，但仍存在违例。

禁止：

- “语义意图显著降低碰撞率”或“IMAPPO 显著优于 MAPPO/HAPPO/MATD3”。
- 根据单 seed calibration 排名、宣称显著性或等效性。
- 把 superseded fragment、当前 checkpoint 或历史 hash pilot 纳入正式统计。
- 把 cyclic/QP-CBF 描述为零碰撞形式化保证。
- 把 VMAS architecture-only 结果描述为语言泛化证据。
- 宣称已有 HIL、实机或完整多机冲突 SITL 证据。

## 8. 精确恢复方法

恢复前必须先检出保存本总结及 checkpoint 的 `testv1` 提交，并确认没有本地代码修改。使用已有 Conda 环境：

```powershell
D:\Programs\anaconda3\envs\rl-test\python.exe run_research_study.py --config configs\research\uav_imappo_ablation.paper.json --only-variants imappo_full --only-seeds 7 --resume
```

恢复顺序：

1. `git status --short --branch`，确认仅预期的实验 artifact 会在运行后变化。
2. 读取 checkpoint，核对 `next_episode=2050` 及两个身份 SHA-256。
3. 运行上述 `--resume`；任何 implementation/protocol mismatch 都应停止，不得绕过。
4. 完成后执行 strict partial validator，预期只有 `imappo_full×seed7` 一个 result、99 missing、无 summary、0 errors/warnings。
5. 将 result/audit/文档提交到 clean snapshot 后，才运行 seed 11 的跨提交 resume 验证。

若不继续这批正式实验，应保留 checkpoint、manifest、config 和本总结，不要把 manifest 手工改为 complete。

## 9. 关键文件入口

- 当前恢复状态：`docs/paper/CONTINUATION_STATE.md`
- 本总结：`docs/paper/CURRENT_PROJECT_SUMMARY.md`
- 可发表性审计：`docs/paper/PUBLICATION_READINESS.md`
- 投稿机器门槛：`docs/paper/SUBMISSION_READINESS_GATE.md`
- 当前机器审计：`docs/paper/audits/submission_readiness_v1.json`
- 正式结果与负结果：`docs/paper/RESULTS_LEDGER.md`
- 方法草稿：`docs/paper/METHODS_DRAFT.md`
- 实验协议：`docs/paper/EXPERIMENT_PROTOCOL.md`
- 精确恢复协议：`docs/paper/EXACT_TRAINING_RESUME_PROTOCOL.md`
- 分块 provenance：`docs/paper/CHUNKED_PAPER_PROVENANCE_PROTOCOL.md`
- 人工偏好协议：`docs/paper/PREFERENCE_ANNOTATION_PROTOCOL.md`
- 研究变更日志：`docs/paper/RESEARCH_CHANGELOG.md`
- 活动正式配置：`configs/research/uav_imappo_ablation.paper.json`
- 活动实验目录：`experiments/paper/uav_imappo_ablation_paper_v2/`
- 核心运行入口：`run_research_study.py`
- artifact validator：`validate_research_artifact.py`

## 10. 收口判定

当前实验进程已经停止，最新可恢复训练状态、全部已知证据等级、负结果、作废原因、资源预算、论文边界和后续恢复步骤均已记录。正式语义泛化协议现已冻结为 60-run/12-hypothesis seed-level 设计，但没有运行 calibration 或 paper artifact。`--require-ready` 在当前 12 项 blocker 下按设计返回非零。除非用户明确要求继续，后续不应自动恢复训练。
