# 多 UAV 语义意图 MARL 工程：当前完整状态总结

> 冻结时间：2026-08-30 15:06（Asia/Shanghai）
> 分支：`testv1`
> 本次封存前基线提交：`36daaf0`（UAV 链式消融最终全量校验提交）
> 总体结论：**UAV 链式消融 100/100 完成并通过全量验证：安全层（CBF + 目标画像先验）显著、语义机制主指标 null、动作掩码显著负效应。其余三项 paper 研究（架构 60、VMAS 100、语义泛化 60）未启动。长期顶刊投稿目标未完成；论文重构分析见 `PAPER_REVISION_ANALYSIS.md`。**

本文件是暂停后恢复工作的单一入口。具体数值的原始依据仍以对应 JSON artifact、manifest、result 和专项文档为准；本文件不把 smoke、pilot、calibration 或作废结果升级为正式论文证据。

## 0. 本轮任务完成状态与目标差距（最终封存）

- **本轮收口任务：已完成。** 已停止继续训练和功能扩展，保存可恢复 checkpoint，汇总完成状态与证据边界，并将全部内容推送到 `testv1` 的 `833c0a97ddda1dceec984c3df0b81e10866de7b1`。
- **工程实现任务：基本完成但未最终验收。** 强基线、安全层、语义表示、跨场景适配、统计与 provenance 路径均已实现；最近一次全量基线为 `191 passed`。语义泛化新增 focused tests 已通过，但新增改动后未重跑全量回归。
- **正式实验任务：进行中（1/4 类完成）。** 已注册四类 paper study 合计需要 `270` 个正式 result（语义泛化 60、UAV 架构 60、链式消融 100、VMAS 两场景 50），当前可纳入论文主统计的有效 result 为 **`100/270`**（UAV 链式消融 100/100，经全量验证 valid、0 errors、0 warnings）。
- **顶刊投稿任务：未完成。** 机器审计的 **`0/12`** 个 final gates 通过；目前只能证明研究平台、协议和执行路径较完整，不能证明核心方法相对强基线具有稳定、显著且可迁移的优势。
- **运行状态：已停止。** 没有启动语义泛化 calibration 或 paper 运行；本轮没有发起任何训练。最后一次进程复查受 Windows 权限限制，但此前已确认没有活跃训练进程。

### 0.1 到“诚实可验证的顶刊投稿准备”的精确缺口

| 证据项 | 当前状态 | 投稿前必须达到 | 尚缺 |
|---|---:|---:|---:|
| 独立人工偏好数据 | 无冻结正式数据集 | writer-disjoint、独立复核、consent/伦理与 adjudication 齐全 | 整项未完成 |
| 冻结语言 relevance gate | 无 final v2 证据 | 多来源训练，在未访问 preference/OOD final 上通过 | 整项未完成 |
| 语义行为泛化 | `0/60` | 6 variants × 10 seeds，并完成 12 个注册假设的 seed-level exact test + Holm | 60 results 与统计 artifact |
| UAV 强基线 | `0/60` | IMAPPO/MAPPO/IPPO/HAPPO/MATD3 等 6 variants × 10 seeds | 60 results |
| UAV 链式因果消融 | **100/100 完成**，full validation valid | 10 variants × 10 seeds，配对 CI、Holm、失败案例与冻结 artifact | 已完成：4/18 Holm 拒绝（CBF/画像先验显著；语义 null；掩码负效应） |
| VMAS 跨场景复现 | `0/50` | navigation 与 dispersion 各 5 algorithms × 10 seeds | 50 results |
| 官方 HARL 数值交叉核验 | 仅完成源码/协议级审计 | 官方框架数值复现并记录差异 | 整项未完成 |
| 多 UAV 冲突 SITL | 仅有组件与 pilot | policy-in-loop，覆盖延迟、丢包、动力学及 fallback | 整项未完成 |
| HIL/受控实机 | 无正式证据 | HIL 或受控实机，且覆盖独立系统辨识边界 | 整项未完成 |
| 冻结投稿包 | 无 | 匿名、校验和完整、所有门禁一致通过 | 整项未完成 |

因此，“距离顶刊还有多远”的硬指标不是约 50% 这一主观比例，而是：**12 个关键门禁仍全部未闭环、至少 170 个注册训练结果尚未形成、人工数据/盲测/SITL/HIL/官方交叉复现与最终冻结包均缺失。** 只有这些证据完成且机器审计变为 `ready`，才可宣称达到投稿准备门槛；是否录用仍由创新性、结果强度和审稿判断决定。

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
- 2026-08-30 完成有效 result=**100/100**，full validation valid；18 假设 Holm 家族 4 拒绝：CBF 与画像先验显著降碰撞、画像先验显著小幅降任务完成率、掩码显著降任务完成率（+0.064）且无安全收益；语义机制全部对照不显著（详见 `RESULTS_LEDGER.md`）。
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
- UAV 链式消融正式结果（10 seeds、100 评估 episodes/seed、Holm 校正）：CBF 与目标画像先验显著降低 hard 档碰撞率；画像先验显著小幅降低任务完成率；动作掩码显著降低任务完成率且无安全收益。
- 语义编码、NLI 门、意图奖励、注意力 critic、one-hot oracle 与意图通道在 hard 档碰撞率与任务完成率上无显著效应（预注册 18 假设族）。

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
- 论文修改方向分析：`docs/paper/PAPER_REVISION_ANALYSIS.md`
- 核心运行入口：`run_research_study.py`
- artifact validator：`validate_research_artifact.py`

## 10. 收口判定

当前实验进程已经停止，最新可恢复训练状态、全部已知证据等级、负结果、作废原因、资源预算、论文边界和后续恢复步骤均已记录。正式语义泛化协议现已冻结为 60-run/12-hypothesis seed-level 设计，但没有运行 calibration 或 paper artifact。`--require-ready` 在当前 12 项 blocker 下按设计返回非零。除非用户明确要求继续，后续不应自动恢复训练。
