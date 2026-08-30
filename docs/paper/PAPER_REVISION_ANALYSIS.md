# 论文修改方向分析（依据 2026-08-30 UAV 链式消融正式结果）

> 冻结时间：2026-08-30（Asia/Shanghai）
> 数据依据：`experiments/paper/uav_imappo_ablation_paper_v2/summary.json` 与 `docs/paper/audits/uav_imappo_ablation_paper_v2_full_validation.json`（valid、0 errors、0 warnings，提交 `36daaf0`）
> 本文是论文重构的决策依据，不替代 `METHODS_DRAFT.md`；正式修改落实后应同步更新本文状态。

## 1. 结果摘要（论文可引用口径）

hard 档 10-seed 均值（3000×200 训练、100 评估 episodes/seed）：

| variant | collision_rate | task_completion |
|---|---:|---:|
| imappo_full | 0.00438 | 0.599123 |
| no_mask | 0.00442 | **0.663275** |
| no_attention | 0.00438 | 0.599076 |
| no_intent_reward | 0.00434 | 0.599689 |
| no_cbf | **0.01946** | 0.599991 |
| no_nli_gate | 0.00432 | 0.600761 |
| prior_only | 0.00430 | 0.602426 |
| no_profile_prior | 0.00832 | 0.613331 |
| identity_oracle | 0.00868 | 0.612191 |
| no_intent | 0.00830 | 0.611683 |

预注册 18 假设 Holm 家族（差异 = variant − imappo_full）**4/18 拒绝**（holm_p=0.0352）：

1. `no_cbf` collision **+0.01508** → CBF 安全层显著降低碰撞率（约 4.4 倍）；
2. `no_profile_prior` collision **+0.00394** → 目标画像规则先验显著降低碰撞率；
3. `no_profile_prior` task **+0.01421** → 同一先验显著小幅降低任务完成率（安全—效率权衡）；
4. `no_mask` task **+0.06415** → 动作掩码显著拖累任务完成率 6.4 个百分点，且碰撞率无差异（+0.00004）。

其余 14 对照全部不显著：语义编码、注意力 critic、NLI 门、意图奖励、one-hot 身份 oracle、意图通道本身、以及学习残差（`prior_only ≈ full`）在 collision/task 上均无可测效应。

**一句话判断：在受控 in-distribution 消融中，可复现的增益全部来自安全层（CBF + 画像先验）与规则先验，语义意图机制整体为 null；动作掩码是显著负组件。**

## 2. 对现有论文叙事的冲击

现有叙事（`RESEARCH_PLAN` C1–C4、`METHODS_DRAFT` §0 主方法）以"语义结构化的意图表示改善安全—效率权衡"为核心。正式结果后逐项裁定：

| 原有主张 | 裁定 | 修改方向 |
|---|---|---|
| C1：语义结构带来策略增益 | **in-distribution 不支持** | 降级为"待泛化实验检验的假设"；主文不得作为已证实贡献 |
| C2：意图一致的安全多智能体优化 | **部分支持** | 保留并强化：CBF 与画像先验的显著安全收益是主结果；但"学习残差"部分需改写（无增量） |
| C3：动态切换与风险泛化 | 未测试（未运行对应 study） | 保持"未来工作/局限"口径 |
| C4：可复现 MARL 评估协议 | **完全支持** | 保留；协议本身（四级隔离、内容寻址 provenance、seed 配对、Holm）可独立成贡献 |
| 动作掩码 | 未立主张，现发现显著负效应 | 新增结论：掩码拖累任务完成率 6.4 个百分点且无安全收益，建议默认移除 |

此外，`no_mask` 的负效应与 `prior_only ≈ full` 的 null 合在一起，说明当前系统里**真正起作用的架构部件是"规则先验 + 安全过滤器"**，RL 学习残差的边际贡献没有被主指标捕获。这必须如实写入讨论。

## 3. 三条重构路线

### 路线 A：安全约束的意图条件 MARL（系统路线，推荐）

- **叙事**：多 UAV 协同中，把意图条件 MARL 拆解为可审计组件，用预注册消融确定哪些组件真正改变安全—效率权衡；结论是安全层（CBF + 目标画像先验）承载全部可复现收益，语义与学习组件目前为 null，动作掩码为负。
- **标题示例**："Safety Layers, Not Semantic Channels: A Preregistered Component-Wise Ablation of Intent-Conditioned Multi-UAV MARL"（或中性版 "A Controlled Decomposition of Intent-Conditioned MARL for Safety-Critical Multi-UAV Coordination"）。
- **目标 venue**：机器人/自动化/航宇方向（`RESEARCH_PLAN` 的 UAV 系统路线），评审更看重安全结果、SITL/实机路径与工程严谨性；语义 null 作为诚实负结果可接受。
- **风险**：需要 SITL/HIL 证据支撑系统路线（12 gates 中仍未闭环）。

### 路线 B：预注册消融 + 诚实负结果（方法论路线）

- **叙事**：以"预注册、逐单元内容寻址、seed 配对 + Holm"的协议为主贡献，展示意图条件 MARL 组件逐个检验后哪些有效、哪些无效；负结果本身是贡献。
- **标题示例**："Preregistered Ablations of Intent-Conditioned MARL: What Moves and What Doesn't"。
- **目标 venue**：ML 可复现性文化强的期刊/会议；竞争小但天花板低于顶刊。

### 路线 C：先补完其余三项研究再定稿（推荐与 A 组合）

- 语义泛化 60 单元是**语义假设最后一个正面机会**（paraphrase/unseen/counterfactual），但 in-distribution null 已显著降低先验；实验已预注册，**不得**以调参或事后口径钓鱼。
- 架构对照 60 单元回答"IMAPPO 组件组合是否优于强基线"；VMAS 100 单元回答外部有效性。
- 顺序建议：**架构（25 h）→ VMAS（40 h）→ 语义泛化（先 calibration 再 60 单元）**；机器当前插电不休眠，可按此连轴执行。

**建议：按路线 A 重构叙事，按路线 C 顺序补齐证据；路线 B 作为兜底定位。** 无论哪条路线，本文第 4 节的文本修改都适用。

## 4. 必须执行的文本修改清单

1. **标题与摘要**：删除任何"语义理解改善安全"的肯定式表述；摘要按"预注册消融 → 安全层收益显著 → 语义机制 null → 掩码负效应"结构重写。
2. **`METHODS_DRAFT.md`**：
   - §0 主方法候选标题改为中性描述（"文本目标概念瓶颈与残差控制"是候选假设，不是已验证方法）；
   - §22 补充执行结果：100/100 完成、4/18 拒绝、null 对照清单；
   - §4/§5.1 意图势函数与画像奖励：注明势函数与意图奖励在主指标无显著效应（`no_intent_reward`、`prior_only`），画像先验的收益来自规则先验上下文而非学习组件；
   - §7.3 CBF 节保留并升级为正式显著结果，但维持"一步离散约束、非连续时间不变性证明"的限定；
   - 新增 Results 章节：主表（第 1 节表格）、Holm 家族表、9 条链式对照的 paired CI 与失败案例（从 summary.json/audit 生成图表）。
3. **`RESEARCH_PLAN.md`**：C1 改为"检验"口径；Phase B 首项勾选（链式消融完成）；明确 Phase C/D 未完成项不变。
4. **允许/禁止表述**（已同步更新 `CURRENT_PROJECT_SUMMARY.md` §7）：
   - 允许：CBF 与画像先验显著降碰撞（Holm 校正后）；画像先验显著小幅降任务完成率；掩码显著降任务完成率且无安全收益；语义机制在 in-distribution 主指标无显著效应。
   - 禁止：语义显著降碰撞、任何"等效"表述（不显著≠等效）、零碰撞保证、单 seed 排序。
5. **讨论部分机制解释**（写作素材，见第 5 节）。

## 5. 讨论部分机制解释（写作素材）

1. **语义通道与画像先验冗余**：full 变体的规则先验已使用 `rule_prior_context=objective_profile`（真实画像上下文），语义编码器在训练分布内无法提供超出该上下文的增量信息；`identity_oracle`（one-hot 身份）同样无增量，说明在 in-distribution 下"知道意图"本身不改变行为——先验已经把意图信息用尽了。
2. **hard tier 由 CBF 主导**：full 碰撞率已接近环境下限（~0.004），剩余语义组件没有可改进的安全余量；no_cbf 的 4.4 倍碰撞率证明瓶颈在安全层而非表示层。
3. **学习残差无增量**：`prior_only ≈ full` 表明残差学习在 3000 episodes 后未超出规则控制器；与 §2.1 中"从零学习弱于规则控制器"的历史观察一致，主方法的核心行为仍是规则性的。
4. **掩码负效应**：姿态掩码可能屏蔽了任务进展所需动作（如允许穿越威胁区的进攻姿态边缘），6.4 个百分点的任务损失大于其名义安全收益（碰撞无差异）——掩码应在未来实现中默认关闭或仅作为诊断通道。
5. **与 CityNav 的一致性**：relevance gate 在真实导航语言上 FAR 0.96 fail + in-distribution 语义组件 null，构成完整证据链：当前语义通道既不能泛化识别（OOD），也没有行为增量（ID）。论文应把两个负结果并列为"语义通道的证据现状"，并指明唯一未测试的正向机会是预注册的语义泛化实验。

## 6. 与 12 gates 及后续工作的关系

- 本次完成只闭合 1/270 项 study（UAV 链式消融）；12 个 final gates 仍基本未闭环（人工偏好数据、HARL 数值交叉复现、SITL、HIL、冻结投稿包等）。
- 剩余三项 study 的执行顺序与预算：架构 60（70.76–96.04 CPU-h）→ VMAS 100（67.52–89.83 CPU-h）→ 语义泛化（先 calibration）。
- 无论后续结果如何，本文的 null/负结果表述不随新数据改写历史；新结果只追加新证据段。
