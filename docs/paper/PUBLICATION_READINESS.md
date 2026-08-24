# 可发表性差距审计

## 2026-08-24 机器化 submission gate

新增 `submission_readiness.v1.json` 和独立审计器，把最终投稿条件拆为 12 个 critical gates：独立人工偏好、blind language gate、语义泛化 artifact、语义泛化统计、UAV 强基线、UAV 因果消融、两项 VMAS、官方 HARL 数值交叉核验、多机冲突 SITL、HIL/实机、冻结匿名 artifact。当前结果为 `not_ready`、**0/12 final gates met**。

这里的 0/12 是严格完成判定，不是对既有工程进度的否定，也不是录用概率：calibration、smoke、代码实现和负结果审计仍贡献于约 48%–52% 的阶段性证据准备度，但任何一项都不能替代对应 final artifact。语义泛化 paper/calibration 配置、无泄漏 validator 和 seed-level exact/Holm generator 已完成；因尚无 60/60 artifact，对应两个 final gates 仍不通过。

## 2026-08-24 长实验可靠性更新

已关闭“单个算法×种子中断会丢失全部训练进度”的工程缺口：IMAPPO/HAPPO/MATD3 现在具备绑定协议与源码指纹的 episode 边界原子恢复，CPU 确定性测试验证连续与中断恢复路径逐 tensor 相等；最近完整回归 **183 passed**。六个 `*.paper.json` 配置已通过 dry-run，遗留 `uav_imappo_main.paper.json` 中伪 `concat` critic、pilot 等级和 50-episode 评估已修正为独立的 `uav_imappo_main_paper_v2`。

这提升的是实验可恢复性与 provenance，不提升论文效果证据等级。navigation/dispersion 五算法 100-episode calibration 已完成；预算器计入训练期监控、UAV collision probe 和算法差异，并将最终评估 100 episodes 与训练监控 20 episodes 解耦。navigation 50-run 预算为 **33.42–44.36 active CPU-hours**。MAPPO 同配置复跑得到完全相同 return，但 CPU time 从异常的 457.98 s 回落至 141.28 s，证明旧单次计时不可直接外推。正式 multi-seed 结果及六个硬缺口仍未关闭，因此总体 paper 证据仍维持 **48%–52%**。

UAV v3 六算法 calibration 也已完成：正式 60-run 预算从历史 smoke-wall 的 60.9–105.2“GPU-hours”更正为 **70.76–96.04 active CPU-hours**。HAPPO 独立 actor 顺序更新是主成本（36.17–48.22 h），MATD3 为 13.91–20.24 h。MATD3 calibration 的 wall=3711.34 s、CPU=121.42 s，再次显示宿主时间污染；论文只能引用 process-CPU active-time 作为规划量。calibration 仍为单 seed，不增加效果证据等级。

UAV 十变体链式消融现在另有正式 `paper` 注册：10 seeds、100 final episodes、18 个 Holm 家族假设，共 100 个训练单元。与其逐字段同构的单 seed calibration 已通过 10/10 result、14 checksums、9 条对照和 0 warning 审计，正式预算为 **70.65–78.87 active CPU-hours**。calibration 中多条机制点差近零，只能视为“正式实验可能产生负结果”的设计预警；它没有关闭因果消融门槛。

首个正式消融单元还暴露并关闭了分块 provenance 缺口：旧完成结果没有保存 checkpoint 的实现指纹，故该单元被完整保留但判为 superseded，不能进入统计。新协议把 canonical result-protocol hash 与 implementation hash 同时写入 result、manifest 和 resume history，拒绝跨实现 resume，并可严格审计 `valid_partial` 而不生成提前统计。工程/协议准备度小幅提高，但因正式有效单元仍需重跑，paper 证据比例不增加。

修复后 `imappo_full × seed 7` 曾从 clean `317204a` 重跑并通过 `valid_partial`；但第一次跨 commit resume 又揭示相同单目标 spec 被注入冗余 `objectives` 字段，seed 11 被 strict validator 拒绝。两 seed fragment 均已 superseded，merge 幂等性现由字典与 hash 双测试固定。活动正式消融进度因此诚实回到 0/100，总体 paper 证据比例继续保持 48%–52%。

## 2026-08-24 最新判定（覆盖下文历史比例）

工程/协议准备度约 **95%**，论文证据准备度约 **48%–52%**；比例表示门槛完成度，不是录用概率。下降不是工程退步，而是 CityNav 真正一次性终测推翻了 relevance gate 的可迁移性假设。当前不能诚实声称系统能区分“操作偏好”与新的真实城市导航目标语言。

新增已完成项：CityNav 文本访问前预注册、32,637 条 one-shot OOD 终测、VMAS navigation/dispersion architecture-only 五算法 smoke、两套 10-seed paper 配置和 runtime plan、正式人工 preference 数据冻结入口。CityNav FAR=96.15%（95% CI 95.94%–96.35%），预注册 fail；因此语言门槛明确未通过。

当前六个不可替代硬缺口：

1. 独立招募的 writer-disjoint 人工偏好 train/dev/test 与伦理/consent 记录；
2. 重新训练的多来源 relevance gate，并在未访问的人类 preference test 与新的 OOD test 上冻结验证；CityNav 不得复用作调参或第二次 final；
3. clean commit 上的六轴 UAV 10-seed 架构、10-seed 链式消融和统计主结果；
4. VMAS 两场景 10-seed 原生 return 复现及官方 HARL 数值交叉核验；
5. 多机冲突 policy-in-loop SITL，而非仅有单/双机链路 smoke；
6. HIL 或受控实机证据，以及与系统辨识覆盖率一致的延迟/动力学安全边界。

在这些缺口关闭前，合适定位仍是“完整研究平台 + 已记录关键负结果”，不是顶刊 ready。两套 VMAS 计划经逐算法 workload calibration 后合计为 **67.52–89.83 active CPU-hours**；UAV 架构与消融的校正预算分别为 **70.76–96.04** 与 **70.65–78.87 active CPU-hours**。它们均不是 GPU device-hours，且单 calibration seed 尚未覆盖运行时间方差。任何 paper run 都不得在 dirty worktree 上启动。

## 总体判断

当前工程已经从“概念原型”进入“有否定结果约束、动态干预、约束审计、跨动力学验证、artifact 验证和经验鲁棒化的研究框架”，但尚未形成可投稿的完整算法证据。2026-08-20 的计算语义审计撤回了旧 40-run 架构 pilot 的 MAPPO 解释；工程准备度因协议验证和链式消融提升到约 **90%**，论文证据准备度则诚实下调到约 **52%–55%**。这两个比例是项目门槛完成度，不是录用概率。

现在可以诚实地区分真实语义表示、身份 oracle、随机/哈希对照，并在共同随机数条件下评估 seen、paraphrase、unseen、单目标反事实和 episode 内切换。UAV v2 已修复威胁不可观测和伪任务进展；5-seed 动态 pilot 显示冻结 NLI 原型门控只在 distance/energy/time/threat 的静态可控性及 safety/threat 的在线响应上成立。energy/collision 在线响应失败，safety 行为 CI 跨零，残差 RL 没有贡献。当前方法候选应表述为“语言偏好解码 + 经典控制先验 + 可审计软安全层”，而非语言 MARL 优势或严格安全保证。

## 十项投稿门槛

| 门槛 | 当前状态 | 证据或缺口 | 投稿前判定 |
|---|---|---|---|
| 1. 可证伪研究问题 | 基本完成 | 已冻结 H1–H4，并区分表示诊断与行为证据 | 通过 |
| 2. 真实语义编码 | 部分完成 | MiniLM/NLI revision 固定且有 5-seed pilot；跨措辞极性仍不稳定 | 未通过 |
| 3. 公平表示基线 | 基本完成 | true one-hot、random dense、legacy hash；paraphrase identity oracle 已明示 | 通过 |
| 4. 无泄漏泛化协议 | 基本完成 | 19 train intents、2 seen、4 paraphrase、6 unseen；自动拒绝标签/文本泄漏 | 通过 |
| 5. 公平统计与复现 | 部分完成 | 精确 paired test/Holm、resume provenance、checksum 和计算语义 validator 已实现；旧 10-seed 架构 artifact 因算法身份错误判 invalid，尚无 clean paper 主结果 | 未通过 |
| 6. 强算法基线 | 部分完成 | MAPPO、local IPPO、MATD3、I-MAPPO 及独立 actor/顺序 factor HAPPO 的 v3 smoke 已通过实际路径审计；HAPPO 对照官方 HARL 协议并固定参考 commit，仍缺 clean v3 paper run 与官方框架数值交叉复现 | 未通过 |
| 7. 算法新颖性与因果消融 | 部分完成 | 10 变体/9 条链式单因素契约的完整 smoke 已通过 10/10 result 与 checksum 审计；只有 1 seed/10 episodes，pilot 未执行，学习贡献仍未成立 | 未通过 |
| 8. 跨环境与规模泛化 | 部分完成 | 4→8/12/16 smoke、VMAS navigation/dispersion 5-seed pilots 已完成；VMAS 尚非语言任务，规模无多 seed | 未通过 |
| 9. UAV 真实性 | 部分完成 | 已有 5-seed Crazyflie 刚体/旋翼 pilot；命令约束与物理间距存在失配。Betaflight SITL 单机/双机 smoke 通过（2026-08-02）；仍缺冲突场景、HIL/真机 | 未通过 |
| 10. 冻结论文产物 | 部分完成 | 自动 CSV/图表/统计报告/哈希 manifest 已支持架构和契约消融；旧生成目录已加入失效通知。无 clean-commit frozen 主结果、最终失败案例与匿名复现包 | 未通过 |

## 本轮已关闭的高风险问题

### 真实 one-hot 基线

旧实现把 attack/stealth/frozen 三模式编码称为 one-hot，但它不是 25 个自然语言意图的身份基线。当前 `IntentLibrary.create_onehot` 为每个 catalog intent 分配独立坐标；维度小于 catalog 大小时直接失败。paper 配置中的 one-hot 维度已改为 25。

### Held-out 与 paraphrase 评估

泛化 suite 将 19 个标签用于训练，查询分为 2 个 seen 校准点、4 个同义改写和 6 个完全 held-out 标签。验证器拒绝 unseen 标签出现在训练集合、paraphrase/unseen 文本与训练描述完全相同、重复 query key、未知标签或非法 split。

### 细粒度任务语义

旧环境主要用 attack/stealth 二元姿态决定威胁区奖励。当前新增显式的 25 意图奖励画像，对 distance、energy、collision、safety、task、time 和 threat 七个奖励分量设置温和倍率。该机制默认关闭以保护历史实验；新泛化、pilot 和未来 paper 配置显式开启。

奖励画像是任务定义，不是学习结果。论文必须给出倍率表、敏感性分析和 `profiles disabled` 消融，防止收益被误解为人工奖励调参。

### 真正配对的环境随机性

训练 episode、周期评估、collision probe、最终风险档和每个文本 query 均使用由研究 seed 推导的确定性 reset seed。所有变体面对相同初态和目标布置；seed 公式写入 manifest。

## v4 smoke 的诊断结论

`uav_intent_generalization_smoke_v4` 完成 4 个变体、1 seed、每 query 1 个评估 episode。v4 在 v3 的确定性配对基础上修复了 neutral 被当作 stealth 的边界问题：

- pretrained semantic 的 paraphrase top-1 retrieval 为 0.50，平均 margin 为 -0.034；
- legacy hash 的 paraphrase top-1 为 0.00；
- random dense 与 one-hot 为 1.00，因为它们获得 canonical-label identity oracle，不能表述为文本理解；
- 2 回合训练后四种方法的行为指标几乎相同，未形成任何效果证据。

该结果说明协议能揭示表示差异，但 16 维 smoke 投影损失了部分语义邻域；pilot 应比较 64、128、384 维或直接使用原始 embedding。所有行为数值只用于管线诊断。

## 最短可投稿路径

### Gate P1：泛化 pilot

运行 `uav_intent_generalization.pilot.json`，至少 5 seeds。通过条件：无 NaN/崩溃；semantic 的 paraphrase retrieval 明显优于 legacy hash；seen→paraphrase/unseen 行为差距可稳定估计；运行时间和方差足以确定 paper 预算；检查 64 维投影是否仍破坏语义邻域。

### Gate P2：强基线与消融

接入 IPPO、MATD3/MADDPG、规则任务分配 + ORCA/势场，并完成 w/o masking、w/o potential、w/o attention、profiles disabled、去除姿态对齐和投影维度敏感性。

### Gate P3：扩展有效性

至少增加两个公开多智能体连续控制任务，或选择 UAV 系统路线并完成风、噪声、通信扰动与 SITL。只在当前简化环境上给出结果，不足以支撑顶刊外部有效性。

### Gate P4：paper/frozen

冻结不少于 10 seeds、每 seed/风险档不少于 100 个确定性 episode；生成 paired CI、IQM、最坏分位、失败案例、自动图表、完整 manifest 与容器摘要。

## 当前建议

下一步不应直接运行 10-seed paper 配置，也不应把 v4–v7 继续循环调参后称为盲测。外部来源审计确认 AerialVLN 可作为人类 UAV 导航语言 OOD 负对照，但不能提供六维偏好真值；128 条 smoke 在未校准 0.20 profile 偏移阈值下仍激活 39.06%，当前拒答能力不合格。仍须独立招募并冻结带目标/极性的人类偏好数据，用 dev 校准阈值后在完整冻结 OOD 集报告误接收率。只有解码器校准/拒答、在线意图切换、多 seed safety 与 CBF 可行性均稳定后，才进入完整 paper 实验。

安全合同代码已从旧七轴 profile 迁移为六个可协商偏好 + 固定 collision constraint。此前架构 v3/消融 smoke 因使用旧合同已 superseded，尽管其历史 artifact 自身仍完整；clean commit 后必须重跑。

长实验现在支持原子 variant×seed 分块与 episode-boundary 精确 resume。VMAS 两场景 calibration 已关闭预算前置门槛；UAV 架构 v3 与消融的历史 60.9–105.2/82.2–105.0 “GPU-hours”仅为 smoke wall-time 占用代理，尚未同口径校准，不能当作 GPU compute。正式 UAV paper 启动前仍需 100-training-episode calibration，且必须在 clean commit 上执行。

CPU 资源审计显示：冻结双模型构造约 11.28 s，33-query 首次批量推理约 2.43 s；profile 缓存后每条约 0.93 µs，重复完整 MiniLM 编码约 72 ms/33 条。在线系统必须在任务下达时预编码并缓存，不能在控制环内冷启动 1.42 亿参数 NLI 模型。

8-UAV cyclic CBF 的 CUDA 标量同步已消除。固定 28 pairs、4 iterations 的旧实现对照基准中，filter+diagnostics 从 54.80 ms 降至 0.83 ms（66.24×），最大动作误差 `4.25e-7`。完整消融 smoke 的缓存变体平均 12.56 s，`no_cbf` 13.14 s，说明 CBF 不再主导该短实验成本。该结果只证明实现等价性与工程可运行性，不属于算法效果证据。

QP-CBF 已在 5-seed adversarial pilot 中把碰撞率从 cyclic-4/none 的 0.0184 降至 0.0144，且完成度差异接近零。优化后 4-UAV CPU 延迟约 1 ms，但 adversarial 审计成功率只有 98.52%；因此安全工程门槛从“缺少可行性审计”推进为“已量化不可行率，尚需高阶动力学与备用策略”。

Crazyflie 刚体 pilot 进一步显示：速度空间 QP 在 5 seeds 下实现 99.17% 目标成功和零
物理碰撞步，且相对无过滤显著改善间距、碰撞、成功和 RMSE；但仍有 23.83% 的步低于
规划安全距离。真实性门槛因此从“无高保真动力学”推进为“已有高保真风险降低证据，
尚无鲁棒轨迹保证与 SITL/HIL/实机闭环”。

80 ms 延迟预算的鲁棒 QP 在 ground/drag/downwash 组合物理下把安全违规步比例降至
0.0556%，并相对原始 QP 同时提高目标成功 5 个百分点；但违规没有归零，80 ms 预算也
尚未由独立系统辨识覆盖率验证。安全证据门槛推进为“已有多种子经验鲁棒性”，仍未达到
“形式化保证或真实系统验证”。
