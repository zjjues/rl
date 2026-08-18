# 研究变更日志

## 2026-08-18：修复主实验 provenance，加入精确统计与自动论文产物

### 工程变更

- 发现 `uav_imappo_main` 由分块 `--resume` 运行组装时，顶层 `config.json`、`manifest.json`、`RESULT_CARD.md` 和 `summary.json` 被最后一次 IPPO-only 运行覆盖；40 个逐 seed 结果仍存在，但证据链相互矛盾。
- `run_research_study.py` 现在只允许 resume 增加新变体，禁止改变 seeds、环境、训练、意图、评估和泛化协议，也禁止用同名 key 重定义变体；缓存结果必须逐项核验 seed、variant key 和完整变体定义。
- composite manifest 升级为 schema v2，保留逐次命令、内嵌配置、Git commit、dirty 状态和完成时间；不再覆盖历史调用。
- 新增 `validate_research_artifact.py` / `src/research_artifact.py`，核对预注册配置、manifest、40 个逐 seed 结果、summary raw 值和 SHA-256。
- 修复前审计保存在 `docs/paper/audits/uav_imappo_main_pre_repair.json`；修复后审计为 valid、40/40 results、44 checksum entries、0 errors，唯一警告为原始训练使用 dirty worktree。
- 配对统计新增精确 sign-flip test、paired standardized effect size 和 18 项主比较的 Holm FWER 校正；非有限值按 pair 同步剔除。
- 新增 `generate_paper_artifacts.py`，自动生成 CSV、bootstrap-CI 图、paired forest plot、统计报告和生成文件哈希清单。
- 将已删除 runner 中仍有价值的几何诊断迁移到 `src/intent_geometry.py`，并仅恢复泛化协议 v1/v2/v7/v8 的最小继承链。

### 论文影响

- 10-seed 架构 pilot 现在是内部一致、带 checksum 的可验证 pilot，但因历史 dirty worktree 和每层仅 50 个评估回合，仍不能升级为 frozen paper evidence。
- I-MAPPO 对 MAPPO 的 collision/task 比较经 Holm 校正后均不显著；hard 碰撞的未校正配对差为 +0.12576、95% CI [0.03364, 0.21996]，但 Holm p=0.296875。
- I-MAPPO 相对 IPPO/MATD3 的显著结果表现为碰撞更少但任务完成更低的权衡，不构成全面优势。
- 所有架构 pilot 变体使用 one-hot intent 并关闭 intent reward，因此这些结果不能证明自然语言或语义意图的价值。

### 后续验证

- 运行 5-seed 语义完整方法因果消融，先验证 NLI gate、attention、mask、intent reward 和 CBF 的独立贡献；
- 消融通过后在 clean commit 上冻结不少于 10 seeds、每 seed/tier 不少于 100 回合的 paper 配置；
- 将可导出的 I-MAPPO 策略接入 Betaflight 双机冲突场景，而不是只做定高 smoke；
- 独立人类/外部语言数据和 HIL/实机仍是不可由合成实验替代的硬门槛。

## 2026-08-02：Betaflight SITL 单机闭环打通 + 多机原型

### 工程变更

- **SITL 源码 5 处修复**（WSL `/home/zhaji/rl-sitl/betaflight`）：
  - `src/main/rx/rx.c`：`frameStatusUdp` 添加 SIMULATOR 路径，RC 帧间隔不触发 RXLOSS
  - `src/main/flight/imu.c`：`isUpright()` SIMULATOR 分支始终返回 true，绕过倾角检查
  - `src/platform/SIMULATOR/sitl.c`：包含 `fc/runtime_config.h`，每帧 FDM 持续 `ENABLE_ARMING_FLAG(ARMED)`
  - `src/platform/SIMULATOR/sitl.c`：添加 `--port-offset <n>` 参数支持多实例端口隔离
  - 构建：`make TARGET=SITL -j$(nproc) OPTIONS=SITL_ATTITUDE_DIRECT`
- **Bridge 多机支持**：`ports_for_drone()` 端口偏移 (base + 10×id)
- **Runner 改进**：`script -q -c` 无缓冲、UTF-8 编码、启动延迟 3s
- **新增文件**：`run_betaflight_sitl_multi_smoke.py`、`run_betaflight_sitl_multiseed.py`
- **测试**：85 passed（主环境），18 个实验配置冻结

### 论文影响

- Betaflight SITL × PyBullet 闭环 pipeline 从不可用推进到端到端通过验收
- 单机 V19：电机接收率 99%，满油门输出，高度响应 +0.15m
- 多机 Multi V1：双无人机各自独立 SITL 实例，端口偏移隔离
- 当前状态为 smoke test 级别，未达到冻结论文实验标准（需 10+ seeds）

### 后续验证

- 多 seed 统计验证（当前运行中）
- 多机冲突场景 + 安全层
- 鲁棒压力测试（延迟、噪声、风扰）
- HIL 复核

## 2026-08-01：加入跟踪延迟预算鲁棒 QP

### 工程变更

- 从刚体 pilot 暴露的命令/轨迹失配出发，新增 `robust_qp`；约束距离增加
  `2 * max_speed * tracking_latency_budget`，默认预算 80 ms、margin 0.04 m。
- PyBullet 评测支持官方 `PYB_GND_DRAG_DW` 组合物理，并分别记录原始安全距离、约束
  距离和鲁棒 margin。
- 新增 1 个纯函数测试；先运行 3-seed smoke，再冻结 5-seed pilot，未从 pilot 反调参数。

### 论文影响

- 鲁棒 QP 相对原始 QP 在安全违规、间距、成功、RMSE 和能量上均得到不跨零的配对
  改善，成为当前最强安全层候选。
- 物理安全违规仍非零，必须称为延迟预算下的经验鲁棒化，不能称为形式化鲁棒 CBF。

### 后续验证

- 用系统辨识或独立校准轨迹验证 80 ms 预算覆盖率；加入状态噪声和延迟超预算压力测试；
  在 SITL/HIL 中复核安全违规尾部。

## 2026-08-01：接入隔离 PyBullet 刚体动力学与高度通道规划

### 工程变更

- 新建 `rl-pybullet`，固定官方模拟器提交；主 `rl-test` 在不兼容安装后精确回滚，
  81 项测试与 `pip check` 全部恢复通过。
- Windows conda-forge MKL NumPy 在 `linalg.inv` 卡死，隔离环境改用 PyPI/OpenBLAS
  NumPy；Setuptools 固定 80.9.0 兼容上游 `pkg_resources`。
- 新增纯 NumPy/SciPy 控制与安全层、两类交通场景、检查和评测运行器。
- v1 暴露死锁后加入通用高度通道；v2 标记为开发诊断，随后冻结 5-seed pilot。

### 论文影响

- 补入刚体/旋翼动力学证据，但对象是结构化目标控制器，不能替代独立语言数据，也
  不能称为 SITL/HIL。
- QP 相对无过滤改善碰撞、间距、成功和 RMSE；命令约束完全可行仍不能保证物理轨迹
  满足安全距离，限定了一阶安全层的适用边界。

### 后续验证

- 冻结扩展场景与 10-seed paper 协议；加入鲁棒/预测安全层；接入 SITL/HIL 或实机。

## 2026-08-01：NLI 目标解码、离散 CBF 审计与动态意图协议

### 工程变更

- 新增固定 revision 的 DeBERTa-v3 NLI 极性解码器、相似度门控及训练原型最近质心门控；依赖锁定 `sentencepiece==0.2.2`。
- UAV v2 将 safety 的主动避让半径与 collision 的近程屏障响应分离。
- 每个环境步新增 CBF 最大/平均约束违例、违例比例和预测最小间距；明确仅审计一步线性约束。
- 新增 episode 内低→高文本意图切换，使用同观测的旧意图反事实动作测量响应延迟，避免轨迹混杂。
- v4 经观察后转为开发集；目标特定 NLI 假设修复后先冻结 v5 再运行。v5 失败后亦冻结，不允许继续调参复用。下一阶段使用全新的 held-out 套件。

### 论文影响

- 5-seed 结果支持“门控显著降低非目标串扰”，但不支持 safety 稳定可控或 CBF 独立降低碰撞。
- 动态 oracle 证明主动 safety 控制器可响应；语言主方法的失败可归因到目标类别/极性解码，而不是环境完全不可辨。
- 当前 CBF 在动作盒约束下仍有小幅残余违例，必须作为软安全层报告，除非后续加入可行性求解和形式化证明。
- 残差学习没有可测增益，论文不得以 MARL 学习改进为主结论。

### 后续验证

- 冻结原型类别门控并在从未观察的新措辞上验证；
- 将动态意图与 CBF 诊断扩至 5 seeds、扰动和规模变化；
- 增加语言冷启动/缓存延迟、模型大小及实时预算；
- 若语义解码稳定，再运行 paper 级 10 seeds；否则停止扩大算力并更换监督方案。

## 2026-08-01：原型门控动态 pilot 与资源审计

### 工程变更

- 新增七目标+中性目标类别质心门控、15 类 low/high/neutral 极性原型对照，以及 v6/v7 冻结后修复套件。
- 新增 `benchmark_semantic_runtime.py`，分开测量双模型冷构造、首次 query 批处理、缓存 profile 和重复完整编码，并报告参数量。
- 运行 `uav_nli_prototype_dynamic_pilot_v1`：5 seeds、3 variants、完整 33 query、四类 episode 内切换和 CBF 逐步审计。

### 论文影响

- 15 类纯原型极性失败并停止；NLI 原型门控保留为当前最佳但尚未通过的方法候选。
- 静态排序不能替代在线响应：energy 静态相关为正但动态响应率为 0，暴露出 profile 幅度压缩。
- safety/threat 在线首步响应稳定，但 safety 轨迹级最小间距相关 CI 跨零；论文不能只报告动作响应。
- 双模型 CPU 冷启动约 11.3 s，必须采用任务级预编码/缓存；该架构目前不适合在高频控制环内解析新文本。

### 后续验证

- 停止使用开发者手写 v4–v7 继续调参；建立来源独立的目标/极性标注语料和最终 held-out 集；
- 将 collision 从可降低语言偏好改为不可违反的安全约束，避免概念与规范冲突；
- 求解动作盒下的 CBF 可行投影或报告不可行率；
- 仅在语言解码与在线切换通过新 pilot 后进入扰动、规模、SITL 和 10-seed paper。

## 2026-08-01：不可放宽安全合同与 QP-CBF

### 工程变更

- collision 从语言偏好空间移入不可放宽安全合同；低 collision/safety 文本不再降低基础 CBF 距离。
- 新增盒约束 `pairwise_qp`，显式最小化对 nominal action 的二范数改动并满足线性成对约束。
- 新增求解报告成功、审计成功、迭代、fallback、毫秒延迟和不可行状态保留。
- suite 继承支持 `exclude_query_keys`；v8 正式候选只保留六个可协商目标并排除 collision 三条反事实。
- 新增独立标注数据验证器和冻结 MiniLM multinomial classifier 训练入口，按标注者隔离 split。

### 论文影响

- 5-seed adversarial pilot 支持 QP 相对 4 轮/无过滤降低碰撞且不损失完成度的方向性证据；相对 32 轮差距很小。
- 优化后 QP 在 4 UAV CPU 上约 0.75–1.01 ms/步，比 Python/Torch 循环投影更快；该延迟结论仍需多 seed/多规模复现。
- adversarial 仍约 1.48% 状态不可满足当前一步约束，论文必须报告可行率，不能写成零违例保证。
- collision 不再作为“可控语言偏好”，避免把不安全请求响应能力包装成贡献。

## 2026-08-01：连续目标画像、反事实协议与共同随机数修复

### 工程变更

- 规则残差先验从 attack/neutral/stealth 三姿态扩展为 distance、energy、collision、safety、task、time、threat 七维连续画像；历史 `intent_retrieval` 路径保留为消融。
- 新增 `dual_ridge`、单调 `concept_anchor`、极性归一化 `contrastive_anchor` 与概念原型增强 `prototype_ridge` 四种文本到画像解码器，并记录解码器、锚点和校准元数据。
- generalization v2 新增 7 组、每组 3 条仅改变一个目标的 minimally contrastive queries；结果同时保存逐 query 目标画像、预测画像、MAE、profile correlation 与逐目标 Spearman。
- 修复旧 controllability 评估中 query index 改变环境 reset seed 的混杂：同一训练 seed、风险层和评估 episode 的所有反事实 query 现在共享完全相同的初态和目标轨迹。
- 新增 `objective_profile_oracle`，只在评估时读取真实画像，用于验证环境—控制—指标链是否可辨识；不得作为可部署方法。
- 加入 threat controllability、oracle profile 传递与 4 项相应回归测试。

### 论文影响

- `uav_counterfactual_grounding_smoke_v1` 因 query 初态不匹配，只能保留为失败审计，不能支持因果意图遵循结论。
- matched-seed v2 证明旧异常主要受场景混杂影响；v3/v4 进一步把画像解析和学习残差隔离。
- `prototype_ridge` 在 v4 的 33-query 单种子 smoke 上将总体/反事实画像 MAE 降到 0.134/0.102，七个反事实目标的预测排序均正确；但动作层只对 task/time 完全单调，distance/energy/safety 部分单调，threat 指标退化。因此当前瓶颈已从单纯语义表示转移到环境可观测性和控制接口。

### 后续验证

- 在新 study ID 下运行多 seed 反事实 pilot；
- 修复安全距离物理尺度、威胁区不可观测和任务进展与任意动作幅值耦合的问题；
- 在冻结的新环境版本上重新预注册画像解码器与残差学习实验。

## 2026-08-01：UAV benchmark v2 与五种子原型语义 pilot

### 工程变更

- 新注册 `uav-scheduling-v2`，保留 `v0` 逐步兼容；v2 观测增加最近威胁区相对向量，局部维数由 30 变为 33。
- v2 的任务进展由“任意二维动作幅值”改为目标接近度与动作—目标方向一致性共同驱动。
- 新增连续 `distance_to_threat`、`collision_distance_spearman` 和 `policy_residual_magnitude`；二值 threat/collision 事件继续并列报告。
- `uav_prototype_grounded_residual_pilot_v1` 完成 5 seeds × 5 variants × 33 queries × 3 risk tiers；随后完成 200-episode 残差尺度/势奖励敏感性 smoke。

### 论文影响

- 原型语义方法在 hard/critical 的七个目标行为相关均为正；相对 direct ridge，energy 与 threat 相关分别提高 1.0 和 1.1，critical safety 提高 1.2，但 distance/task 各低 0.5。
- 学习残差与零残差语义先验在五种子 pilot 的七项相关完全相同。200 episodes 后残差幅度达到 0.013–0.053，但没有带来绝对性能或可控性提升；当前不得把贡献表述为“MARL 学习改进”。
- `prototype_ridge` 当前更准确的定位是可解释文本目标解析 + 模型先验控制；残差学习只有在扰动补偿实验通过后才可能成为主方法组成。

## 2026-08-01：扩展任务遵循指标并向量化统计后处理

### 工程变更

- UAV step info 与统一 evaluator 新增 energy remaining、action magnitude、speed、distance to target、minimum neighbor distance 和 threat-zone violation；VMAS 不伪造这些 UAV 指标。
- generalization summary 和 paired comparison 动态汇总存在的资源指标；controllability 新增 energy、distance、time 和 safety-separation preference Spearman。
- `uav_intent_resource_metrics_smoke_v1` 完成 12 queries × 2 tiers × 3 variants 的 raw/split/paired/controllability 全链路。
- bootstrap 对 singleton 直接返回精确区间，对 mean/median/IQM 采用分块向量化重采样；同一 smoke 单变体汇总由约 200 秒降至 0.34 秒。
- VMAS navigation 与 dispersion 均完成 5-seed/1000-transition pilot；全仓 53 项测试通过。

### 论文影响

- 任务遵循不再只由 collision/task/return 三个粗指标替代，可以直接检验 energy-saving、rapid-response、separation 和 threat-avoidance 等文本目标。
- 1-seed resource smoke 中部分相关为负，进一步说明当前方法尚未实现全面多目标遵循；正式结论必须依赖新 5-seed pilot。
- 两个 VMAS pilot 证明 MAPPO/IPPO/MATD3/I-MAPPO 基础设施可跨公开场景运行，但 VMAS 原任务仍不提供语言语义证据。

### 后续验证

- 以扩展指标重新运行 5-seed semantic residual pilot，并预注册安全、任务、能量、时间四类 co-primary alignment；
- 为多重 co-primary 假设加入 FDR 或层级检验策略；
- 生成 Pareto/frontier 与 seed-level paired forest plots。

## 2026-08-01：将 VMAS 公开连续控制环境纳入正式运行器

### 工程变更

- 在 `rl-test` 安装并锁定官方 `vmas[gymnasium]==1.5.2`，同时解析 gym 0.26.2、shimmy 2.0.1、pyglet 1.5.27；`pip check` 无损坏依赖。
- 修复旧 VMAS 绘图入口的用户级 Matplotlib 缓存依赖，将缓存定向到系统临时目录。
- `run_research_study.py` 支持 `environment.name=vmas:<scenario>`：动态探测 obs/state/action dimensions，使用同一非覆盖目录、manifest、paired statistics、result card 和 checksums。
- 显式拒绝在 VMAS 上注册 UAV 特定 rule planner 或 rule-residual，避免用不兼容观测布局冒用方法名。
- `vmas_navigation_formal_smoke_v1` 完成 I-MAPPO(one-hot)、MAPPO、IPPO、MATD3 四个真实连续动作变体；新增 VMAS 1.5.2 API 与拒绝规则测试。全仓 50 项测试通过。

### 论文影响

- 工程已有第二个公开 MARL benchmark 的正式复现路径，初步关闭单一自建环境的工程缺口。
- 原生 VMAS navigation 不消费语言意图，因此该 smoke 只验证优化器/critic 的跨环境实现，不能作为语义泛化证据。论文必须明确区分“算法基础设施跨环境”和“语言任务跨环境”。

### 后续验证

- 在 navigation 与另一个 VMAS cooperative scenario 上运行 5-seed pilot，至少比较 MAPPO/IPPO/MATD3；
- 若要声称语义方法跨环境，需要预先定义不泄漏原任务的语言条件目标或选择已有语言任务 benchmark；
- paper 运行锁定 VMAS scenario kwargs、版本和环境 transition 预算。

## 2026-08-01：加入 UAV 扰动与零样本规模外推协议

### 工程变更

- `uav-scheduling-v0` 新增默认关闭的 wind standard deviation、observation noise、action delay 和 neighbor communication dropout；所有扰动由环境 reset seed 驱动。
- 显式零扰动与旧版环境在相同 seed/action 下逐步等价；组合扰动同 seed 可复现。动作时延使用零动作队列，通信丢包只屏蔽邻机相对位置/速度。
- 研究风险层可逐层覆盖扰动参数与 `n_agents/n_targets`，完整值写入 config/manifest。
- 分散 actor 执行时按实际 observation batch 推断 UAV 数；critic 仍只在训练规模使用。4 UAV 训练策略可在固定 3-neighbor、30 维局部观测下直接执行 8/12/16 UAV。
- `uav_robustness_smoke_v1` 完成 nominal/wind/sensor/latency-dropout/combined 五档；`uav_scale_transfer_smoke_v1` 完成 4/8/12/16 UAV。全仓 48 项测试通过。

### 论文影响

- 外部有效性从“无扰动单一规模”推进到可预注册的动力学、感知、控制时延和通信退化矩阵。
- scale smoke 只证明 shared local actor 的形状可迁移；1 seed、2 eval episodes 的数值不得表述为规模泛化效果。
- 正式鲁棒性结论需要 5-seed pilot 选择扰动强度，再以 10+ seeds、固定强度和最坏分位/CVaR 冻结。

### 后续验证

- 以 nominal→single disturbance→combined 的固定矩阵运行多 seed pilot；
- 增加最小机间距、威胁违规率、能耗和路径长度，避免只依赖 collision/task/return；
- 接入第二个公开连续多智能体环境，并评估环境特定接口是否改变结论。

## 2026-08-01：接入真实连续动作 MATD3 强基线

### 工程变更

- 新增 `matd3_baseline.py`，实现共享分散 actor、集中式 twin-Q critics、replay buffer、target policy smoothing、delayed actor update 与 Polyak soft update。
- MATD3 策略不接收意图向量；训练任务标签只设置环境奖励画像，与 MAPPO/IPPO 一样标记为 `task_labels_hidden_from_policy=true`。
- 训练 episode 使用相同 `seed * 1_000_000 + episode` 重置协议，并复用统一的最终风险层和 held-out query 评估器。
- 日志显式保存 replay size、critic loss、最近 actor loss、当前步是否更新 actor 和累计 actor update 次数，避免 delayed-update 相位造成“actor 未训练”的假象。
- `uav_matd3_smoke_v1` 验证完整训练与 held-out 评估；`uav_matd3_logging_smoke_v2` 验证 100 条 replay、34 次 actor 更新、有限 actor/critic loss。全仓 43 项测试通过。

### 论文影响

- 强基线从 MAPPO/IPPO/规则控制扩展到真实 off-policy 连续动作方法，关闭了“只与同类 PPO 比较”的明显审稿风险。
- smoke 数值仅证明实现路径，不用于比较算法优劣。MATD3 必须在相同 5-seed pilot 和后续 10-seed paper 预算下与主方法配对。

### 后续验证

- 运行同协议 5-seed MATD3 feasibility 并做校验和审计后的跨研究 paired comparison；
- 记录样本数而非只记录 episode 数，公平讨论 on-policy/off-policy 样本效率；
- paper 前与独立公开 MATD3/MADDPG 实现交叉核验关键结果。

## 2026-08-01：负结果驱动的语义残差控制重构

### 触发证据

- `uav_intent_generalization_feasibility_v2` 完成 objective-grounded semantic、raw pretrained、IPPO 和 MAPPO 的 5-seed/50-episode 先导实验；结果只在部分 easy/paraphrase 指标出现方向性信号，未形成稳定 unseen 优势。
- `uav_rule_planner_feasibility_v1` 在相同 seeds、风险层、查询套件和 reset 公式下完成。经新增的跨研究校验工具验证源 SHA256 和协议兼容性后，规则控制器在 hard 主场景相对 objective-grounded direct I-MAPPO 的碰撞率低 0.140（95% bootstrap CI [0.076, 0.204]），任务完成度高 0.074（差值按 learning-rule 为 -0.074，CI [-0.085, -0.062]）。
- 上述结果表明直接扩大纯学习策略预算没有充分依据；5 seeds/5 eval episodes 仍是探索性证据，不能作为最终论文结论。

### 工程变更

- 新增 `compare_research_studies.py`：校验两个源实验的 checksum、seeds、环境、风险层、评估次数和泛化 suite，再生成独立、非覆盖的 paired comparison 与结果卡。
- 将规则控制核心抽取为共享 `compute_rule_actions`，确保 rule-only 和混合策略使用同一份目标跟踪、邻机势场避碰实现。
- 新增 `policy_mode=residual_rule`。actor 的高斯均值以规则动作的 inverse-tanh 为中心，只学习受 `residual_action_scale` 限制的语义条件残差；残差均值头零初始化，初始确定性动作与规则控制器一致。
- rollout 显式保存每步 base action，PPO 更新时复用同一基准动作，保证行为策略与新策略 log-prob 的概率比口径一致。
- 新增语义残差/无语义残差/direct/rule-only 因子 smoke 和结构测试；新增 `pytest.ini`，避免历史绘图脚本被误收集并依赖用户级 Matplotlib 缓存。
- 当前 36 项回归测试全部通过，`pip check` 无损坏依赖。

### 论文影响

- 主方法候选从“纯语义条件 PPO”调整为“经典安全导航先验 + 语义条件残差 MARL”。该设计允许分别检验导航先验、学习残差与语义条件三项因果作用。
- `uav_semantic_residual_smoke_v1` 中 residual 与 rule-only 在 5 回合后几乎相同，而 direct policy 在 hard 场景碰撞率为 0.500、residual 为 0.025。该结果仅证明初始化和管线行为符合设计，尚不证明语义贡献。
- 正式贡献必须满足 semantic residual 优于 nonsemantic residual 和 rule-only，而不只是优于从零学习策略；否则论文只能声称混合控制稳定性，不能声称语义泛化收益。

### 后续验证

- 运行 5-seed/50-episode 残差 pilot，检验 residual 是否保持 rule prior，并估计语义相对无语义残差的效应大小；
- 若语义效应接近零，重构训练任务使不同意图要求可辨别的控制折中，并加入在线 intent switch；
- 接入 MATD3/MADDPG、扰动与规模外推后，才进入 10-seed paper 冻结阶段。

### v1→v2 机制修订与可控性审计

- v1 pilot 发现 inverse-tanh 规则中心在饱和动作处残差梯度过小，且 residual 通过外部 posture 获得了语义 oracle。
- v2 改用 signed-headroom 潜变量映射；零残差仍严格等于规则动作，饱和动作保留向内梯度，PPO 在保存的潜变量上计算概率比。
- 规则上下文被拆成 neutral、oracle_posture、intent_retrieval；full 只从文本向量检索，nonsemantic 使用 neutral，oracle 仅用于传统规划上界。
- 新增只读 `analyze_intent_controllability.py`，在验证源校验和后按 seed 计算意图偏好—行为折中相关。该指标因在 pilot 后定义，被自动标记为 exploratory post-hoc。
- v2 显示安全—效率 Pareto 交换；objective 表示的安全折中相关优于 raw/nonsemantic 的部分场景，但 task preference 尚未对齐，不能声称全面语言遵循。

## 2026-08-01：建立无泄漏意图泛化与细粒度任务协议

### 工程变更

- 新增 `intent_generalization.py` 与 `uav_intent_generalization_suite.v1.json`，冻结 19 个训练意图、2 个 seen、4 个 paraphrase 和 6 个 unseen 查询。
- suite 验证器拒绝未知/重复标签、unseen 标签泄漏、paraphrase 精确文本泄漏和非法 split。
- `IntentLibrary` 新增 true one-hot catalog identity 与按标签保序子集；维度小于 catalog 大小时显式失败。
- I-MAPPO 可在不加入训练库的情况下编码评估 query：semantic/hash 消费新文本，random/one-hot 使用 canonical identity oracle。
- 研究运行器新增表示检索诊断和 split 行为汇总；query 先在 seed 内平均，避免伪重复样本。
- 新增 `intent_objectives.py`，为 25 个 UAV 意图注册七维奖励画像；旧实验默认关闭，新泛化/pilot/paper 配置显式开启。
- 训练和所有评估路径采用确定性 reset seed 调度，确保不同变体共享相同初态与目标布置；公式写入 manifest。
- 新增 `uav_intent_generalization.pilot.json`，并将正式 one-hot 维度修正为 25。
- neutral 姿态使用独立中间值，不再被错误解释为 stealth；威胁区对 neutral 不加 attack bonus 或 stealth penalty。
- 单元测试从 12 项扩展到 23 项，全部通过。

### 论文影响

- 主假设现在可以在标签和文本均隔离的 held-out 条件下被证伪，而不是只比较固定意图库内性能。
- one-hot/random-dense 的 paraphrase 结果被明确限定为 canonical-label oracle，不得写成文本泛化。
- 环境任务语义从 attack/stealth 二分类扩展为可审计的多目标奖励画像，但必须增加 profiles-disabled 与倍率敏感性消融。
- paired bootstrap 的配对单位现在同时包含训练 seed 与确定性评估场景，减少环境随机性混杂。

### v4 smoke 诊断（不可作为论文结论）

- `pretrained_semantic` paraphrase top-1 retrieval：0.50，mean margin：-0.034；
- `legacy_hash` paraphrase top-1 retrieval：0.00；
- random-dense/one-hot oracle：1.00；
- 1 seed、2 个训练回合后四个变体的行为差异极小，尚无方法效果证据。

### 后续验证

- 比较 64/128/384 维语义表示，确认随机投影是否破坏邻域；
- 运行 5-seed 泛化 pilot，估计 seen→paraphrase/unseen gap；
- 接入 IPPO、MATD3/MADDPG 与规则规划基线；
- 增加 profiles-disabled、倍率扰动与姿态对齐消融。

## 2026-08-01：建立并验收本地 `rl-test` 实验环境

### 工程变更

- 检测到同名 Conda 环境已存在，因此保留其内容并将 Python 从 3.8.20 原位升级到 3.10.20，避免删除用户已有环境。
- 重装 Python 3.10 对应的 NumPy、SciPy、Matplotlib、Gymnasium 和历史二进制轮子，消除解释器升级后的 ABI 不兼容。
- 安装并对齐 `torch==2.4.1+cpu`、`torchvision==0.19.1+cpu`、`sentence-transformers==3.2.1` 与 `transformers==4.57.6`。
- 新增 `environment-rl-test.yml` 和 `docs/paper/EXPERIMENT_ENVIRONMENT.md`，记录重建方式、关键版本、设备和模型 revision。
- `pip check` 无损坏依赖；12/12 单元测试通过。
- 完成 legacy-hash/one-hot 对照 smoke（2 runs）和 pretrained-semantic smoke（1 run），两项 manifest 状态均为 `complete`。

### 论文影响

- 真实预训练语义编码路径已经在固定模型 revision 下贯通模型下载、384 维编码、固定随机投影、训练和评估。
- 每次 smoke 运行均保存 Python、平台与核心依赖版本，为后续 pilot/paper 结果的环境审计提供模板。
- 当前结果只有 1 个 seed、1 个评估 episode，数值方差与置信区间不可识别，只能作为可执行性证据，不能用于论文效果声明。

### Smoke 观测（不可作为论文结论）

| Variant | Seed | Collision rate | Task completion | Episode return |
|---|---:|---:|---:|---:|
| pretrained semantic | 7 | 0.000 | 0.426 | -4.665 |
| legacy hash | 7 | 0.000 | 0.602 | -2.093 |
| one-hot | 7 | 0.625 | 0.639 | -5.323 |

### 后续验证

- 运行多 seed pilot，先检查学习曲线、运行时与失败率；
- 增加 seen/unseen/paraphrase 意图划分，直接验证语义泛化假设；
- pilot 通过后再冻结 paper 级配置并运行至少 10 seeds、每 seed/风险档至少 100 个评估 episode；
- 仅在计划跨环境验证时安装并锁定 PettingZoo/VMAS。

## 2026-08-01：建立正式研究运行与统计冻结机制

### 工程变更

- 新增 `run_research_study.py`，由预注册 JSON 配置驱动 smoke/pilot/paper 研究。
- `paper` 级运行强制至少 10 seeds、每 seed/风险档至少 100 个评估 episode，并默认要求干净 Git 工作区。
- 运行器拒绝覆盖已有研究目录，支持显式 resume，并保存 manifest、完整配置、逐 seed 结果、统计汇总、结果卡和 SHA256 校验和。
- 新增 `research_statistics.py`，提供 mean/median/IQM、bootstrap 95% CI、paired difference、win rate 和 performance profile。
- 正式汇总以相同 seed 进行 pretrained semantic 对各基线的配对比较。
- 新增 `configs/research/uav_intent_representation.paper.json` 和独立 smoke 配置。

### 论文影响

- 主结果不再仅报告点估计；论文表格可以直接使用冻结汇总中的 IQM、置信区间和配对效果。
- 调参输出与最终无偏测试在目录和运行入口上分离。
- 任何试图在连续动作 UAV 研究中注册 QMIX/VDN 名称的配置会被拒绝。

### 后续验证

- 在可用 Python 研究环境中执行 smoke 配置；
- 冻结依赖版本与容器摘要；
- 为跨环境与未见意图实验增加单独的 paper 配置。

## 2026-08-01：重构意图表示与语义口径

### 工程变更

- `IntentLibrary` 新增 `create_pretrained`、`create_random_dense` 和 `create_legacy_hash`。
- 预训练路径使用冻结的 `sentence-transformers` 模型和固定随机投影；缺依赖时显式失败，不允许自动退回随机表示。
- 所有意图库保存 `representation_type`、`semantic_geometry`、模型、投影种子和姿态元数据。
- 预训练模型锁定到显式 Hugging Face revision；paper 配置缺少 revision 时拒绝运行。
- `semantic_library` 保留为 legacy hash 的兼容别名并发出警告。
- I-MAPPO 配置新增编码器模型、投影种子与随机码种子。
- 新实验矩阵删除伪 `QMIX/VDN` 替代项，加入 pretrained semantic、random dense、legacy hash 和 one-hot 四组表示对照。
- pilot 调参结果改写到 `experiments/pilot/`，不再覆盖历史 Stage7 正式汇总文件。

### 论文影响

- 历史 hash 实验统一称为 `legacy hash intent code`，不再称为语义 embedding。
- 论文主比较升级为“真实预训练语义几何 vs 随机稠密/legacy hash/one-hot”。
- 表示元数据将随每个 seed 结果保存，避免实验名称与实际编码器不一致。

### 后续验证

- 检查语义模型缓存和依赖；
- 验证同义句相似度显著高于无关意图；
- 加入 seen/unseen/paraphrase 划分；
- 重新运行表示对照实验。

## 2026-08-01：修复意图—战术姿态一致性

### 工程变更

- 为 UAV 意图增加 attack/stealth/neutral 元数据。
- 训练时按环境战术姿态过滤意图候选；neutral 意图可用于两类姿态。
- semantic/random/hash 表示与 one-hot 共用姿态驱动的动作掩码逻辑。

### 论文影响

- 排除了“文本意图与环境奖励目标相冲突”的主要混杂因素。
- `w/o masking` 消融现在对表示库模式产生真实可区分的干预。

### 后续验证

- 单元测试所有预置意图的姿态分类；
- 记录每个 episode 的意图标签与姿态；
- 增加去除意图—姿态对齐的专门消融。

## 2026-08-01：建立论文导向的工程治理体系

### 工程变更

- 新增论文工作区、研究计划、方法草稿、实验协议、贡献台账和结果台账。
- 将 smoke、pilot、paper、frozen 定义为不同证据等级。
- 明确禁止以替代模型冒用 QMIX/VDN 名称。
- 明确 legacy hash 只能作为表示对照，不能作为 semantic 主方法。

### 论文影响

- 核心论点从“hash 语义库降低碰撞”调整为可证伪的研究问题：具有语义几何的文本表示能否带来未见意图泛化与安全收益。
- 每项候选贡献必须绑定消融、强基线、统计和泛化证据。
- 历史 Stage7 结果降级为 pilot，不再作为最终论文主结果。

### 后续验证

- 接入预训练文本编码器；
- 修复意图—姿态一致性；
- 完成正式统计与结果冻结工具；
- 在新协议下重新运行实验。
