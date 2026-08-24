# 研究变更日志

## 2026-08-24：Episode 边界精确恢复、RNG 隔离与 paper 配置修复

### 工程变更

- IMAPPO/MAPPO/IPPO/HAPPO 训练入口支持从下一 episode、累计日志和未满 rollout buffer 恢复；IMAPPO/HAPPO 私有 intent/update RNG 写入算法 checkpoint。
- MATD3 新增在线/目标网络、优化器、完整 replay、更新游标和私有 RNG 的 checkpoint；训练入口恢复累计 environment steps，保持 warmup 与 delayed policy update 相位。
- 研究运行器原子保存所有 Python/NumPy/Torch CPU/CUDA RNG，并用完整注册 spec 与研究源码 SHA-256 绑定 checkpoint；最终 result 原子落盘后才清除 checkpoint。
- checkpoint cadence 成为正整数注册字段；默认 1。MATD3 10,000-transition checkpoint 实测增长至 18.22 MiB，据此把 3000/2000-episode paper 配置注册为 50，最终 episode 强制保存。
- 修复 `uav_imappo_main.paper.json` 的遗留伪 `concat` critic、错误 pilot 等级和不足的 50-episode eval，使用新 study id 避免覆盖历史 pilot。五份 paper 配置均 dry-run 通过。
- 分块 paper result/manifest 新增 canonical protocol 与 implementation SHA-256；resume 拒绝旧无指纹 fragment 或跨实现运行。
- artifact validator 新增严格 `--allow-partial`：只接受 missing-pair 集合精确、已有结果身份/校验和有效且无 summary 的 `valid_partial`。
- 首个完成但缺 result 指纹的正式消融单元原样保留并标记 superseded；新 validator 对其反证审计为预期 invalid。
- 新增中断恢复等价性测试；本批最终全量回归 **182 passed, 14 warnings**。

### 论文影响

- 可以在 Methods/Artifact Appendix 中声称确定性 CPU 测试路径的 episode-boundary bitwise resume，并明确 CUDA 跨平台非确定性边界。
- 不能把恢复协议当作效果证据；当前 paper 结果仍为空。navigation 与 dispersion 均完成 5/5 单 seed calibration。

### Calibration 与预算证据

- navigation 五算法 calibration 完成；MAPPO 同配置复跑 return 完全相同，验证当前 VMAS seed 路径的结果可重复性。
- 旧 attention 16,556 s 墙钟被 active-time 复跑证伪；稳定 process CPU 为 141–209 s/variant，MAPPO 首轮 458 s 瞬时异常也由复跑识别并保留。
- runtime planner schema v2 区分 wall/process-CPU time，并逐算法计入训练、周期监控、UAV collision probe 与最终评估。最终评估保持 100 episodes，训练监控独立注册为 20；navigation/dispersion 新预算分别为 33.42–44.36/34.09–45.47 active CPU-hours。
- 新增 `monitor_eval_episodes`，将训练期监控与最终统计样本量解耦；paper 最终 100 episodes/tier 不变，monitor 固定 20。
- UAV v3 六算法 calibration 完成；60-run 预算校正为 70.76–96.04 active CPU-hours。MATD3 的 3711 s wall/121 s CPU 分离作为第二个宿主时间污染证据保留。
- 新增 10-seed/100-run UAV 链式消融 paper 注册及同构 calibration；运行时规划器现在拒绝同名变体的任意执行字段漂移。
- 消融 calibration 完成 10/10，artifact valid、0 warning；正式预算为 70.65–78.87 active CPU-hours。单 seed 近零差异只记录为负结果风险，不进入效果台账。

## 2026-08-20：CityNav 一次性 OOD 终测失败、VMAS 架构复现与人工数据冻结入口

### 工程变更

- 新增 `final_ood_registration.py`、CityNav v1/v2 预注册、schema-only importer、Wilson FAR 汇总和 one-shot evaluator/attempt marker。v1 在仅查看 archive paths 后发现 difficulty subset 重复，文本未打开即由 v2 透明取代。
- 固定 CityNav commit `372ecbd...7710`、archive SHA-256 `121d052e...65bd` 及四个 canonical split 文件哈希；32,637 条全部进入唯一一次冻结评估。
- VMASAdapter 增加 horizon enforcement 和不虚构 UAV objective 的 info 映射；research protocol 强制 VMAS 为 architecture-only。
- 新增 navigation/dispersion 各 5 算法 smoke、10-seed paper 配置、artifact audit 和 runtime plan。
- 新增 `audit_formal_preference_dataset` 与 `freeze_preference_dataset.py`，机器执行 13 类覆盖、独立复核、writer-disjoint split、每类/每 split 最小量、consent 版本和 JSONL 哈希冻结。

### 论文影响

- CityNav FAR 为 96.15%（Wilson 95% CI 95.94%–96.35%），预注册 outcome=fail。该结果必须保留并禁止用 CityNav 反调 gate。
- 当前 gate 不能进入论文主方法；语言门槛退回“需独立人工偏好数据和多来源负例”。
- VMAS 只能支撑架构复现，不能支撑语言泛化、UAV 安全迁移或偏好准确率。

## 2026-08-20：外部语言数据合同与可恢复实验预算

### 工程变更

- 新增 `external_language_corpus.py` 和 AerialVLN importer。外部 episode 只保留原始来源、版本、split、record id 和文本，不生成研究者推断的六维偏好标签。
- manifest validator 强制许可、来源版本、用途、标签兼容性和 JSONL SHA-256；导航指令若声明为偏好监督会失败。
- 人类偏好 schema 新增独立 reviewer、分歧裁决、批次/prompt/语言/consent 来源字段，并输出裁决前 raw agreement 与 Cohen's kappa。
- 修复 v8 只在 suite 表面删除 collision、而 decoder/reward 仍把它作为第七偏好轴的矛盾。semantic profile 统一为六轴，canonical collision weight 固定 1.0，环境拒绝 collision relaxation；旧 smoke 被标记为 superseded。
- 固定 AerialVLN v8 压缩包和 2,310 条 `val_unseen` 派生指令哈希。128 条 OOD smoke 在未校准 0.20 偏移阈值下仍有 39.06% 激活，暴露出拒答机制缺口。
- `run_research_study.py` 新增 `--only-variants`/`--only-seeds`。部分运行保留完整注册配置、manifest 标为 `partial`；只有所有 variant×seed 结果都存在并通过身份检查时才生成统计和 `complete` 状态。
- 新增基于 smoke 实测墙钟和 environment-step workload 的预算器。架构 v3 60 runs 粗估 60.9–105.2 GPU-hours；5-seed 链式消融 50 runs 粗估 82.2–105.0 GPU-hours，均标记为高不确定性并要求先做 100-episode calibration。

### 论文影响

- AerialVLN 只支持“真实 UAV 域语言 OOD 误接收率”证据，不能支持六维偏好准确率或行为对齐结论。
- 长实验不再需要靠缩减预注册 episode 数避免中断；分块命令保持同一完整协议和结果身份。
- 修正此前约 52 GPU-hours 的过度乐观消融估计；在未做 calibration 前不得承诺最终算力成本。

### 后续验证

- 下载并固定 AerialVLN v8 原包/派生 JSONL 哈希，运行解码器 OOD 拒答 smoke；
- 招募独立 writer/reviewer，冻结人类偏好 test；
- 在 clean commit 上运行 100-episode calibration，再更新 GPU 预约和分块大小；
- 为 partial→complete 生命周期增加端到端中断恢复集成测试。

## 2026-08-20：加入独立 actor 的顺序 HAPPO 强基线

### 工程变更

- 依据 Kuba et al. ICLR 2022 原论文和 PKU-MARL/HARL 官方实现（固定 `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`）加入 `src/happo_baseline.py`。
- HAPPO 为每个 UAV 建立独立 actor，不共享参数。每次 rollout update 随机排列 agents；一个 agent 完成全部 PPO epochs 后，使用更新前后 likelihood ratio 乘到 factor，再训练下一个 agent；集中式 MLP critic 在所有 actor 之后更新。
- 协议 validator 强制 `actor_parameter_sharing="independent"`、`update_scheme="random_sequential_likelihood_factor"`、无 intent、无 mask、direct policy、无 safety filter。artifact validator 逐 result 核验 actor count 与实现元数据，阻止共享 MAPPO actor 被重命名为 HAPPO。
- HAPPO checkpoint 分别保存 8 个 actor/optimizer；单元测试覆盖无参数别名、逐 actor 更新、有限 factor、agent-count 拒绝和完整 checkpoint round-trip。
- 新增架构 v3 smoke/paper 配置。v3 smoke 已完成 6 variants × 1 seed，artifact 审计为 `valid`：6/6 results、10 checksums、0 errors。
- 修复 `generate_paper_artifacts.py` 的证据等级硬编码：smoke 图标题、报告文件名、seed/eval 数量全部来自配置，单 seed 报告禁止效果推断；不再输出沿用旧 pilot 的固定结论。

### 论文影响

- 强基线集合推进为 I-MAPPO/no-mask、MAPPO、IPPO、HAPPO、MATD3 六条可区分计算路径。HAPPO smoke 的 8 独立 actors 共 916,528 个可训练 actor 参数，集中式 critic 173,057 个；这些规模必须与性能共同报告。
- HAPPO smoke 日志中的随机首 agent 为 2/7/5 等，sequential factor 有限但最大值曾达 4.90，提示正式实验需报告 ratio/factor 稳定性。
- 1-seed/10-episode smoke 只验证实现，不支持 HAPPO 与任何算法的性能排序；正式强基线证据仍需 clean 10-seed v3 paper run，并最好对官方 HARL 环境适配做独立交叉复现。

## 2026-08-20：CBF 等价加速与完整链式消融 smoke

### 工程变更

- 定位到 8 UAV 的 cyclic pairwise CBF 在 CUDA 上对每个 pair 反复调用 `.item()`，每步触发大量 host/device 同步；diagnostics 又重复同类同步。
- 保持原 Gauss--Seidel pair 顺序、4 轮投影、active-set 判断和全局 action-box clipping，将 28 条小规模约束合并为一次 host-side float32 投影事务，并在同一遍中生成诊断。评估资源统计复用策略刚产生的诊断，不再重复求值。
- 新增 `benchmark_cbf_runtime.py` 和随机张量旧实现对照测试。RTX 3050 上 filter+diagnostics 从 54.80 ms 降至 0.83 ms（66.24×）；CPU 从 7.61 ms 降至 0.76 ms（10.08×）。动作最大绝对误差 `4.25e-7`，诊断最大误差 `2.03e-8`。
- 完整重跑 `uav_imappo_ablation_smoke_v2`：10 variants × 1 seed × 10 episodes，manifest `complete`，artifact validator 检查 10/10 results、14 checksums、9 条链式契约和所有有效计算路径，状态 `valid`、0 errors。
- 自动生成 `docs/paper/generated/uav_imappo_ablation_smoke_v2/`：消融均值、18 条主比较、资源审计、森林图、报告和生成文件 manifest。

### 论文影响

- 优化只改变执行位置和同步次数，不改变注册的 CBF 数学约束；它不是新的安全算法，也不提升形式化保证。host-side 小规模求解假设及数值等价容差必须在实现细节中披露。
- 冷启动 full 变体耗时 68.65 s；其余缓存变体平均 12.56 s。`no_cbf` 为 13.14 s，说明优化后 CBF 已不再主导该 smoke 的端到端成本。
- 单 seed smoke 的 CI 退化且 exact/Holm p 均为 1，只证明实验、统计和制图管线连通。所有 effect 点估计禁止写成方法贡献；因果结论仍需 clean 多 seed pilot/paper 运行。

### 后续验证

- 在 clean commit 上重跑 smoke，确认无 dirty warning；
- 先执行资源预算合适的多 seed 消融 pilot，再运行 10-seed 架构 paper 协议；
- 继续补齐独立语言语料、跨场景泛化、冲突 SITL、HIL/实机证据。

## 2026-08-20：撤回伪 MAPPO 解释，加入可机器审计的链式消融

### 证据更正

- 语义协议审计发现，历史 `uav_imappo_main` 中名为 `mappo` 的变体实际注册为 `algorithm="imappo"`；同时 `critic_mode="concat"` 从未有实现分支，旧代码将其落入 attention critic。MATD3 忽略该字段，IPPO 又在构建时静默覆盖为 local critic。
- 因此 2026-08-18 的“标准 concat MAPPO”解释撤回。40 个逐 seed 文件、summary 和 checksum 仍保留为历史 artifact，但最新审计状态为 `invalid`，错误记录在 `docs/paper/audits/uav_imappo_main_semantic_protocol_audit.json`。
- 该更正不是统计显著性变化，而是 treatment identity 失败；任何 MAPPO 优劣、attention-vs-concat 或强基线结论均不得继续引用这些数值。

### 工程变更

- 新增 `src/research_protocol.py`：拒绝未实现的 critic mode、保留算法名与实际 algorithm 不一致、以及 IPPO 非 local / MAPPO 非 centralized critic 等协议错误。
- 新增 `intent_source="none"`：I-MAPPO actor/critic 获得严格全零 intent；任务标签、奖励画像、共同随机数和 posture-derived action mask 保持不变，并在 metadata 中显式披露 mask 侧信道。
- 新增 `src/research_ablation.py`：消融形成以 full treatment 为根的有向树；每个非 treatment 变体必须有且仅有一个 reference，并精确声明 changed fields、主风险层、主指标和可证伪假设。任何未声明漂移都会在 dry-run 前失败。
- 统计 runner 按契约计算 `variant-reference` 链式效应，而不再把所有消融错误地与 full 比较；Holm family 只包含预注册 primary tiers × metrics。
- 冻结 MiniLM/CrossEncoder 按 model/revision/device 在单进程复用，避免不同变体重复加载；逐 seed 结果新增 wall time、CUDA 峰值、模型参数和文本模型缓存审计。
- 新增 `generate_ablation_artifacts.py`，从通过审计的 artifact 自动生成均值表、链式比较表、资源表、森林图、报告和哈希 manifest。
- 新增修正架构协议 `uav_marl_architecture_v2.{smoke,paper}.json`：真实 MAPPO 使用 centralized MLP critic，IPPO 显式 local，MATD3 标记为 centralized twin critics。架构 smoke 已完成并通过 5/5 variant identity 与 checksum 审计；数值不作效果推断。

### 论文影响

- 强基线门槛从“已有 10-seed 主实验”退回为“实现和 smoke 已验证，paper 主实验未执行”。这是必要的证据降级。
- 增强消融预注册 10 个变体、9 条链式对比：mask、attention、intent shaping、CBF、NLI gate、learned residual、semantic rule prior、semantic-vs-identity、identity-vs-no-intent。
- `identity_oracle` 是 canonical-label 控制，不是文本理解；`no_intent` 的 mask 侧信道必须与结果同时披露。

### 后续验证

- 完成增强消融 smoke artifact 审计，再决定 50-run pilot 的预算；
- 在 clean commit 上运行修正的 50-run architecture paper protocol，而不是修补或重命名旧结果；
- 独立语言语料、冲突场景策略进 SITL、HIL/实机仍是不可由本轮工程修复替代的硬门槛。

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
