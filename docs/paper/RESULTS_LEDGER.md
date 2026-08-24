# 正式结果台账

## 2026-08-24 VMAS navigation calibration（完成，非论文结果）

- 范围：5 algorithms × seed 7 × 100 training episodes × 100 steps × 20 eval episodes；只注册原生 episode return。
- 点值：attention-PPO **-2.711723**、MAPPO **-0.790685**、IPPO **-1.913966**、HAPPO **0.814603**、MATD3 **1.513628**。单 seed 禁止算法排序、显著性或等效性主张。
- process CPU seconds：142.59、141.28、153.75、208.94、151.84；CUDA peak 22.88–27.09 MiB。
- MAPPO 首轮 return 与复跑值逐 double 相同（`-0.7906847595237195`），但首轮 CPU=457.98 s、复跑 CPU=141.28 s；首轮文件保留并排除出预算。
- 旧 attention pre-instrumentation 结果 wall=16,556.20 s 且无 CPU time，受宿主挂起污染，已保留为 legacy、禁止据此外推。
- 计入 on-policy 训练期监控、MATD3 无周期评估以及 final evaluation 后，navigation paper 预算为 **33.42–44.36 active CPU-hours**；它不是 GPU device-hours，单 calibration seed 仍有中高不确定性。
- 审计：`docs/paper/audits/vmas_navigation_architecture_v1_calibrated_runtime_plan.json`；图表：`docs/paper/generated/vmas_navigation_architecture_v1_calibration_v2_active_time/`。

## 2026-08-24 训练恢复验证（工程证据）

- IMAPPO、独立 actor HAPPO、MATD3：四 episode 连续运行与 episode 2 后中断/恢复的最终模型逐 tensor 相等。
- MATD3 额外验证目标网络、replay、environment-step 与 delayed actor-update 游标；协议/代码身份错误必须拒绝加载。
- 全量测试：**176 passed, 14 warnings**。该结果属于复现基础设施验证，不是算法效果结果。

## 2026-08-24 VMAS dispersion calibration（完成，非论文结果）

- 范围同 navigation：5 algorithms × seed 7 × 100 training episodes × 100 steps × 20 eval episodes，仅原生 return。
- 点值：attention-PPO **0.050000**、MAPPO **0.016667**、IPPO **0.033333**、HAPPO **0.033333**、MATD3 **0.133333**；单 seed 禁止排序。
- process CPU seconds：121.17、126.64、173.95、261.34、135.36；CUDA peak 26.88–29.77 MiB。
- 50-run paper active-time 预算为 **34.09–45.47 active CPU-hours**，不是 GPU device-hours。
- 审计：`docs/paper/audits/vmas_dispersion_architecture_v1_calibrated_runtime_plan.json`；图表：`docs/paper/generated/vmas_dispersion_architecture_v1_calibration_v1_active_time/`。

## 2026-08-24 UAV architecture v3 calibration（完成，非论文结果）

- 范围：6 algorithms × seed 7 × 100 training episodes × 100 max steps；训练监控/碰撞 probe 各 20 episodes，最终 easy/medium/hard 各 20 episodes。
- hard-tier `(collision_rate, task_completion)`：IMAPPO `(0.737, 0.54388)`、no-mask `(0.828, 0.53636)`、MAPPO `(0.809, 0.54098)`、IPPO `(0.639, 0.54625)`、HAPPO `(0.740, 0.55310)`、MATD3 `(0.536, 0.54618)`。单 seed 禁止排序或显著性主张。
- process CPU seconds：45.16、41.05、39.83、39.45、289.33、121.42；CUDA peak 24.92–35.62 MiB。
- MATD3 wall=3711.34 s 与 CPU=121.42 s 严重分离，属于宿主挂起/调度污染；wall 不进入预算。
- 逐算法 workload 外推的 60-run paper 预算为 **70.76–96.04 active CPU-hours**；HAPPO 占 36.17–48.22 h，MATD3 占 13.91–20.24 h。
- 审计：`docs/paper/audits/uav_marl_architecture_v3_calibrated_runtime_plan.json`；产物：`docs/paper/generated/uav_marl_architecture_v3_calibration_v1_active_time/`。

## 2026-08-20 CityNav 预注册一次性外部 OOD 终测

- 范围：四个 canonical split，共 32,637 条；31,751 条规范化唯一文本。
- 冻结 gate：SHA-256 `8518d9be...87fd`，threshold `0.0244081132`。
- 总体 accepted 31,381；FAR **0.961516**，Wilson 95% CI **[0.959374, 0.963549]**。
- 四个 split FAR 为 0.9564–0.9699；预注册判定 **fail**。
- 证据级别：真正未查看文本的一次性外部 OOD，但只有 navigation-negative ground truth；不能估计偏好准确率。
- 约束：结果不可删除，CityNav 不得再用于调 gate/threshold。

## 2026-08-20 VMAS architecture-only smoke v1

- navigation 与 dispersion：各 5 algorithms × 1 seed × 3 eval episodes，artifact valid，dirty-worktree warning。
- navigation episode-return 点值：attention PPO 0.4047、MAPPO 0.3730、IPPO 0.3939、HAPPO -0.1106、MATD3 -1.6318。
- dispersion 点值：attention PPO/MAPPO/IPPO/MATD3 均 0.1111，HAPPO 0.2222。
- 解释：极短单 seed pipeline smoke；严禁排序、显著性或等效性主张。task completion 使用 UAV schema 时为 0，故明确排除；只保留原生 return。

当前没有满足 `paper` 或 `frozen` 协议的结果。

## 2026-08-20 AerialVLN OOD 语言 smoke 与合同迁移

AerialVLN v8 `val_unseen` 导出 2,310 条无偏好标签的人类 UAV 导航指令；原包、派生 JSONL 和 128 条确定性抽样均有 SHA-256。六轴 NLI prototype decoder 在该抽样上的最大 profile 偏移 median/p95 为 0.0829/0.4972；未校准 0.05/0.10/0.20 阈值的激活率为 58.59%/46.88%/39.06%。这是 OOD 风险 smoke，不是分类结果，不能据此选阈值；它暴露出纯导航语言被误判为 `distance:low` 的主要失败模式。

同轮审计发现旧 decoder/reward 仍允许语言改变 collision weight，现已迁移为六轴 preference + 固定 collision safety contract。下列架构 v3 与消融 smoke 文件仍是其旧 snapshot 的有效 pipeline artifact，但已被新合同 supersede；必须 clean 重跑后才可继续 calibration/pilot。

## 2026-08-20 架构 v3 HAPPO smoke

`uav_marl_architecture_v3_smoke` 在 v2 的 I-MAPPO/no-mask、MAPPO、IPPO、MATD3 上加入 HAPPO。HAPPO 使用 8 个独立 actor、随机顺序更新、前序 likelihood-ratio factor 和 centralized MLP critic；result metadata 与 validator 均核验这些身份。artifact 为 `valid`：6 variants、1 seed、6 results、10 checksums、0 errors，保留 dirty warning。

自动报告：`docs/paper/generated/uav_marl_architecture_v3_smoke/SMOKE_STATISTICAL_REPORT.md`。报告生成器已按 `level=smoke` 明确禁止效果推断。HAPPO hard-tier 点估计 collision/task/return 为 0.9867/0.5590/−41.33，但只有 1 seed、3 eval episodes，**不得**与其它方法排序或写入摘要。该点只用于确认连续动作、独立 actor 和顺序更新端到端连通。

## 2026-08-20 增强链式消融 smoke v2

`experiments/smoke/uav_imappo_ablation_smoke_v2/` 已从头完成 10 个变体 × 1 seed × 10 episodes。artifact 审计为 `valid`：10/10 result files、14 checksum entries、9 条预注册链式 comparison、0 errors；dirty-worktree warning 被保留。生成报告位于 `docs/paper/generated/uav_imappo_ablation_smoke_v2/`。

本实验只验证 full、mask、attention、intent shaping、CBF、NLI gate、learned residual、semantic rule prior、identity oracle 和 no-intent 的执行路径及统计契约。由于只有一个 seed，bootstrap CI 退化为点、exact/Holm p 均为 1；下表只记录管线点估计，不进入论文效果台账：

| 注册比较（variant − reference） | hard collision Δ | hard task Δ |
|---|---:|---:|
| no-mask − full | 0.0000 | +0.0519 |
| no-attention − full | 0.0000 | +0.0000 |
| no-intent-reward − full | 0.0000 | +0.0000 |
| no-CBF − full | +0.0133 | +0.0006 |
| no-NLI-gate − full | 0.0000 | −0.0046 |
| prior-only − full | 0.0000 | −0.0000 |
| no-profile-prior − full | 0.0000 | +0.0116 |
| identity-oracle − no-profile-prior | 0.0000 | +0.0000 |
| no-intent − identity-oracle | 0.0000 | −0.0000 |

这些值不能证明任何机制有效或无效；尤其不能把零差异解释为等效。`no_intent` 仍保留 posture-derived action mask 侧信道，`identity_oracle` 也不是自然语言理解基线。

## 2026-08-20 架构 pilot 解释撤回与修正 smoke

最新语义协议审计将 `uav_imappo_main` 判为 `invalid`。虽然 40/40 per-seed 文件与 44 项 checksum 完整，但 treatment identity 不成立：`mappo` key 实际使用 `algorithm="imappo"`，未实现的 `critic_mode="concat"` 又执行为 attention。该 artifact 只能作为失败案例研究，不能作为 MAPPO、attention-vs-concat 或强算法比较证据。

审计：`docs/paper/audits/uav_imappo_main_semantic_protocol_audit.json`。2026-08-18 的 pre/post repair 审计只证明文件身份和 provenance 修复，在加入计算语义验证之前生成，不覆盖本次错误。

修正后的 `uav_marl_architecture_v2_smoke` 已完成 5 个变体 × 1 seed × 10 episodes，并通过 artifact 审计：I-MAPPO attention、I-MAPPO no-mask、centralized-MLP MAPPO、local-critic IPPO、centralized-twin-critic MATD3 的注册算法与实际路径一致。其 hard-tier 点估计仅用于管线诊断，不进入论文效果台账；正式证据必须来自 clean 10-seed、100 eval episodes/tier 的 `uav_marl_architecture_v2_paper`。

## 2026-08-18 I-MAPPO 架构 pilot（历史数值；方法解释已撤回）

Study：`experiments/pilot/uav_imappo_main/`。8 UAV、6 targets、3000 training episodes，I-MAPPO/MAPPO/MATD3/IPPO 各 10 个配对 seeds；每 seed/tier 50 个评估回合。四种方法均使用 one-hot intent 并关闭 intent reward。

当时的结构 validator 结果为 `valid`，只证明 40/40 per-seed results、44 checksum entries 与 JSON 身份一致；2026-08-20 新增语义协议检查后结果为 `invalid`。修复前顶层 provenance 只覆盖 IPPO，历史审计分别保存在：

- `docs/paper/audits/uav_imappo_main_pre_repair.json`
- `docs/paper/audits/uav_imappo_main_post_repair.json`

| Variant | Easy collision | Medium collision | Hard collision | Easy task | Medium task | Hard task |
|---|---:|---:|---:|---:|---:|---:|
| I-MAPPO | 0.1191 | 0.2090 | 0.5478 | 0.5510 | 0.5556 | 0.5542 |
| MAPPO | 0.1135 | 0.1777 | 0.4221 | 0.5481 | 0.5530 | 0.5518 |
| MATD3 | 0.3779 | 0.4691 | 0.5640 | 0.5807 | 0.5876 | 0.5864 |
| IPPO | 0.3030 | 0.4059 | 0.6249 | 0.6479 | 0.6533 | 0.6562 |

这些数值是历史输出，不再解释为 I-MAPPO 相对 MAPPO 的效应；Holm 校正不能修复错误的算法身份。

生成产物：`docs/paper/generated/uav_imappo_main/`，包含两张图、两个 CSV、完整统计报告和哈希 manifest。

限制：除 pilot、50 eval、dirty worktree 和无自然语言外，还存在决定性的 algorithm/critic identity 错误。因此不能登记为任何方法比较证据。

## 2026-08-02 Betaflight SITL 单机闭环 smoke

`betaflight_sitl_closed_loop_smoke_v19`（seed=7，单机，WSL2 Betaflight × Windows PyBullet）：

| 指标 | 值 | 阈值 | 状态 |
|------|-----|------|------|
| 电机包接收率 | 99.03% | ≥80% | ✅ |
| 轨迹有限 | true | true | ✅ |
| 电机输出 max | 1.0 | >0.05 | ✅ |
| 电机输出 mean (解锁后) | 0.374 | — | — |
| 高度响应 | 0.099→0.249m (+0.15m) | >0.10m | ✅ |

SITL 构建参数：`OPTIONS=SITL_ATTITUDE_DIRECT`，4 处解锁/姿态补丁。

### 多 seed 验证（5 seeds）

`betaflight_sitl_closed_loop_smoke_v19_multiseed`（seeds=7,42,123,256,512）：

| 指标 | Mean | Std | Min | 状态 |
|------|------|-----|-----|------|
| 电机包接收率 | 0.9957 | — | 0.9903 | ✅ |
| 高度增益 | 0.134m | 0.024m | — | ✅ |
| 电机输出 max | 1.000 | — | 1.000 | ✅ |
| 电机输出 mean | 0.410 | — | — | — |
| 通过率 | **5/5** | — | — | ✅ |

总耗时 181.6s（5 seeds 串行）。所有 seed 全部通过验收标准。
此结果为 smoke 级别（5 seeds），不是冻结论文实验（需 10+ seeds 及冲突场景）。

## 2026-08-02 Betaflight SITL 多机闭环 smoke

`betaflight_sitl_multi_closed_loop_smoke_v2`（seed=7，双机，各独立 SITL 实例，端口偏移）：

| 无人机 | 高度响应 | 电机包率 | 电机 max | 电机 mean | 状态 |
|--------|----------|----------|----------|-----------|------|
| Drone 0 | 0.100→0.308m (+0.21m) | 99.60% | 1.0 | 0.401 | ✅ |
| Drone 1 | 0.100→0.217m (+0.12m) | 99.78% | 1.0 | 0.324 | ✅ |

多实例通过 `--port-offset` 实现（bf0:9002-9004, bf1:9012-9014）。
此结果为 smoke 级别（1 seed），不是冻结论文实验。

## 2026-08-01 PyBullet Crazyflie 跨动力学 pilot

`pybullet_transfer_smoke_v1` 将高层速度控制与 QP/循环安全层放入 Crazyflie 刚体/旋翼
模拟器。QP 将最小间距从无过滤的 0.122 m 提高到 0.209 m，但目标成功率为 0，暴露
局部屏障在对称互换任务中的死锁。观察后开发的高度通道 smoke v2 只作诊断。

冻结后的 `pybullet_transfer_lanes_pilot_v1` 使用 5 seeds、两类场景、三个目标 profile
和 300 个 30 Hz 控制步。QP 目标成功率 0.9917（95% CI [0.975,1.000]），碰撞步比例
为 0，最终 RMSE 0.0425 m，平均延迟 0.4335 ms。相对无过滤，最小间距增加 0.0905 m
（[0.0824,0.0983]），碰撞步比例减少 0.00756（[-0.01956,-0.00044]），安全距离违规
步比例减少 0.2952（[-0.3241,-0.2574]），成功率增加 0.0667（[0.0250,0.1167]）。

QP 与 100 轮循环投影的安全/任务差异不稳定，但快 0.3377 ms。尽管 QP 命令约束残差
为 0、审计成功率为 1，真实轨迹仍有 23.83% 步低于安全距离，明确显示一步速度模型与
PID/旋翼闭环的失配；本结果不支持形式化安全或 sim-to-real 保证。

## 2026-08-01 延迟预算鲁棒 QP pilot

`pybullet_robust_qp_smoke_v1` 在观察上述模型失配后开发，因此只作诊断。随后冻结的
`pybullet_robust_qp_pilot_v1` 使用 5 seeds、300 步，并启用 PyBullet ground/drag/
downwash 组合物理。鲁棒距离使用预注册 80 ms 跟踪预算，对应 0.04 m 最坏双机闭合 margin。

鲁棒 QP 的成功率为 0.9167，物理碰撞步比例 0，安全距离违规步比例 0.000556，最终
RMSE 0.0815 m，命令约束审计成功率 1，平均求解 0.4117 ms。相对原始 QP，最小间距
增加 0.01173 m（95% CI [0.01058,0.01328]），安全违规步比例减少 0.00378
（[-0.00611,-0.00144]），成功率增加 0.0500（[0.00833,0.09167]），RMSE 减少
0.00912 m（[-0.01556,-0.00275]），归一化控制能量减少 0.01350
（[-0.01486,-0.01242]）。相对无过滤，成功率增加 0.125，安全违规步比例减少 0.0330。

结果表明保守 margin 在该模拟器中同时改善安全和任务，而非靠静止实现；但 0.0556%
的物理安全违规仍非零，因此只支持“经验鲁棒性改善”，不支持确定性安全保证。

## 2026-08-01 NLI、CBF 与动态意图诊断

`uav_nli_semantic_cbf_pilot_v1` 完成 5 seeds × 6 variants × 3 风险层。v4 盲措辞下，similarity-gated NLI 的七个反事实目标 profile rank 为 distance/energy/collision/safety/task/time/threat = 0.866/1/0.866/0.5/1/1/1，非目标平均误差接近 0；ungated NLI 的非目标 MAE 为 0.096–0.181。行为上主方法在 hard 的七目标相关为 0.5/1/0.7/-0.3/1/1/1，critical 为 0.6/1/0.5/0.1/1/1/1。safety 没有稳定单调性。CBF 与 no-filter 的碰撞率相同（hard 0.002、critical 0.024），平均动作修正仅 0.005/0.015，故该 pilot 不支持独立安全收益主张。

`uav_dynamic_intent_cbf_smoke_v1` 引入同状态反事实动态切换和 CBF 违例诊断。主动安全半径修复后，oracle safety 低→高在首步产生 hard/critical 0.398/0.926 的动作差；门控 NLI 仅 0.014/0.082。原始概率审计发现 v4 safety-high 被 NLI 解为 0.549，瓶颈位于元话语假设而非控制器。过滤后 critical 最大一步约束违例仍为 0.0174，说明动作盒裁剪/有限次循环投影后仍可能不可行；不能将当前实现称为严格安全保证。

修正为目标特定直接陈述后，`uav_dynamic_intent_postfix_blind_smoke_v1` 在预先冻结的 v5 新措辞上恢复 safety/energy/threat 的首步动态响应，但总体 profile MAE 仍为 0.124，energy/collision/time 多项行为相关为负，且最大非目标 range 达 0.497。v5 因而是失败的后修复盲测，并从此冻结为诊断集，不得继续调参后重复声称确认成功。下一候选 `nli_prototype_gated` 只用独立训练原型建立七目标+中性最近质心，尚无结果。

以上结论把当前论文定位收窄为“语义目标解码与可审计控制协议”的候选，而不是已证实的语言 MARL 或严格安全控制方法。残差 PPO 仍未显示贡献。

## 2026-08-01 v6/v7 与 5-seed 动态确认 pilot

`uav_prototype_gate_blind_smoke_v1`（v6）表明目标原型门控消除了串扰，但单假设 NLI 极性仍过弱；energy/collision 动态响应均未过 0.05 阈值。`uav_polarity_prototype_blind_smoke_v1`（v7）中，15 类纯原型极性在 distance/collision/task/threat 的 profile rank 为 0，并产生最大 0.7 串扰，路线被停止。相同 v7 上的 NLI 原型门控表现较好，因此进入 5-seed pilot；v6/v7 在观察后均冻结，不得再作为无偏集。

`uav_nli_prototype_dynamic_pilot_v1` 完成 5 seeds、33 queries、hard/critical、每层 3 episodes。主方法 hard 的 distance/energy/collision/safety/task/time/threat 行为相关均值为 1.0/0.8/0.3/-0.1/0.5/0.9/1.0；critical 为 1.0/0.8/0.7/-0.3/0.5/1.0/1.0。collision 与 safety 的 hard CI 跨零，safety 的 critical CI 也跨零。动态切换中，safety 与 threat 响应率均为 1、平均删失延迟 0；energy 与 collision 响应率均为 0、删失延迟 20 步。oracle 对四类均有响应（collision hard/critical 为 0.87/0.93），证明失败不是协议完全不可激活。

CBF 审计中，主方法 hard/critical 平均最大残余一步违例为 0.00032/0.00464，旧相似度门控为 0.00169/0.01026；碰撞率为 0/0.0089。该差异可能来自 profile 改变的安全距离，且评估 episode 太少，不能声称碰撞概率优势。

`experiments/runtime/uav_nli_prototype_dynamic_runtime_v1.json`：CPU 冷构造 11.285 s，33-query 首次批处理 2.427 s，缓存 profile 每批均值 30.7 µs（每 query 0.93 µs），重复完整编码 72.0 ms/批；MiniLM/NLI 参数量为 22.7M/141.9M。运行时结论是“任务级缓存可行、控制环冷启动不可行”。

## 2026-08-01 盒约束 QP-CBF 5-seed pilot

`uav_qp_cbf_pilot_v1` 比较 QP、4/32 轮循环投影和无过滤，使用 5 seeds、hard/critical/adversarial、每层 10 episodes。adversarial 碰撞率：QP 0.0144、cyclic-4 0.0184、cyclic-32 0.0152、none 0.0184。QP 相对 cyclic-4 和 none 的配对差均为 -0.004，95% CI 分别为 [-0.0056,-0.0020]、[-0.0064,-0.0016]；相对 cyclic-32 为 -0.0008，CI [-0.0016,0]。QP 与各方法的任务完成差均接近 0 且 CI 跨零。

QP hard/critical/adversarial 审计可行率为 99.96%/99.28%/98.52%，平均最大残余一步违例为 0.000005/0.000862/0.002896；不可行状态被保留而非删除。初版无条件执行 fallback，平均延迟 14.31 ms。修复为只在失败/超 tolerance 时 fallback 后，独立 runtime smoke 的 critical/adversarial 延迟降至 0.751/1.006 ms；cyclic-4 为 1.078/1.100 ms，cyclic-32 为 7.021/7.020 ms。后修复延迟是 1-seed smoke，不改变 v1 的 5-seed安全数值，正式实验需用新 study ID 复现。

## 2026-08-01 prototype-grounded residual 5-seed pilot

`uav_prototype_grounded_residual_pilot_v1` 使用 benchmark v2、5 seeds、50×20 training transitions、33 条自然/反事实 query、easy/hard/critical 三层和每层 3 个评估 episode。prototype residual 在 hard 的 distance/energy/collision-margin/safety/task/time/threat 相关均值为 0.5/0.5/0.5/0.2/0.5/1.0/1.0；critical 为 0.5/0.5/0.5/0.3/0.5/1.0/1.0。除 safety 外，各项 CI 均保持正值；safety 的 hard/critical CI 跨零。

相对 direct ridge，prototype 在 hard 的 energy +1.0（CI [1.0,1.0]）、threat +1.1（[0.7,1.5]），critical 的 energy +1.0、collision-margin +0.8（[0,1.5]）、safety +1.2（[0.8,1.5]）、threat +1.1；但 distance 和 task 均为 -0.5。prototype residual 与 prototype zero-residual prior 的七项目标差全部为 0，表明学习部分没有可测贡献。

`uav_residual_learning_sensitivity_smoke_v1` 将训练扩到 200 episodes。scale=0.25/0.5 的 hard 残差幅度为 0.0127/0.0506，但 zero prior、两个尺度及 no-potential 的目标相关完全相同；absolute return 也未改善。该结果只是一种子调参证据，但否定了立即扩大 nominal PPO 训练的必要性。

## 2026-08-01 反事实目标 grounding smoke v1–v4

v1 首次加入 7×3 minimally contrastive queries，但不同 query 使用不同 reset seed，oracle 也出现反向相关；该结果被判定存在场景混杂，不得作因果证据。v2 采用 common-random-number 配对后，真实画像 oracle 在 distance、energy、task、time 上恢复正确单调关系，collision/safety 只在 hard tier 可辨，threat violation 仍退化。

v3 隔离学习残差后，`contrastive_anchor` 的反事实文本画像在七维均按 low→mid→high 排序；task/time 行为在 easy/hard 均为 Spearman 1.0，energy 为 1.0，collision/safety 主要在 hard tier 有效。v4 的 `prototype_ridge` 将总体画像 MAE 从 direct ridge 的 0.181 降至 0.134，counterfactual MAE 为 0.102；其行为相关为：task/time easy=hard=1.0，distance/energy easy=hard=0.5，safety easy=-0.5、hard=0.5，collision 与 threat 在多数层退化为常量。

以上全部为 1-seed smoke，只能支持三项工程结论：共同随机数是必要条件；原型增强改善画像校准；当前环境/控制器不足以让七维目标全部可辨。不得声称方法已实现完整自然语言遵循。

## 2026-08-01 VMAS formal smoke

安装并锁定 VMAS 1.5.2 后，`vmas_navigation_formal_smoke_v1` 在公开 continuous navigation scenario 上完成 I-MAPPO(one-hot)、MAPPO、IPPO、MATD3 四个变体，生成正式 manifest、逐 seed 结果、summary、result card 和 SHA256。1 seed、3 training episodes、2 eval episodes 仅证明正式跨环境路径；VMAS 原任务不使用语言条件，数值不得作为语义贡献证据。

5-seed `vmas_navigation_pilot_v1`：I-MAPPO return 0.387（CI [0.342, 0.426]）、MAPPO 0.382、IPPO 0.310、MATD3 0.553（std 0.678）；I-MAPPO-IPPO paired difference +0.077（[0.024, 0.114]），其余核心区间跨零。`vmas_dispersion_pilot_v1`：I-MAPPO/MAPPO 0.0267、IPPO 0.0200、MATD3 0.0467，核心 paired intervals 均触及或跨零。不同场景回报尺度不直接平均；两项均为优化器稳定性 pilot，不是语言实验。

## 2026-08-01 扩展资源指标 smoke

`uav_intent_resource_metrics_smoke_v1` 验证 energy/action/speed/distance/min-separation/threat violation 的逐 query、split、paired 与 controllability 汇总。只有 1 seed；例如 objective unseen-hard 的 energy/time/safety-distance 相关出现负值，只能作为“指标能揭示失败”的管线证据，不能估计方向。新的多目标指标必须在 5-seed pilot 复现。

## 2026-08-01 MATD3 smoke

`uav_matd3_smoke_v1` 验证 10 episodes、200 transitions 后的训练、easy/hard 与 held-out query 路径；`uav_matd3_logging_smoke_v2` 进一步记录 100 transitions、34 次 delayed actor updates、最近 actor loss 0.684、critic loss 0.000768。43 项回归测试全部通过。两项均为 1-seed smoke，不得用于算法效果结论。

5-seed `uav_matd3_feasibility_v1` 与残差 v2 使用相同环境、seeds、风险层、评估次数和 query suite。跨研究校验和审计后，objective residual 相对 MATD3 的 easy/hard 碰撞率差为 -0.176/-0.336，回报差为 +7.781/+2.392，但完成度差为 -0.059/-0.085。该结果反映 1000-transition 早期安全—完成度折中；MATD3 尚可能未收敛，不得声称最终支配。

## 2026-08-01 扰动与规模 smoke

- `uav_robustness_smoke_v1` 完成 nominal、wind、sensor noise、2-step latency+25% dropout 和 combined 五档，两个控制器均生成完整校验和结果。
- `uav_scale_transfer_smoke_v1` 将 4 UAV 训练 actor 零样本执行到 8/12/16 UAV；1-seed 下 semantic residual 的 collision rate 分别为 0/0/0/0.05。

限制：每档只有 1 seed、2 episodes，所有数值只验证协议和张量形状，不支持鲁棒性或规模优势结论。

## 2026-08-01 5-seed feasibility 与规则基线否定结果

`uav_intent_generalization_feasibility_v2` 使用 5 seeds、50 training episodes、每风险层 5 个评估 episodes，比较 objective-grounded semantic、raw pretrained、IPPO 与 MAPPO。objective-grounded 方法在部分 easy/paraphrase 回报或碰撞指标上有方向性优势，但未获得稳定的 unseen 优势；同一方法在部分 hard/paraphrase 指标反而更差。因此本实验只支持“管线可运行且存在局部信号”，不支持语义泛化主张。

`uav_rule_planner_feasibility_v1` 使用同一 seeds、环境、风险层、query suite 和 reset 公式。独立产物 `uav_intent_vs_rule_feasibility_v1` 在比较前验证了两边 SHA256 和协议兼容性。主场景 treatment-minus-rule 结果如下：

| Tier | Metric | Mean difference | 95% paired bootstrap CI | 方向 |
|---|---|---:|---:|---|
| easy | collision rate | -0.036 | [-0.128, 0.116] | 不稳定 |
| easy | task completion | -0.135 | [-0.150, -0.116] | 规则更优 |
| easy | episode return | -12.914 | [-17.185, -9.450] | 规则更优 |
| hard | collision rate | +0.140 | [+0.076, +0.204] | 规则更优 |
| hard | task completion | -0.074 | [-0.085, -0.062] | 规则更优 |
| hard | episode return | -0.621 | [-2.719, +2.100] | 不稳定 |

限制：只有 5 seeds 和每 seed/tier 5 个 episodes；这是决定停止扩大 direct-policy 实验的 pilot 证据，不是最终论文比较。

## 2026-08-01 语义残差架构 smoke v1

`uav_semantic_residual_smoke_v1` 是 1-seed、5-training-episode 的因子 smoke。其目的只是验证共享规则先验、零残差初始化和 PPO rollout 概率口径。

| Variant | Easy collision | Easy completion | Hard collision | Hard completion |
|---|---:|---:|---:|---:|
| semantic residual | 0.000 | 0.709 | 0.025 | 0.541 |
| nonsemantic residual | 0.000 | 0.709 | 0.025 | 0.543 |
| semantic direct | 0.000 | 0.579 | 0.500 | 0.461 |
| rule planner | 0.000 | 0.708 | 0.025 | 0.543 |

解释：residual 在极短训练后保持 rule prior，而 direct policy 明显不稳定；semantic 与 nonsemantic residual 几乎相同，尚无语义增益证据。该表不得进入论文主结果。

## 2026-08-01 残差 pilot v1/v2 与探索性可控性

`uav_semantic_residual_pilot_v1` 使用 inverse-tanh 中心且所有 residual 都通过外部 posture 获得规则增益。objective/raw/nonsemantic 几乎重合，暴露出动作饱和梯度过小和 posture oracle 旁路两个问题；该版本被 v2 取代，但保留为方法失败记录。

`uav_semantic_residual_context_pilot_v2` 改用 latent headroom 残差。objective/raw 的先验姿态只由文本向量检索，nonsemantic 使用 neutral，rule-only 单独标成 oracle。5-seed 主结果显示明确 Pareto 交换而非全面支配：objective 相对 nonsemantic 在 unseen easy/hard 的完成度高 0.0039/0.0044、回报高 0.911/0.699，但碰撞率也高 0.0323/0.0423；objective 相对 raw 在 unseen easy/hard 的碰撞率低 0.0410/0.0280，但完成度低 0.0235/0.0218。

对 v2 原始结果进行只读 checksum 审计后，`uav_semantic_residual_controllability_exploratory_v1` 计算了事后探索性指标。all-query 安全折中 Spearman：objective easy 0.182（CI [-0.008, 0.401]）、hard 0.351（[0.158, 0.553]）；raw 为 0.018/0.169，nonsemantic 为 -0.104/0.220，oracle rule 为 0.571/0.667。objective 相对 raw 的 easy/hard 配对差为 0.164/0.182，相对 nonsemantic 的 easy 差为 0.286。task preference 对完成度没有稳定优势。

限制：controllability 指标在看到 v2 平均结果后定义，必须标为 exploratory；只有在新 paper 配置中预注册并复现后才可支持正式主张。目前证据只提示安全语义可控性，不支持完整多目标语言遵循。

## 2026-08-01 泛化 smoke v4

Study ID：`uav_intent_generalization_smoke_v4`。该运行使用 true one-hot、19/6 train/unseen 标签隔离、细粒度奖励画像、neutral 独立姿态和确定性配对 reset seeds。4 个变体均完成，23 项回归测试全部通过。

| Representation | Paraphrase top-1 | Mean margin | Seen return | Paraphrase return | Unseen return |
|---|---:|---:|---:|---:|---:|
| pretrained semantic | 0.50 | -0.034 | -4.058 | -3.888 | -3.949 |
| random dense identity oracle | 1.00 | 0.525 | -4.059 | -3.888 | -3.949 |
| legacy hash | 0.00 | -0.301 | -4.059 | -3.891 | -3.949 |
| one-hot identity oracle | 1.00 | 1.000 | -4.087 | -3.871 | -3.969 |

限制：1 seed、2 个训练回合、每 query 1 个评估 episode；行为结果只验证公平配对管线。v1/v2/v3 依次缺少细粒度奖励画像、确定性配对 seed 或 neutral 独立语义，已由 v4 取代。

产物目录：`experiments/smoke/uav_intent_generalization_smoke_v4/`。

## 2026-08-01 smoke 管线验证

以下结果只验证本地 `rl-test` 环境、编码器和训练/汇总管线可执行。每组只有 1 个 seed 和 1 个 easy-tier 评估 episode，不可比较显著性，也不得进入摘要、主表或结论。

| Study ID | Level | Encoder/Control | Seeds | Eval episodes | Collision rate | Task completion | Episode return | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| uav_intent_pretrained_semantic_smoke | smoke | pretrained semantic | 1 | 1 | 0.000 | 0.426 | -4.665 | complete |
| uav_intent_representation_smoke | smoke | legacy hash | 1 | 1 | 0.000 | 0.602 | -2.093 | complete |
| uav_intent_representation_smoke | smoke | one-hot | 1 | 1 | 0.625 | 0.639 | -5.323 | complete |

产物目录：

- `experiments/smoke/uav_intent_pretrained_semantic_smoke/`
- `experiments/smoke/uav_intent_representation_smoke/`

## 历史结果说明

### Stage7 legacy

- 3 seeds、3000 episodes；
- semantic 条件实际为 static hash；
- 可作为先导实验，不得表述为真实语义或在线推理结果。

### closed_loop_tuning round 01

- 已完成 semantic-hash、one-hot、MAPPO、连续共享价值替代模型、full 和 w/o masking 的 5-seed 运行；
- w/o reward 与 w/o attention 未完成；
- 当前 `stage7_results.json` 被一次 1-seed / 3-episode smoke 运行覆盖；
- 尚无统一统计置信区间和冻结清单。

## 登记模板

| Study ID | Level | Git commit | Encoder | Baselines | Seeds | Eval episodes | Main result | Status |
|---|---|---|---|---|---:|---:|---|---|
| 待定 | paper | 待定 | pretrained semantic | 待定 | >=10 | >=100/seed/tier | 待定 | 未开始 |
