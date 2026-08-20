# 可发表性差距审计

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
| 6. 强算法基线 | 部分完成 | 修正的 centralized-MLP MAPPO、local IPPO、MATD3 与 I-MAPPO smoke 已通过实际路径审计；10-seed 旧结果不可用，仍缺 clean v2 paper run、公开实现交叉核验/HAPPO 类基线 | 未通过 |
| 7. 算法新颖性与因果消融 | 部分完成 | 已预注册 10 变体/9 条链式单因素契约并加入 zero-intent/identity 控制；增强 smoke 尚在验证，pilot 未执行，学习贡献仍未成立 | 未通过 |
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

下一步不应直接运行 10-seed paper 配置，也不应把 v4–v7 继续循环调参后称为盲测。应建立来源独立、带目标/极性标注的人类或外部语料，训练并校准偏好解码器；预注册一个从未参与开发的最终语言测试集。只有 energy/collision 等在线切换达到稳定响应、safety 在多 seed 上可辨、CBF 可行性问题被解决后，才进入扰动/规模和 paper 级实验。

CPU 资源审计显示：冻结双模型构造约 11.28 s，33-query 首次批量推理约 2.43 s；profile 缓存后每条约 0.93 µs，重复完整 MiniLM 编码约 72 ms/33 条。在线系统必须在任务下达时预编码并缓存，不能在控制环内冷启动 1.42 亿参数 NLI 模型。

QP-CBF 已在 5-seed adversarial pilot 中把碰撞率从 cyclic-4/none 的 0.0184 降至 0.0144，且完成度差异接近零。优化后 4-UAV CPU 延迟约 1 ms，但 adversarial 审计成功率只有 98.52%；因此安全工程门槛从“缺少可行性审计”推进为“已量化不可行率，尚需高阶动力学与备用策略”。

Crazyflie 刚体 pilot 进一步显示：速度空间 QP 在 5 seeds 下实现 99.17% 目标成功和零
物理碰撞步，且相对无过滤显著改善间距、碰撞、成功和 RMSE；但仍有 23.83% 的步低于
规划安全距离。真实性门槛因此从“无高保真动力学”推进为“已有高保真风险降低证据，
尚无鲁棒轨迹保证与 SITL/HIL/实机闭环”。

80 ms 延迟预算的鲁棒 QP 在 ground/drag/downwash 组合物理下把安全违规步比例降至
0.0556%，并相对原始 QP 同时提高目标成功 5 个百分点；但违规没有归零，80 ms 预算也
尚未由独立系统辨识覆盖率验证。安全证据门槛推进为“已有多种子经验鲁棒性”，仍未达到
“形式化保证或真实系统验证”。
