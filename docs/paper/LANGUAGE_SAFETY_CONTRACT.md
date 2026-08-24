# 语言偏好与碰撞安全合同修正

## CityNav 终测后的合同升级

六轴/碰撞分离仍然有效，但“有 relevance gate 即可安全接收自由文本”已被否定。预注册 CityNav 终测的误接受率为 96.15%，因此开发版 gate 只能保留为失败基线，paper 配置继续机器拒绝其 `final_blind_test=false` 元数据。安全入口现在要求两级证据：(1) writer-disjoint 独立人工偏好 test 上的相关性与六轴分类；(2) 多来源 navigation/task OOD 上冻结阈值的低 FAR。任一未通过都必须 neutralize profile，且不能用最终 OOD 反调阈值。

## 发现的问题

v8 泛化 suite 已删除 collision counterfactual，并在文档中称 collision 为不可协商约束；但旧实现仍把 `collision` 放在 `OBJECTIVE_KEYS`，NLI/prototype decoder 输出 7 维 profile，多个 canonical label 还把 collision reward weight 设为 0.7–1.6。环境的 `set_objective_profile` 也接受任意 collision override。

因此“语言不能放宽碰撞安全”的旧表述并未由代码兑现。即使 CBF 存在，语言仍能改变 collision reward shaping；而无 CBF 的架构比较中，该漂移直接改变训练目标。这是构念和安全合同缺陷，不是文档措辞问题。

## 修正

- `PREFERENCE_OBJECTIVE_KEYS` 固定为 distance、energy、safety、task、time、threat；`collision` 独立列入 `SAFETY_CONSTRAINT_KEYS`。
- semantic adapter、NLI hypotheses、prototype、目标诊断和 controllability 只使用六个可协商轴。
- 所有 canonical intent 的 collision reward weight 固定为 1.0；label 仍可改变 safety margin preference，但不能降低接触惩罚。
- 环境拒绝任何 `collision != 1.0` 的 objective profile。
- 从 v2/v7 suite 源文件删除 collision counterfactual，而非在 v8 加载后过滤；避免父 suite 在验证前短暂承载非法偏好。
- 删除 `collision_preference_spearman` 和 `collision_distance_spearman`。碰撞率仍是必须报告的安全结果，但不再被解释为可协商偏好的响应变量。

## 回归合同

单元测试断言：(1) collision 不在语言 objective keys；(2) 25 个 canonical label 的 collision weight 集合严格等于 `{1.0}`；(3) 环境拒绝 collision relaxation；(4) decoder 输出列数等于六；(5) v8 只含六轴 counterfactual。

## 对既有证据的影响

2026-08-20 之前生成的语义消融 smoke 和架构 v3 smoke 使用旧 reward/profile 合同。artifact 文件与其自身 snapshot 仍可通过完整性审计，但已被本修正 supersede，不能作为六轴安全合同的效果证据。clean commit 后必须重跑 smoke，随后才能开始 calibration/pilot/paper。

## OOD 暴露出的下一缺口

AerialVLN v8 `val_unseen` 的 2,310 条人类 UAV 导航指令被固定为无偏好标签的 OOD 负对照。在 128 条确定性抽样 smoke 上，六轴 NLI prototype decoder 的最大 profile 偏移中位数为 0.0829；若事后查看 0.05/0.10/0.20 三个未校准阈值，激活率分别为 58.59%/46.88%/39.06%。100/128 条最大偏移被分到 `distance:low`。

这说明仅删除 collision 轴仍不足以安全解释自由文本：当前 decoder 会把大量纯路径指令当作距离偏好。由于阈值尚无独立 preference dev 数据，不能从本 smoke 选择阈值。正式方法需要在独立 dev 上校准拒答，并以冻结 AerialVLN OOD 集报告误接收率。
