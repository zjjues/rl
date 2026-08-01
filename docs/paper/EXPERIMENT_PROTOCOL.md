# 正式实验协议

## 0. 反事实意图遵循补充协议

- 每个目标至少包含 low/neutral/high 三个 minimally contrastive 文本，显式目标画像只用于评测真值；
- 同一 seed、risk tier、evaluation episode 的所有 query 必须共享 reset seed，禁止 query index 改变初态；
- 先报告文本画像 MAE、逐目标排序和 off-target error，再报告对应行为指标；
- 真实画像 oracle 必须先证明环境—控制—指标链可辨，否则该目标不得进入 confirmatory 假设；
- query-level 相关只在每个训练 seed 内计算，跨 seed bootstrap；不能把 33 条 query 当成 33 个独立样本；
- 任何看过 smoke/pilot 后修改的锚点、查询或增益都必须使用新 study ID，并在 paper 运行前冻结。

## 1. 结果等级

- `smoke`：验证代码路径；不得进入论文表格。
- `pilot`：用于选择超参数和发现失败模式；不得作为最终无偏结果。
- `paper`：预注册配置、完整 seed、固定评估协议，可进入论文。
- `frozen`：绑定代码提交、环境清单与校验和，不再覆盖。

目录必须采用 `experiments/{level}/{study_id}/`，禁止 smoke 结果覆盖 paper 结果。

## 2. 主假设

- H1：语义条件残差策略相比无语义残差策略，在保持规则导航安全性的同时改善意图特定任务效用。
- H2：语义模型在未见同义改写和未见组合意图上具有更好的零样本/少样本迁移。
- H3：意图查询注意力与势函数奖励分别贡献安全性和适应速度，组合效果优于单独组件。
- H4：方法在提高安全性时，不显著降低任务完成率、吞吐量或能效。

## 3. 基线

### 必需连续动作基线

- MAPPO；
- IPPO；
- MADDPG 或 MATD3；
- 规则式目标分配 + ORCA/势场避碰；
- I-MAPPO + one-hot；
- I-MAPPO + random dense code；
- I-MAPPO + legacy hash；
- I-MAPPO + pretrained semantic embedding。
- 规则先验 + nonsemantic MAPPO residual；
- 规则先验 + objective-grounded semantic I-MAPPO residual。

离散 QMIX/VDN 仅可在离散动作环境中作为真实实现运行，不能用其他模型替代后继续使用该名称。

## 4. 消融

- w/o action masking；
- w/o potential reward；
- w/o intent attention；
- frozen / slow / normal potential update；
- random dense vs hash vs pretrained semantic；
- 去除意图—姿态对齐；
- 不同意图维度和意图库规模。
- direct policy vs rule-only vs nonsemantic residual vs semantic residual；
- residual scale 与零残差初始化；
- raw pretrained vs objective-grounded adapter。

## 5. 场景矩阵

- 风险：easy / medium / hard / adversarial；
- 规模：4 / 8 / 12 / 16 UAV；
- 目标：静态、动态、突发重分配；
- 不确定性：风、定位噪声、传感噪声、通信时延与丢包；
- 意图：seen、paraphrase、unseen、compositional、mid-episode switch。
- 公开 benchmark：VMAS navigation 加至少一个 cooperative scenario；原生无语言任务只能验证 MARL 优化器，不能替代 UAV 意图遵循实验。

### 5.1 意图数据隔离

- split 单位是 canonical intent label，而不是随机文本行；
- unseen label 不得出现在训练意图库或调参日志中；
- paraphrase 文本不得与训练描述完全相同；
- 查询集在 pilot 前冻结，paper 运行不得增删；
- random-dense/one-hot 的 paraphrase canonical identity 必须标注为 oracle；
- 表示 retrieval 只能作为机制诊断，不能替代行为泛化结果。

## 6. 指标

### 安全

- episode collision probability；
- 每千步碰撞数；
- 最小机间距及其 5% 分位数；
- 威胁区违规率；
- CVaR 或最坏分位回报。

### 任务与资源

- 任务完成率；
- 完成时间；
- 覆盖率/吞吐量；
- 能耗；
- 路径长度。

### 意图响应

- 意图遵循得分；
- 切换后响应延迟；
- 未见意图泛化差距；
- 文本 embedding 相似度与行为相似度的相关性。
- 意图安全偏好与实际安全—完成度 operating point 的 per-seed Spearman 相关；
- collision preference→低碰撞、task preference→高完成度的分量相关；
- query 间碰撞率与完成度动态范围，避免高相关来自近乎常量行为。

## 7. 统计协议

- pilot 至少 5 seeds，paper 主结果建议至少 10 seeds；
- 每个 seed、场景和风险档至少 100 个确定性评估 episode；
- 报告逐 seed 原始值、均值、标准差、median、IQM 与 bootstrap 95% CI；
- 核心比较使用相同 seed 的 paired bootstrap；
- 跨研究配对必须先校验两个源目录的 SHA256，以及 seeds、环境、风险层、评估次数和查询 suite；比较输出写入独立目录，不修改源研究；
- 同时报告绝对差、相对差与效果方向一致的 seed 比例；
- 预先指定主指标，避免从大量指标中事后选择有利结论。
- 多条 query 不是独立随机 seed；必须先在每个训练 seed 内按 split 聚合，再跨 seed 计算 CI；
- 所有变体使用相同的环境 reset seed 调度，确保配对比较面对相同初态和目标布局；
- 奖励画像倍率必须在 paper 前冻结，并报告 profiles-disabled 与倍率敏感性消融。
- controllability query 相关必须先在每个 seed 内计算，再跨 seed bootstrap；pilot 后新增的探索性定义不得作为 confirmatory 结果，paper 配置前必须冻结。

## 8. 冻结规则

正式结果目录必须包含：

- `manifest.json`：研究等级、Git 提交、日期、命令、机器和依赖；
- `config.json`：完整解析后配置；
- `seed_*/result.json`：逐 seed 原始结果；
- `summary.json`：由统一统计脚本生成；
- `checksums.sha256`：输入与输出校验和；
- `RESULT_CARD.md`：成功、失败、异常和论文可用结论。
