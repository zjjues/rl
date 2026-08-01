# 独立语言偏好标注协议

## 目的

开发者手写的 v4–v7 已用于方法诊断，不能继续充当无偏语言测试。正式解码器必须改用来源独立的标注语料，并将标注者而非句子随机分配到 train/dev/test，防止个人措辞风格泄漏。

任务只标注六个可协商偏好：`distance`、`energy`、`safety`、`task`、`time`、`threat`，每个偏好为 `low` 或 `high`，另设 `neutral`。`collision` 是不可放宽的安全约束，不接受 low/high 标注。

## JSONL 字段

每行必须包含：

- `id`：不可重复记录 ID；
- `text`：标注者自然写出的操作指令；
- `objective`：六个偏好之一或 `neutral`；
- `polarity`：`low`、`high` 或与 neutral 配套的 `neutral`；
- `annotator_id`：匿名稳定标注者 ID；
- `source`：招募批次/平台/实验轮次；
- `split`：`train`、`dev`、`test`。

`preference_dataset.py` 自动拒绝重复 ID、跨 split 规范化文本、跨 split 标注者、非法 collision 偏好和缺失类别。正式数据建议每类至少 50 条、每个 split 至少 5 名独立标注者；test 在模型/阈值冻结前不得查看。

## 标注流程

1. 向标注者展示不含方法关键词的 UAV 场景说明与六项偏好定义。
2. 随机给出目标与极性，让其写一条自己会对操作员说的指令；禁止复制示例。
3. 第二名标注者只看文本，独立选择目标/极性；分歧由第三人裁决并保留原始标签。
4. 记录语言背景、航空/机器人经验的分组统计，但不收集可识别个人信息。
5. 按标注者分割数据；冻结 test 哈希后训练模型。
6. 报告 macro-F1、逐类召回、混淆矩阵、置信校准和拒答率；随后才进入控制实验。

## 文献定位

[PixL2R](https://proceedings.mlr.press/v155/goyal21a.html) 将自由文本任务描述映射为机器人奖励；[Text2Interaction](https://proceedings.mlr.press/v270/thumm25a.html) 将语言偏好与安全控制器结合；[Constrained Multi-Objective RL](https://proceedings.mlr.press/v164/huang22a.html) 强调同时学习偏好与满足约束。本工程采用相同的核心分离：语言调节可协商性能轴，碰撞安全合同不允许由语言放宽。
