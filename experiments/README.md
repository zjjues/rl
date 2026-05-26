# 实验产物说明

本目录只保留当前论文相关的实验状态。

## 保留内容

- `stage7_llm/`：语义意图编码库条件。历史目录名仍为 `llm`，论文中建议表述为“语义意图编码库”。
- `stage7_baseline_onehot/`：one-hot 意图编码基线。
- `vmas_stage1/`：VMAS dispersion 验证实验，用于说明跨环境接入和方法适用边界。
- `STAGE7_REPORT.md`：Stage7 结果摘要。
- `stage7_results.json`：Stage7 关键指标汇总。

## 保留规则

正式实验目录中只保留：

- `summary.json`
- 每个 seed 的 `result.json`
- 最终图表
- 必要的意图库文件

不保留：

- 训练 checkpoint
- 高频 `metrics.csv`
- 高频 `metrics.jsonl`
- smoke 测试输出
- 临时调参结果

后续增加实验时，请按相同规则整理，避免仓库再次膨胀。
