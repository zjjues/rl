# 实验产物与证据等级说明

本目录同时包含历史实验和按新协议生成的研究产物。论文写作以 `docs/paper/RESULTS_LEDGER.md` 为准。

## 保留内容

- `stage7_semantic_library/`：历史 legacy hash 条件，仅作为 pilot；不得表述为真实语义 embedding。
- `stage7_baseline_onehot/`：one-hot 意图编码基线。
- `vmas_stage1/`：VMAS dispersion 验证实验，用于说明跨环境接入和方法适用边界。
- `closed_loop_tuning/`：闭环调参脚本的短配置验证结果，用于确认 5 seed 多候选流程可运行；不作为论文主结果。
- `STAGE7_REPORT.md`：Stage7 结果摘要。
- `stage7_results.json`：Stage7 关键指标汇总。

历史目录不会自动迁移或删除，以保留可追溯性。新的运行入口不会再覆盖这些文件。

## 新目录

- `smoke/`：路径验证；
- `pilot/`：调参与探索；
- `paper/`：预注册正式实验；
- `frozen/`：投稿冻结资产。

`run_research_study.py` 会为研究生成 manifest、完整配置、逐 seed 结果、bootstrap/IQM 汇总、结果卡和校验和。

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
