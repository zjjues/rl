# I-MAPPO 空基多智能体协同调度实验仓库

本仓库基于 EPyMARL/PyMARL 多智能体强化学习框架改造，当前主要用于论文实验中的 I-MAPPO 算法、UAV 调度环境、语义意图编码实验和少量 benchmark 验证。

原 EPyMARL 框架能力包括多智能体算法训练、环境封装、配置管理和结果绘图等。本仓库保留这些基础代码，但 README 不再展开上游框架的完整英文说明；如需查看原始框架用法，可参考 EPyMARL 官方仓库。

## 当前仓库内容

- `src/`：核心代码，包括 I-MAPPO、UAV 调度环境、语义意图库、VMAS 适配器和实验脚本。
- `experiments/`：当前保留的论文相关实验状态。
- `docker/`：原框架 Docker 相关文件。
- `requirements.txt`、`env_requirements.txt`、`pac_requirements.txt`：依赖文件。
- `run.sh`、`runalgo.sh`、`run_interactive.sh`：原有运行入口。
- `plot_results.py`：结果绘图脚本。
- `P0实验完整报告.md`：当前 P0 阶段实验综合说明。

已删除旧版 `reports/`、smoke 测试、中间日志、checkpoint 和过时英文报告，方便后续继续增加实验。

## 当前保留的实验状态

当前只保留与论文主线直接相关的实验结果：

### 1. Stage7 语义意图编码实验

路径：

- `experiments/stage7_llm/`
- `experiments/stage7_baseline_onehot/`
- `experiments/STAGE7_REPORT.md`
- `experiments/stage7_results.json`

说明：

- `stage7_llm/` 实际对应“语义意图编码库”条件，历史目录名仍保留为 `stage7_llm`。
- `stage7_baseline_onehot/` 对应 one-hot 意图编码基线。
- 两组均包含 I-MAPPO 与 MAPPO 对比，seed 为 `7, 11, 23`。
- 当前论文写作时建议称为“语义意图编码库”或 “semantic intent library”，不建议继续把它作为真实 LLM 在线推理实验表述。

核心结论：

- 语义意图编码 I-MAPPO 的最终评估碰撞率约为 `0.289`。
- one-hot 意图编码 I-MAPPO 的最终评估碰撞率约为 `0.424`。
- 碰撞率相对下降约 `31.8%`。
- 任务完成率基本保持在 `0.75-0.79` 区间，没有明显牺牲任务效率。

### 2. VMAS Stage1 验证实验

路径：

- `experiments/vmas_stage1/`

说明：

- 场景为 VMAS `dispersion`。
- 该实验用于验证跨环境接入和方法适用边界。
- 当前结果不支持把 VMAS 作为主优势实验：I-MAPPO 与 MAPPO 在确定性评估中表现接近，且该任务缺少明显的安全/效率/协同意图权衡。

论文中建议将 VMAS 作为补充讨论或局限性分析，不作为主结果重点展开。

## 实验目录保留规则

后续新增实验时，`experiments/` 中只建议提交：

- `summary.json`
- 每个 seed 的 `result.json`
- 最终图表，如训练收敛图、碰撞率图、风险分档图
- 必要的意图库文件，如 `intent_library.json`
- 简短实验说明

不建议提交：

- `.pt` checkpoint
- `metrics.csv`
- `metrics.jsonl`
- smoke 测试输出
- 临时调参结果
- 大量中间图表

`.gitignore` 已加入 `experiments/**/*.pt`，防止训练 checkpoint 误入仓库。

## 常用运行方式

安装基础依赖：

```bash
pip install -r requirements.txt
```

安装环境相关依赖：

```bash
pip install -r env_requirements.txt
```

运行 UAV / I-MAPPO 实验脚本可从以下入口开始查看：

- `src/imappo_experiments.py`
- `src/imappo_vmas_experiments.py`
- `src/intent_llm.py`
- `src/imappo.py`
- `src/envs/uav_scheduling_env.py`

具体参数以脚本内 CLI 定义为准。新增正式实验前，建议先确定输出目录命名，避免再次混入 smoke 或临时结果。

## 当前论文建议口径

当前代码和实验更适合支撑以下表述：

- “意图编码驱动的多智能体强化学习”
- “语义意图编码库相较 one-hot 意图编码可降低碰撞率”
- “高层意图可由规则、专家知识库或大模型生成，本文重点研究意图表示如何注入底层 MARL 决策”

不建议把现有 Stage7 结果直接表述为“真实 LLM 在线推理闭环实验”，因为当前 `stage7_llm` 使用的是静态语义意图库和 hash embedding。

## 许可证

本仓库保留原 EPyMARL/PyMARL 相关许可证文件：

- `LICENSE`
- `NOTICE`
