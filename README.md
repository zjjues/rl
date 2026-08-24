# I-MAPPO 空基多智能体协同调度实验仓库

本仓库基于 EPyMARL/PyMARL 多智能体强化学习框架改造，当前主要用于论文实验中的 I-MAPPO 算法、UAV 调度环境、语义意图编码实验和少量 benchmark 验证。

原 EPyMARL 框架能力包括多智能体算法训练、环境封装、配置管理和结果绘图等。本仓库保留这些基础代码，但 README 不再展开上游框架的完整英文说明；如需查看原始框架用法，可参考 EPyMARL 官方仓库。

## 当前仓库内容

- `src/`：核心代码，包括 I-MAPPO、UAV 调度环境、语义意图库、VMAS 适配器和实验脚本。
- `configs/research/`：smoke、pilot、paper 研究的预注册配置。
- `docs/paper/`：论文计划、方法草稿、实验协议、贡献/结果台账和研究变更日志。
- `experiments/`：历史实验以及按证据等级隔离的新实验状态。
- `docker/`：原框架 Docker 相关文件。
- `requirements.txt`、`env_requirements.txt`、`pac_requirements.txt`：依赖文件。
- `run.sh`、`runalgo.sh`、`run_interactive.sh`：原有运行入口。
- `plot_results.py`：结果绘图脚本。
- `run_research_study.py`：不覆盖已有结果的正式研究运行入口。
- `audit_submission_readiness.py`：审计 11 项最终证据门槛，防止把 partial、calibration 或 superseded artifact 标成投稿完成。

已删除旧版 `reports/`、smoke 测试、中间日志、checkpoint 和过时英文报告，方便后续继续增加实验。

## 当前保留的实验状态

当前只保留与论文主线直接相关的实验结果：

### 1. Stage7 legacy hash 意图编码先导实验

路径：

- `experiments/stage7_semantic_library/`
- `experiments/stage7_baseline_onehot/`
- `experiments/STAGE7_REPORT.md`
- `experiments/stage7_results.json`

说明：

- `stage7_semantic_library/` 实际使用文本 SHA256→PRNG 的确定性 hash 码本。
- `stage7_baseline_onehot/` 对应 one-hot 意图编码基线。
- 两组均包含 I-MAPPO 与 MAPPO 对比，seed 为 `7, 11, 23`。
- 该实验只作为历史 pilot，用于生成新研究假设；不得表述为真实语义 embedding 或在线大模型推理实验。

历史观察：

- legacy hash I-MAPPO 的最终评估碰撞率约为 `0.289`。
- one-hot 意图编码 I-MAPPO 的最终评估碰撞率约为 `0.424`。
- 相对 one-hot 下降约 `31.8%`，但该差异不能归因于文本语义。
- 任务完成率基本保持在 `0.75-0.79` 区间，没有明显牺牲任务效率。

### 2. VMAS Stage1 验证实验

路径：

- `experiments/vmas_stage1/`

说明：

- 场景为 VMAS `dispersion`。
- 该实验用于验证跨环境接入和方法适用边界。
- 当前结果不支持把 VMAS 作为主优势实验：I-MAPPO 与 MAPPO 在确定性评估中表现接近，且该任务缺少明显的安全/效率/协同意图权衡。

论文中建议将 VMAS 作为补充讨论或局限性分析，不作为主结果重点展开。

## 新实验的证据等级与目录

新实验必须使用以下隔离路径：

- `experiments/smoke/{study_id}/`：代码路径验证；
- `experiments/pilot/{study_id}/`：调参与探索；
- `experiments/paper/{study_id}/`：预注册正式结果；
- `experiments/frozen/{study_id}/`：最终冻结投稿资产。

完整规则见 `docs/paper/EXPERIMENT_PROTOCOL.md`。`paper` 级运行默认要求干净 Git 工作区、至少 10 seeds、每 seed/风险档至少 100 个评估 episode，并拒绝覆盖已有研究目录。

## 实验目录保留规则

后续新增实验时，`experiments/` 中只建议提交：

- `summary.json`
- 每个 seed 的 `result.json`
- 最终图表，如训练收敛图、碰撞率图、风险分档图
- 必要的意图库文件，如 `intent_library.json`
- 简短实验说明
- `manifest.json`、`config.json`、`RESULT_CARD.md` 和 `checksums.sha256`

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

安装真实语义意图实验依赖：

```bash
pip install -r requirements-research.txt
```

先验证研究配置而不启动训练：

```bash
python run_research_study.py \
  --config configs/research/uav_intent_representation.smoke.json \
  --dry-run
```

正式配置在提交并冻结研究快照后运行：

```bash
python run_research_study.py \
  --config configs/research/uav_intent_representation.paper.json
```

运行 UAV / I-MAPPO 实验脚本可从以下入口开始查看：

- `src/imappo_experiments.py`
- `src/imappo_vmas_experiments.py`
- `src/intent_semantic_encoder.py`
- `src/imappo.py`
- `src/envs/uav_scheduling_env.py`
- `automated_closed_loop_tuner.py`

具体参数以脚本内 CLI 定义为准。新增正式实验前，建议先确定输出目录命名，避免再次混入 smoke 或临时结果。

## 当前论文口径

当前代码和实验更适合支撑以下表述：

- 可以说：“历史先导实验显示意图表示形式可能影响安全性，促使我们设计真实语义、随机稠密、legacy hash 与 one-hot 的受控比较。”
- 尚不能说：“语义理解已经显著降低碰撞率。”该结论必须等待预训练语义 embedding、未见意图泛化和严格统计实验完成。
- 高层意图可来自规则、专家、上层规划器或自然语言接口；论文主问题是具有语义结构的表示能否提高安全泛化与动态适应。

论文工作的唯一事实来源是 `docs/paper/`。任何重大代码、环境、奖励或实验协议变更都应同步写入 `docs/paper/RESEARCH_CHANGELOG.md`。

## PyBullet 跨动力学验证

高层控制器与安全层可在隔离环境中通过 Crazyflie 刚体/旋翼动力学验证：

```powershell
conda env create -f environment-rl-pybullet.yml
conda activate rl-pybullet
python run_pybullet_transfer_study.py --config configs/research/pybullet_transfer_lanes.pilot.json
```

该路径验证的是控制器/安全层跨动力学迁移，不等同于 SITL、HIL 或实机实验。结果与
限制记录在 `docs/paper/RESULTS_LEDGER.md`。

## 许可证

本仓库保留原 EPyMARL/PyMARL 相关许可证文件：

- `LICENSE`
- `NOTICE`
