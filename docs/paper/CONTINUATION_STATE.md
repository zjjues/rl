# 项目暂停与恢复状态

> 最后更新：2026-08-20（Asia/Shanghai）  
> 状态：**已按用户要求暂停。当前快照准备提交至 `testv1`；不要把历史 `uav_imappo_main` 当作有效论文证据。**

## 0. 精确暂停点

- Git 分支：`testv1`
- 远端：`https://github.com/zjjues/rl.git`
- Conda 环境：`D:\Programs\anaconda3\envs\rl-test\python.exe`
- GPU：NVIDIA RTX 3050 6GB Laptop；PyTorch 2.4.1+cu124
- 活跃实验已终止，不再有后台进程写入工作区。
- `uav_imappo_ablation_smoke_v2` 在用户要求暂停时完成 **7/10** 个变体：
  `imappo_full`、`no_mask`、`no_attention`、`no_intent_reward`、`no_cbf`、`no_nli_gate`、`prior_only`。
- 尚未完成：`no_profile_prior`、`identity_oracle`、`no_intent`。
- 中断产物目录：`experiments/smoke/uav_imappo_ablation_smoke_v2/`；其 `manifest.json` 已明确标记为 `interrupted`，不得作为完整消融结果引用。

## 1. 本轮关键科学纠错

历史 40-run 目录 `experiments/pilot/uav_imappo_main/` 的文件和校验和虽然一致，但方法身份不成立：

- 名为 `mappo` 的变体实际记录了 `algorithm: imappo`。
- 原 `critic_mode: concat` 没有对应实现，实际落入 attention 路径。
- MATD3 未按配置中的 critic mode 执行；IPPO 的 local critic 曾被静默覆盖。

因此，该历史结果只能保留作 provenance 记录，不能支持 MAPPO 基线或 attention-vs-concat 论文结论。语义审计见：

- `docs/paper/audits/uav_imappo_main_semantic_protocol_audit.json`
- `docs/paper/generated/uav_imappo_main/INVALIDATION_NOTICE.md`

已新增机器可检查的协议约束：

- `src/research_protocol.py`：校验算法、critic 和保留变体名的一致性。
- `src/research_ablation.py`：校验有根、无环、单因素漂移和预注册主假设。
- `src/imappo.py`：加入真正的 `intent_source: none`；未知 critic mode 直接拒绝。
- `src/objective_semantic_adapter.py`：冻结文本模型进程级缓存及资源审计。
- `generate_ablation_artifacts.py`：生成消融均值、配对比较、资源表、图和报告。

## 2. 已完成的有效工程证据

- 修正版架构 smoke：`experiments/smoke/uav_marl_architecture_v2_smoke/`
- 对应审计：`docs/paper/audits/uav_marl_architecture_v2_smoke.json`
- 5 个方法路径已区分：I-MAPPO、I-MAPPO no-mask、真正 MAPPO、local-critic IPPO、centralized-twin MATD3。
- 该 smoke 仅用于验证流水线，单 seed/少量 episode 不能用于方法优劣推断。
- 最近一次完整测试：**111 passed**，另有 1 个可选 PettingZoo warning；新增生成器相关定向测试也通过。提交前仍应重新执行完整测试。

## 3. 运行时瓶颈

增强消融 smoke 的单变体观测耗时约为：

| 变体 | 墙钟时间（约） |
|---|---:|
| `imappo_full` | 116.5 s |
| `no_mask` | 97.4 s |
| `no_attention` | 95.5 s |
| `no_intent_reward` | 110.6 s |
| `no_nli_gate` | 104.9 s |
| `no_cbf` | 17.7 s |

差异表明当前主要瓶颈是逐步执行的 pairwise CBF，而不是已缓存的文本模型。**恢复后不要直接启动 50-run paper 配置**；先优化 CBF，做数值等价测试与基准，再估算实验预算。

## 4. 恢复顺序

1. 检出 `testv1` 并创建新的研究分支/干净提交点。
2. 执行完整测试并 dry-run 两份 paper 配置。
3. 优化 `src/rule_based_baseline.py` 的 pairwise CBF，新增随机张量数值等价测试和运行时基准。
4. 从头重跑完整 `uav_imappo_ablation_smoke_v2`，不要在已标记 interrupted 的目录上伪装续跑。
5. artifact 审计并生成消融报告，再决定 pilot 的 seeds/episodes/evaluation 预算。
6. 补齐 clean frozen 主实验、独立语言数据、跨场景泛化、策略在冲突 SITL 中部署，以及 HIL/受控实机证据。

## 5. 恢复命令

```powershell
# 环境验证
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 完整测试
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' -m pytest -q

# 协议 dry-run
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' run_research_study.py --config configs\research\uav_marl_architecture_v2.paper.json --dry-run --allow-dirty
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' run_research_study.py --config configs\research\uav_imappo_ablation.pilot.json --dry-run --allow-dirty

# CBF 优化和验证后，从头运行完整 smoke（先移动/另存当前 interrupted 目录）
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' run_research_study.py --config configs\research\uav_imappo_ablation.smoke.json --allow-dirty

# smoke 完成后审计并生成文档
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' validate_research_artifact.py --study-dir experiments\smoke\uav_imappo_ablation_smoke_v2 --config configs\research\uav_imappo_ablation.smoke.json --output docs\paper\audits\uav_imappo_ablation_smoke_v2.json
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' generate_ablation_artifacts.py --study-dir experiments\smoke\uav_imappo_ablation_smoke_v2 --config configs\research\uav_imappo_ablation.smoke.json --output-dir docs\paper\generated\uav_imappo_ablation_smoke_v2
```

## 6. 距离顶刊的诚实判断

当前工程基础、协议防错和可复现设施已经较完整，但投稿证据仍明显不足，暂不满足“顶刊就绪”。核心缺口是：干净冻结的多 seed 主结果、可信强基线和因果消融、外部语言/人工数据、跨任务泛化，以及真实控制链路中的 HIL/实机验证。暂停前的保守成熟度评估约为 **52%–55%（论文证据维度）**；不能用工程完成度替代论文证据强度。
