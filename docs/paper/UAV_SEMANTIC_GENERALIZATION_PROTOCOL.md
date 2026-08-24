# UAV 语义泛化正式协议

## 研究问题与证据边界

本协议检验：在训练标签固定且 held-out 文本不进入训练的条件下，冻结文本表示能否改善 paraphrase 和 unseen intent 下的 UAV 行为。研究对象是“文本表示到行为的迁移”，不是强算法架构比较；MAPPO/HAPPO/MATD3 由独立架构协议处理。

正式配置为 `configs/research/uav_intent_generalization.paper.json`，calibration 为同构的 `uav_intent_generalization.calibration.json`，suite 入口为 `uav_intent_generalization_suite.v8.json`。当前尚未运行 calibration 或 paper artifact；本文档冻结协议，不提供效果结论。

## 冻结查询集合

继承解析后的 v8 suite 恰有 30 条查询：seen 2 条、paraphrase 4 条、unseen 6 条、counterfactual 18 条。后者是六个 preference axes 各 low/mid/high 三条。

六个可协商偏好轴固定为 distance、energy、safety、task、time、threat；collision 不属于语言偏好空间，保持唯一不可放松安全约束。

行为评估只使用 12 条 seen/paraphrase/unseen 查询。18 条 counterfactual 只用于冻结表示和 profile prediction 的描述性诊断，不进入行为显著性家族。这样避免将少量固定查询当作独立重复样本，也避免把 counterfactual 调试观察升级为确认性证据。

## 六个表示条件

| Variant | 查询输入 | 论文角色 |
|---|---|---|
| objective-grounded semantic | 文本 MiniLM + 冻结 NLI prototype profile | treatment |
| pretrained semantic | 文本 MiniLM 几何 | confirmatory baseline |
| legacy hash | 查询文本的确定性 hash code | confirmatory non-semantic baseline |
| random-dense oracle | canonical label 对应随机稠密码 | descriptive identity oracle |
| identity oracle | canonical label one-hot | descriptive upper-bound anchor |
| no intent | 全零 intent | confirmatory no-conditioning baseline |

random-dense 和 one-hot 直接接收 canonical label identity，因此不是文本理解模型；它们不得进入“semantic beats baseline”的确认性 p-value 家族。

## 公平性与泄漏控制

六个 variant 均使用相同 64-D policy input、IMAPPO actor、attention critic、direct policy、neutral rule context、pairwise CBF、关闭 action mask、关闭 intent potential reward。variant 只能改变 intent source 和语义路径必需的 decoder。

关闭 action mask 避免 canonical posture 通过离散 mask 泄漏给策略；关闭 objective-profile rule prior 和 shaping 避免 oracle profile 直接驱动 controller。查询的 canonical posture/profile 仍用于定义环境任务和评价目标，但策略只能通过自己的 intent vector 区分任务。固定 CBF 是所有条件共享的不可协商安全层，不作为语义方法贡献。

训练只使用 suite 注册的 19 个 train labels。unseen labels 从训练库排除；paraphrase/unseen 文本不得与训练描述完全相同。协议 validator 还拒绝任一 variant 自行把 one-hot 维度缩到 25，确保 actor 输入维度一致。

## 训练与评价

- 6 variants × 10 paired seeds=`60` training results；
- 每个 result：3000 training episodes × 200 max steps；
- training monitor：每 100 episodes、20 episodes；
- hard-tier base final：100 episodes；
- 12 个 behavior query 分别在 hard tier 使用 100 episodes；
- checkpoint cadence：50 episodes；
- seed 是唯一确认性统计单位，查询先在 seed 内算术平均。

所有 variants 在同 seed/query/tier 使用相同 reset seed 公式。每个 result 还保存 energy、speed、distance、separation 和 threat 资源指标，用于构念审计；它们不是本协议的确认性效果指标。

## 预注册统计

确认性 family 包含：paraphrase/unseen 两个 splits，pretrained semantic/legacy hash/no intent 三个 baselines，以及 task completion/episode return 两个 metrics，总数为 `2 × 3 × 2 = 12` hypotheses。

每个 hypothesis 先对同一 seed 内的查询取均值，再使用 10 个 paired seeds 做 two-sided exact sign-flip test，并对 12 个 p-values 做 Holm FWER correction。报告 treatment-minus-baseline raw paired differences、mean/median、bootstrap 95% CI、win/tie rate、standardized effect、raw/adjusted p-value。

固定查询集合上的 retrieval、profile MAE/Spearman 以及两个 identity oracle 只作 descriptive diagnostics，不把 query 当作总体抽样单位，也不报告 query-level 显著性。

统计生成器为 `generate_generalization_statistics.py`。它要求完整 artifact 先通过 checksum/provenance validator；缺任一注册 seed、variant、query split 或 family member均拒绝生成结果。

## Calibration 与资源预算

calibration 使用同一 environment、intent、variants、queries 和 reporting contract，只缩减为 1 seed、100×100 training、20 final/query episodes。其目的仅是验证六条路径、失败率和 active CPU-time；不得用于选择 variant、改确认性 hypotheses 或报告效果排序。

calibration 完成后必须使用 `plan_experiment_runtime.py` 按训练、monitor、base final 和 12-query final 的真实 workload 生成 paper 预算。预算完成前不宣称 GPU-hours。

## 可执行检查

协议与同构性审计：

```powershell
D:\Programs\anaconda3\envs\rl-test\python.exe validate_generalization_protocol.py --paper-config configs\research\uav_intent_generalization.paper.json --calibration-config configs\research\uav_intent_generalization.calibration.json --output docs\paper\audits\uav_intent_generalization_protocol_v1.json
```

paper dry-run：

```powershell
D:\Programs\anaconda3\envs\rl-test\python.exe run_research_study.py --config configs\research\uav_intent_generalization.paper.json --dry-run
```

完整 60/60 artifact 产生后才可运行统计：

```powershell
D:\Programs\anaconda3\envs\rl-test\python.exe generate_generalization_statistics.py --study-dir experiments\paper\uav_intent_generalization_paper_v1 --config configs\research\uav_intent_generalization.paper.json --output docs\paper\audits\uav_intent_generalization_statistics_v1.json
```

## 当前冻结哈希

- paper config SHA-256=`cebbabacab763145257034bfa60c2a85dc1fb9469dfa8c18441a33f43779e297`；
- calibration config SHA-256=`2cb6727c239522a80181daa9fe9895f3c9e962288da41b54f77dee207cee9bf5`；
- resolved suite SHA-256=`70e8517322c75cb7d025b0d62d1158f475f460969bbea89873d65c1464d5ffb7`。

任何字段修改都会产生新协议哈希；在查看未来 calibration 输出后，不得静默改写同一 v1 paper hypotheses。
