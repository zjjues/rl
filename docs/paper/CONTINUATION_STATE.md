# 项目持续执行与恢复状态

> 最后更新：2026-08-24（Asia/Shanghai）
> 状态：**目标仍在进行。工程/协议约 95%，paper 证据约 48%–52%。CityNav 一次性终测已真实失败并冻结；VMAS 两场景架构 smoke 已完成；尚无 clean multi-seed 主结果，也没有独立人工偏好数据。**

## 0. 最新恢复点（优先于下文历史保存点）

- Git：`testv1`，远端跟踪 `origin/testv1`；当前 worktree 含大量未提交研究变更。paper run 必须等待 clean snapshot。
- Python：`D:\Programs\anaconda3\envs\rl-test\python.exe`；Torch 2.4.1+cu124；VMAS 1.5.2；RTX 3050 6GB。
- 当前活动目标未完成，不得标记 paper-ready。
- AerialVLN 开发 gate SHA-256 `8518d9be5f73ab83be87c160065a93d2c75678038f8223fef7705d52a9b787fd`，threshold `0.024408113173431428`，metadata 仍为 `final_blind_test=false`。
- CityNav v2 预注册与 one-shot 已完成：32,637 条，accepted=31,381，FAR=0.961516，Wilson 95% CI=[0.959374, 0.963549]，outcome=`fail`。结果位于 `docs/paper/audits/citynav_human_language_final_ood_v2.json`；`.attempt.json` 为 completed。**禁止删除、重跑或用 CityNav 修改 gate/threshold。**
- CityNav archive 位于 `.cache/citynav_final_ood/data.tar.gz`，190,078,685 bytes，SHA-256 `121d052e81a4d3f58fb9c6a45ceac3c616bb191dd5ae39cf5ac2dbd636af65bd`；缓存不上传 Git。
- VMAS navigation/dispersion 各 5 algorithms × 1 seed smoke complete 且 artifact valid，只有 dirty warning。paper 配置各 10 seeds/100 eval episodes，未启动。
- VMAS paper 粗预算：navigation 36.4–55.5 GPUh；dispersion 32.2–49.0 GPUh。两个 smoke 只证明管线，禁止算法排序。
- 正式人工数据入口：`freeze_preference_dataset.py` + `audit_formal_preference_dataset`；当前没有实际独立人类数据，因此语言主张被阻断。
- 可恢复实验仍支持 `--only-variants`、`--only-seeds`、`--resume` 和 partial manifest；正式长运行前先做 100-episode calibration。
- 没有已知活跃后台实验进程。
- 最新全量回归：**164 passed, 14 warnings, 8.71 s**；48 个 config/audit JSON 全部可解析。warnings 仍为可选 PettingZoo 与 Matplotlib/PyParsing deprecation。
- `git diff --check` 在移除本文件 Markdown 行尾空格后通过；CRLF 转换提示不属于 whitespace error。

### 下一步严格顺序

1. 全量 pytest、`git diff --check`、所有新增 JSON 解析与 artifact audit；把精确结果回填本文件。
2. 人工 preference 招募/consent/独立复核（外部协调硬阻塞）；冻结 JSONL 与 test hash 后才能开发 gate v2。
3. 形成 clean commit，重新生成配置/源哈希，重跑六轴 UAV smoke，替换旧七轴 superseded 证据。
4. 执行 100-training-episode calibration，再分块跑 UAV 消融、UAV 架构、VMAS navigation/dispersion paper runs。
5. 官方 HARL 数值交叉核验；多机冲突 policy-in-loop SITL；HIL/受控实机。

### 新增关键文件

- `configs/data/citynav_final_ood_registration.v1.json`：path-only 检查后、文本访问前透明作废。
- `configs/data/citynav_final_ood_registration.v2.json`：实际冻结注册。
- `src/final_ood_registration.py`、`evaluate_final_language_ood.py`：注册验证与 one-shot evaluator。
- `docs/paper/audits/citynav_human_language_final_ood_v2.json{,.attempt.json}`：最终负结果与执行状态。
- `configs/research/vmas_{navigation,dispersion}_architecture_v1.{smoke,paper}.json`。
- `experiments/smoke/vmas_{navigation,dispersion}_architecture_v1_smoke` 及对应 audit/runtime plan。
- `freeze_preference_dataset.py`：未来人工数据冻结入口。

## A. 2026-08-20 历史保存点（被上面的最新状态覆盖）

### A.1 当时精确状态

- Git 分支：`testv1`；远端基准提交 `5097665df1fdeb0a784abb21fb1dfaa8abd27f2b`。
- 本轮大量新增/修改尚未提交，worktree dirty；不得在此状态启动 paper run。
- Conda Python：`D:\Programs\anaconda3\envs\rl-test\python.exe`。
- PyTorch 2.4.1+cu124；GPU NVIDIA RTX 3050 6GB Laptop。
- 最新全量回归：**140 passed, 14 warnings, 9.70 s**。warnings 为可选 PettingZoo 和 Matplotlib/PyParsing deprecation。
- `git diff --check` 通过，仅有 CRLF 提示。
- 无活跃后台进程。
- `.cache/aerialvln-v8.zip` 及解压/派生文件为 Git ignored 本地缓存，不会上传。

## 1. 已完成且需保留的工程变更

### 1.1 CBF 等价加速

- `src/rule_based_baseline.py` 将 8-UAV/28-pair cyclic projection 与 diagnostics 融合为 host float32 顺序事务，保持 lexicographic Gauss--Seidel、4 iterations、active-set 和每次全局 clipping。
- CUDA 54.798 → 0.827 ms（66.237×）；CPU 7.615 → 0.756 ms（10.076×）。最大动作/诊断误差 `4.25e-7`/`2.03e-8`。
- 证据：`benchmark_cbf_runtime.py`、`CBF_RUNTIME_OPTIMIZATION.md`、`audits/cbf_runtime_benchmark_{cuda,cpu}.json`。

### 1.2 完整旧合同 smoke（现已 superseded）

- 消融：`experiments/smoke/uav_imappo_ablation_smoke_v2`，10 variants × 1 seed，artifact valid，dirty warning。
- 架构 v3：`experiments/smoke/uav_marl_architecture_v3_smoke`，I-MAPPO/no-mask/MAPPO/IPPO/HAPPO/MATD3 × 1 seed，artifact valid，dirty warning。
- HAPPO 为 8 个独立 actor/optimizer、随机顺序、前序 likelihood-ratio product factor、centralized MLP critic；官方协议参照 HARL commit `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`。
- 两个 smoke 只证明各自旧 snapshot 管线连通。由于 1.5 的六轴安全合同迁移，均已 superseded，禁止继续作新协议证据。

### 1.3 原子分块与预算

- `run_research_study.py` 新增 `--only-variants`/`--only-seeds`。未齐全时 manifest=`partial` 并列出 missing pairs，不生成 summary/显著性；全部注册 pairs 存在后重新载入全集再聚合。
- `plan_experiment_runtime.py` 从 smoke wall time 和 environment-step workload 估计并生成 resume chunks，不允许缩减 protocol。
- 架构 v3：60 runs，粗估 **60.9–105.2 GPU-hours**。
- 5-seed 消融：50 runs，粗估 **82.2–105.0 GPU-hours**；此前约 52 小时估计作废。
- 范围为高不确定性；必须先做 100-training-episode calibration。
- 审计：`audits/uav_marl_architecture_v3_runtime_plan.json`、`audits/uav_imappo_ablation_runtime_plan.json`。

### 1.4 外部语言来源与人工标注合同

- AerialVLN Kaggle v8 固定：API metadata 393,931,815 bytes；实际 ZIP 66,720,431 bytes；archive SHA-256 `d8f5d47f7409f3d254b90063e1f0eb693cc581a8af828fc37375cb055c1932d2`；CC BY 4.0。
- `val_unseen` 导出 2,310 条唯一导航指令；JSONL SHA-256 `758761ecf82d5908edf4f34577bfa541381ad8a6c7e6202940ad8af0dc0c1119`。
- 外部 importer 不产生 objective/polarity；manifest 机器拒绝把导航指令声明为 preference supervision。
- 人类 preference schema 支持独立 reviewer、第三方 adjudicator、批次/prompt/语言/consent、writer split 隔离和裁决前 raw agreement/Cohen's kappa。
- 文件：`src/external_language_corpus.py`、`audit_external_language_corpus.py`、`configs/data/*`、`EXTERNAL_LANGUAGE_DATA_AUDIT.md`。

### 1.5 六轴语言安全合同修正

- 审计发现旧实现虽从 v8 查询移除 collision，decoder、canonical reward profiles 和环境 override 仍把 collision 当第七偏好轴。
- 现固定六个 preference axes：distance、energy、safety、task、time、threat；collision 独立为 non-negotiable safety axis。
- 所有 canonical label 的 collision reward weight 固定 1.0；环境拒绝 `collision != 1.0` profile；adapter/NLI/prototype/diagnostics 均为六维。
- 这只保证语言不能降低 collision penalty，不等于形式化零碰撞；仍需 CBF/备用控制和真实系统验证。
- v2/v7 源 suite 已直接删除 collision counterfactual；v8 加载后 30 queries，contrast groups 恰为六轴。
- 文档：`LANGUAGE_SAFETY_CONTRACT.md`。

### 1.6 AerialVLN OOD smoke（重要负结果）

- 128 条确定性抽样，sample ID SHA-256 `1b8e8efb75ff6346fcfb0bbbaf4e0cd0e243572f00dad13b1c11605ee4290691`。
- 六轴 NLI prototype decoder 最大偏移 median/p95/max = 0.0829/0.4972/0.4995。
- 未校准阈值 0.05/0.10/0.20 的激活率 = 58.59%/46.88%/39.06%；100/128 最大偏移为 `distance:low`。
- 结论：当前 decoder 对纯导航语言过度激活，没有可接受的拒答证据。阈值不得从此 smoke 选择，需独立 preference dev 校准后在完整冻结 OOD 集评估。
- 审计：`audits/aerialvln_v8_source_manifest.json`、`audits/aerialvln_v8_val_unseen_ood_smoke.json`。

## 2. 仍不可引用或已失效的证据

- `experiments/pilot/uav_imappo_main`：所谓 MAPPO 实际执行 I-MAPPO，未实现 concat 落入 attention；semantic audit invalid。
- 旧 1-seed smoke：单 seed/3 eval episodes，不能排序、推断效果或等效性；且现被六轴合同迁移 supersede。
- AerialVLN OOD smoke：无 preference ground truth，只能暴露误激活风险，不能报告 preference accuracy 或选阈值。
- runtime plan：只是资源规划，不是运行证据。

## 3. 下一步严格顺序

1. 复核全部 diff，形成 clean commit；重新生成 source/config hashes。
2. 在 clean commit 上重跑六轴 v3 smoke 与 10-variant ablation smoke，替换 superseded artifact 并消除 dirty warning。
3. 建立独立人类 preference train/dev/test；至少 13 类×50 条、每 split ≥5 writers，先冻结 test hash；用 dev 校准拒答/置信阈值。
4. 在完整 AerialVLN `val_unseen` 上报告冻结 threshold 的 OOD false-accept rate，并做失败类型分析；不能用 OOD test 反调阈值。
5. 运行 100-training-episode calibration，更新两份 runtime plan，再按 `partial`/`resume` 分块执行 5-seed 消融。
6. clean 10-seed 架构 v3 paper；另用官方 HARL 对至少一个任务/seed 做数值轨迹交叉核验。
7. 补跨场景语言证据、多机冲突 SITL policy-in-loop、HIL/受控实机。后三项仍是顶刊真实性硬门槛。

## 4. 恢复命令

```powershell
# 全量回归
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' -m pytest -q

# 验证外部来源 manifest（本地缓存存在时）
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' audit_external_language_corpus.py `
  --input .cache\aerialvln-v8\val_unseen.json --source-split val_unseen `
  --source-version kaggle-v8 --output .cache\aerialvln-v8\val_unseen.instructions.jsonl

# 重现 128 条 OOD smoke
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' evaluate_external_language_ood.py `
  --records .cache\aerialvln-v8\val_unseen.instructions.jsonl `
  --config configs\research\uav_imappo_ablation.smoke.json `
  --source-archive .cache\aerialvln-v8.zip --max-records 128 `
  --sample-seed 20260820 --device cuda `
  --output docs\paper\audits\aerialvln_v8_val_unseen_ood_smoke.json

# 重建预算计划
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' plan_experiment_runtime.py `
  --paper-config configs\research\uav_marl_architecture_v3.paper.json `
  --smoke-config configs\research\uav_marl_architecture_v3.smoke.json `
  --smoke-results experiments\smoke\uav_marl_architecture_v3_smoke `
  --output docs\paper\audits\uav_marl_architecture_v3_runtime_plan.json

# 示例：paper 配置中仅选择 HAPPO 两个 seed（clean commit 后才能真跑）
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' run_research_study.py `
  --config configs\research\uav_marl_architecture_v3.paper.json `
  --only-variants happo --only-seeds 7,11
```

## 5. 顶刊距离的诚实判断

工程与协议准备度约 **94%**；论文证据准备度仍约 **55%–58%**。本阶段提升来自更严格的构念边界、真实外部语言负对照、暴露拒答失败和可靠恢复，而不是新增显著主结果。六轴合同修正反而要求重跑旧 smoke。clean multi-seed 主实验、独立人类偏好数据、冻结 OOD 拒答、官方 HARL 交叉复现、跨场景、冲突 SITL、HIL/实机仍未完成，目标不得标记 paper-ready。
