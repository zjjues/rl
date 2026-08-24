# 顶刊投稿准备度机器门槛

## 目的

`audit_submission_readiness.py` 将论文所需的最终证据转为版本化、可重复执行的二值门槛。它解决的是“什么证据齐全后才允许标记 submission-ready”，不是用主观百分比预测录用概率。

当前配置为 `configs/research/submission_readiness.v1.json`，审计输出为 `docs/paper/audits/submission_readiness_v1.json`。只有全部 critical gates 都为 `met`，总状态才会是 `ready`。

## 当前结论

2026-08-24 审计结果为：

- status=`not_ready`；
- critical gates=`12`；
- met=`0`；
- blocking=`12`。

`0/12` 表示尚无一项“最终完成门槛”被完整关闭，不表示工程没有进展，也不替代 `CURRENT_PROJECT_SUMMARY.md` 中约 48%–52% 的阶段性证据准备度估计。例如语义泛化正式配置、validator 和统计生成器已经完成，但尚无 60/60 artifact，因此 artifact/statistics 两个 final gates 仍为 `unmet`。

## 十二项 critical gates

| Gate | 通过所需证据 | 当前状态 |
|---|---|---|
| formal human preferences | 冻结 JSONL、SHA-256、13 类覆盖、writer-disjoint split、独立复核、consent | unmet：manifest 缺失 |
| blind language gate | 多来源开发；未访问偏好 test 与新 OOD final；冻结阈值；FAR/FRR 达标 | unmet：final v2 缺失 |
| semantic generalization | 10-seed seen/paraphrase/unseen paper artifact；60/60 results | unmet：配置已冻结，artifact 缺失 |
| UAV strong baselines | 架构 v3 60/60 results，checksum/provenance 全部 valid | unmet：paper artifact 缺失 |
| semantic generalization statistics | seed 内 query 聚合；10 paired seeds；12 exact sign-flip tests；Holm | unmet：等待完整 artifact |
| UAV causal ablation | 链式消融 100/100 results，完整统计与 checksum | unmet：0/100，仅有 episode-2050 checkpoint |
| VMAS navigation | 50/50 原生 return paper results | unmet：paper artifact 缺失 |
| VMAS dispersion | 50/50 原生 return paper results | unmet：paper artifact 缺失 |
| official HARL cross-check | 固定官方 commit，至少 3 seeds/30 update cases，logprob/factor/parameter 误差达标 | unmet：数值审计缺失 |
| multi-UAV conflict SITL | ≥4 agents、10 seeds、100 conflict episodes、≥4 扰动条件和 fallback 审计 | unmet：final SITL 缺失 |
| HIL or real flight | HIL/受控实机、系统辨识覆盖、审批、≥30 trials、零物理碰撞 | unmet：final evidence 缺失 |
| frozen submission bundle | 匿名、checksum valid、不含 partial/superseded、表图与失败案例齐全 | unmet：frozen bundle 缺失 |

## 防止“放一个文件就通过”

三类 gate 使用不同验证器：

1. `study_artifact` 调用完整 `validate_study_artifact`，要求 config、result、summary、manifest、checksums、resource audit、protocol/implementation provenance 全部一致，并检查完成 result 数量。
2. `formal_preference_dataset` 重新计算 records SHA-256，并从 JSONL 重新执行类别覆盖、writer-disjoint、文本无泄漏、独立 reviewer/adjudicator、每类和每 split writer 下限审计；嵌入 audit 必须与重算结果完全一致。
3. `json_contract` 同时检查枚举/布尔冻结字段、样本量下限、误差上限和证据文件 SHA-256。它不能替代人工审查，但会拒绝明显不完整或不达阈值的声明。

所有 evidence path 必须位于仓库根目录内；配置拒绝重复 gate key、未知 kind、缺少 kind-specific path，以及没有任何字段/阈值条件的“文件存在即通过”JSON gate。当前审计工具放在仓库工具层而非 `src/`，因此不会仅因增加论文审计代码而改变训练 implementation fingerprint；episode-2050 checkpoint 的 fingerprint 已复核仍为 `c4e55820…40573`。

## 使用方法

生成审计但允许 `not_ready`：

```powershell
D:\Programs\anaconda3\envs\rl-test\python.exe audit_submission_readiness.py --config configs\research\submission_readiness.v1.json --output docs\paper\audits\submission_readiness_v1.json
```

投稿打包或 CI 中强制全部通过，否则退出码为 2：

```powershell
D:\Programs\anaconda3\envs\rl-test\python.exe audit_submission_readiness.py --config configs\research\submission_readiness.v1.json --require-ready
```

不得通过删除 critical gate、把 critical 改为 false、降低已注册样本量/误差阈值，或创建没有原始证据的手写 JSON 来宣布 ready。任何 v2 配置都必须在访问对应 final evidence 前提交，并在论文变更日志解释改变原因。
