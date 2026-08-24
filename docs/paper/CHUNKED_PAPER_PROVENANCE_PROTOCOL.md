# 分块正式实验的实现同一性与 partial artifact 协议

## 发现的问题

UAV 消融正式研究包含 100 个 `variant × seed` 单元，必须分块执行。首个 `imappo_full × seed 7` 单元从 clean commit `0447ffd` 成功完成，manifest 正确列出 1 completed/99 missing，未生成 summary，checkpoint 也在 result 原子落盘后清除。然而事后审计发现：旧 runner 只把 `implementation_sha256` 写入训练 checkpoint，完成后的 `result.json` 与 manifest 不持久化该值。分块间提交实验结果必然改变 Git commit；只记录 commit 字符串不足以机器证明这些 commit 的训练源码相同。

该首个单元保留在 `experiments/paper/uav_imappo_ablation_paper_v2_superseded_pre_result_fingerprint_20260824/`。其 wall/process CPU 为 1472.10/1319.58 s，hard collision/task 为 0.0042/0.599881，但不得进入论文统计。新 validator 对其给出预期 `invalid`：缺 result provenance、manifest implementation hash 与 registered-protocol hash。审计文件为 `docs/paper/audits/uav_imappo_ablation_first_paper_unit_superseded_audit.json`。

## 修正后的身份合同

`research_provenance.py` 使用有序、无空白的 canonical JSON 计算两类 SHA-256：

- study protocol hash：完整注册 spec；
- result protocol hash：完整 spec、精确 variant 字典和 seed。

runner 继续对自身和 `src/**/*.py` 的路径及字节计算 implementation SHA-256。manifest、每条 `run_history` 与每个 result 现在都持久化 implementation hash；每个 result 同时保存 study id、variant key、seed 和 result protocol hash。已有 result 被 `--resume` 复用前必须与当前四元身份逐字段相等。resume invocation 的 implementation hash 必须等于既有 manifest，否则在加载缓存结果或训练前硬拒绝。

Git commit 仍用于追踪实验数据提交历史，但训练实现同一性由内容寻址的 implementation SHA-256 判定。这样不同 chunk 可位于“仅新增结果/文档”的不同 commits，只要 runner 与研究源码字节不变；任何训练源码变化都会改变 hash 并阻止混合。

## 合法 partial artifact

`validate_research_artifact.py --allow-partial` 不把注册的缺失单元误报成文件损坏。它只在以下条件全部满足时返回 `valid_partial`：

1. manifest 明确为 `partial`；
2. `missing_training_runs` 与磁盘缺失的注册 pairs 完全一致，completed count 与已有 result 数一致；
3. 不存在 `summary.json`，因此未提前产生整体显著性或多重检验；
4. 所有已有 result 的 seed、variant、指标、资源、result protocol hash 和 implementation hash 有效；
5. checksums 覆盖当前目录全部文件且无篡改；
6. paper invocation 均为 clean worktree，run history 不混合 implementation hashes。

默认不传 `--allow-partial` 时仍要求完整 artifact，防止投稿阶段误把 partial 当成最终结果。单元完成不提升论文证据等级；只有 100/100、final summary 与完整 artifact audit 为 `valid` 后才能进行注册的 18 假设推断。

## 修复后重跑验证

从 clean commit `317204a` 重跑 `imappo_full × seed 7` 后，partial audit 返回 `valid_partial`：1/100 results、3 checksum entries、99 个声明 missing pairs、0 errors、0 warnings、无 summary、无 checkpoint。result 与 manifest 的 implementation SHA-256 均为 `20d4c3ef29595c8eec18d501078187f03823ae0d4106c15db66bb5a34af4c1ac`，result protocol SHA-256 为 `ddb4f154a4f0564d06f6e8029fe8f14fe7228d3f4c3e96209addb4c7b23a5129`。

新旧 run 的全部非计时 tier metrics 和 logs 逐 JSON 相等；唯一差异是 1 个 final 与 60 个周期 `safety_filter_solver_time_ms` 墙钟测量。修复后 wall/process CPU=1468.89/1314.50 s，hard collision/task/return=0.0042/0.599881/−18.578883。该复跑证明 provenance 写入没有改变确定性行为路径，但仍只是一枚 seed，禁止效果推断。

上述 1-run 状态随后在第一次跨 commit resume 时被第二项严格审计取代，不能继续作为活动 paper fragment。旧 `merge_resume_specs` 对完全相同的单一 `objective` 仍新增冗余的一元素 `objectives` 数组；训练字段和 implementation hash 未变，但完整 spec hash 因此改变。seed 11 数值运行成功，validator 仍正确拒绝其 result 与 manifest protocol hash。两 seed fragment 原样保留在 `experiments/paper/uav_imappo_ablation_paper_v2_superseded_resume_objective_drift_20260824/`，审计为预期 invalid。

修复后的 merge 对相同单目标 spec 是字典级幂等操作，不再注入派生字段；回归测试同时断言 merged spec 相等和 canonical study hash 相等。由于 runner 字节已改变，先前 seed 7/11 的 implementation hash 也不再等于新实现，二者必须从新 clean snapshot 重跑，不能迁移或手工改 hash。
