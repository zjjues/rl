# 外部语言数据来源与适用性审计

## 2026-08-20 CityNav 最终 OOD 更新

CityNav 被选为真正未查看文本的最终 navigation-language 负对照。官方 release 基于真实城市 3D 数据，包含 32,637 条参与者收集的自然语言目标描述与 human demonstration trajectories，dataset 为 CC BY 4.0。v2 预注册在文本访问前固定 official commit `372ecbd...7710`、190,078,685-byte archive SHA-256 `121d052e...65bd`、四个 canonical split、gate/threshold、总体 FAR 与 Wilson 95% CI。v1 broad glob 因 path-only 检查发现 difficulty 子集重复而在文本打开前作废，理由和替代关系保留在仓库。

唯一一次终测得到 31,381/32,637 被错误接受，FAR **96.15%**（95% CI **95.94%–96.35%**），远高于预注册 10% fail 边界。四个 split 均超过 95.6%，说明不是单一 split 异常。当前 gate 不具备跨数据集拒答能力，CityNav 已消耗且禁止调参；该失败也不能被 AerialVLN development 改善结果掩盖。

开发阶段的 AerialVLN gate 在 `val_seen` 校准 FAR 为 4.95%，在已查看的 `val_unseen` 为 11.95%；同一 128 条样本的 profile activation 从 ungated 58.59% 降至 gated 6.25%（0.05 偏移阈值）。这些只是域内开发诊断，CityNav 终测证明其不可迁移。

## 结论

截至 2026-08-20，没有找到可直接充当本工程六维 UAV 操作偏好监督的公开数据。现有公开 UAV 语言数据主要标注“去哪里、如何飞、看到了什么”，而本工程需要“距离、能量、安全间距、任务完成、时间、威胁暴露”的 low/high 操作偏好。将前者自动映射为后者会产生研究者标签和不可验证的构念漂移，因此明确禁止。

AerialVLN v8 被选为域内 OOD 负对照：它含由 AMT 工作者根据 UAV 参考轨迹视频撰写的自然指令，可检测偏好解码器是否会对纯导航命令过度自信。它不能用于训练六维偏好，也不能报告偏好分类准确率。Kaggle API 在 2026-08-20 返回版本 8、393,931,815 bytes、CC BY 4.0；实际下载压缩包为 66,720,431 bytes，SHA-256 `d8f5d47f...32d2`。`val_unseen` 导出 2,310 条唯一指令，JSONL SHA-256 `758761ec...1119`。完整数据不 vendored 到 Git 仓库。

128 条确定性抽样的 GPU smoke 表明当前 decoder 尚不具备可接受的 OOD 拒答证据：以 profile 最大绝对偏移作为未校准激活量，0.05/0.10/0.20 阈值下激活率为 58.59%/46.88%/39.06%，其中 100/128 的最大偏移为 `distance:low`。这些阈值是在结果中并列展示的诊断点，不能用于选择 operating threshold。结果支持“需要独立 dev 校准拒答”，不支持任一误接收率的正式估计。

## 机器合同

- `configs/data/external_language_sources.v1.json` 固定来源、版本、许可、适用性决定和拒绝理由。
- `src/external_language_corpus.py` 验证来源字段、官方 split、文本跨 split 泄漏和冻结哈希。
- AerialVLN importer 只产生 `usage=ood_negative_control` 的无标签记录。manifest 若把 `navigation_instruction_not_preference` 声明为偏好监督会直接失败。
- `audit_external_language_corpus.py` 可从官方 episode JSON 提取指令并输出 JSONL 与 SHA-256；不复制或推断目标标签。
- `aerialvln_v8_source_manifest.json` 固定原包与派生记录哈希；`aerialvln_v8_val_unseen_ood_smoke.json` 固定抽样 ID 哈希、模型 revision 与诊断结果。

## 正式人类偏好数据仍需完成

公开语料只能补 OOD 负对照，不能替代本任务的独立人类标注。正式数据必须按 writer 隔离 train/dev/test，模型和阈值冻结前不可查看 test；每条记录保留 elicitation 目标、盲审标签、同意/裁决状态、独立裁决者、批次、prompt、语言和 consent 版本。验证器可计算裁决前 raw agreement 与 Cohen’s κ。

建议规模仍为 13 类每类至少 50 条，并确保每个 split 至少 5 名 writer。论文需同时报告 macro-F1、逐类 recall、校准、拒答率，以及在 AerialVLN OOD 负对照上的误接收率。当前缺少招募、伦理/同意文本和实际冻结数据，因此语言证据 Gate 仍为未通过。

## 来源

- AerialVLN 官方论文与补充材料：ICCV 2023；补充材料展示 AMT 指令采集界面。
- AerialVLN 官方仓库与 Kaggle `shuboliu/aerialvln` v8。
- GeoText-1652 官方 ECCV 2024 仓库：图像 caption/bbox spatial relation，非偏好。
- ROSETTA 官方项目/代码：公开偏好示例与框架，但未提供与本六轴直接兼容的冻结完整标签集。
- FLIGHT 官方项目：UAV 任务与 reasoning 文本，VLM 标注，不是独立人类偏好。
