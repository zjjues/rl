# I-MAPPO 实验计划与结果（历史架构结果已撤回解释）

> 最后更新：2026-08-20
> 实验分支：testv1
> 证据等级：历史文件完整但方法协议无效；不得作为算法比较结果

**撤回说明：** 2026-08-20 审计确认 `mappo` key 实际注册 `algorithm="imappo"`，且 `critic_mode="concat"` 没有实现、旧代码执行为 attention。下述 40-run 数值保留用于 provenance 和失败分析，但所有“标准 MAPPO”“attention-vs-concat”和强基线结论撤回。最新审计见 `docs/paper/audits/uav_imappo_main_semantic_protocol_audit.json`。

## 1. 实验概述

### 研究问题
历史注册问题是 I-MAPPO 相对 MAPPO/MATD3/IPPO 是否有优势；由于 MAPPO treatment identity 失败，该问题未被本实验回答。

### 实验环境
- **环境**：uav-scheduling-v2，8 架无人机 + 6 个移动目标
- **训练**：3000 episodes，200 steps/episode，rollout=128，batch=64
- **设备**：NVIDIA RTX 3050 6GB GPU，CUDA 12.5，PyTorch 2.4.1+cu124
- **统计**：10 seeds，bootstrap 95% CI，IQM

### 变体设计

| 变体 | 算法 | Critic | Action Mask | 说明 |
|------|------|--------|-------------|------|
| **imappo** | MAPPO | attention | ✅ | 意图作为 Query 的交叉注意力 Critic |
| **mappo** | 实际 I-MAPPO | 请求 concat、实际 attention | ❌ | 标签错误，非标准 MAPPO |
| **matd3** | MATD3 | centralized twin critics | ❌ | concat 配置字段被算法忽略 |
| **ippo** | IPPO | local | ❌ | concat 配置在构建时被覆盖 |

所有变体使用 onehot 意图编码（25 维），环境配置一致。

### 评估维度
- 3 个风险等级：easy（稀疏）、medium（中等）、hard（密集）
- 50 evaluation episodes per seed per tier
- 指标：collision_rate（碰撞率）、task_completion（任务完成率）、episode_return

---

## 2. 历史输出（40 runs；禁止作算法效应解释）

### 碰撞率（↓ 越低越好）

| 变体 | Easy | Medium | Hard |
|------|------|--------|------|
| **mappo** | **0.113** | **0.178** | **0.422** |
| imappo | 0.119 | 0.209 | 0.548 |
| ippo | 0.303 | 0.406 | 0.625 |
| matd3 | 0.378 | 0.469 | 0.564 |

### 任务完成率（↑ 越高越好）

| 变体 | Easy | Medium | Hard |
|------|------|--------|------|
| **ippo** | **0.648** | **0.653** | **0.656** |
| matd3 | 0.581 | 0.588 | 0.586 |
| imappo | 0.551 | 0.556 | 0.554 |
| mappo | 0.548 | 0.553 | 0.552 |

### 已撤回的旧解释

Holm p 值和 bootstrap CI 数学上仍对应这些文件，但 comparison label 错误，不能修复 treatment identity。修正协议位于 `configs/research/uav_marl_architecture_v2.paper.json`，必须重新训练。

---

## 3. SITL 验证（补充实验）

### 单机闭环

`betaflight_sitl_closed_loop_smoke_v19`（seed=7，WSL2 Betaflight × Windows PyBullet）：

| 指标 | 值 | 阈值 | 状态 |
|------|-----|------|------|
| 电机包接收率 | 99.03% | ≥80% | ✅ |
| 电机输出 max | 1.0 | >0.05 | ✅ |
| 电机输出 mean（解锁后） | 0.374 | — | — |
| 高度响应 | +0.15m | >0.10m | ✅ |

多 seed 验证（5 seeds）：5/5 通过，高度增益 mean=0.134m, std=0.024m。

### 双机闭环

`betaflight_sitl_multi_closed_loop_smoke_v2`（seed=7，双机，端口偏移）：

| 无人机 | 高度响应 | 电机包率 | 电机 max | 状态 |
|--------|----------|----------|-----------|------|
| Drone 0 | +0.21m | 99.60% | 1.0 | ✅ |
| Drone 1 | +0.12m | 99.78% | 1.0 | ✅ |

---

## 4. 文件结构

```
experiments/
├── pilot/
│   └── uav_imappo_main/          # 主实验（40 seeds）
│       ├── imappo/               # 10 seeds
│       ├── mappo/                # 10 seeds
│       ├── matd3/                # 10 seeds
│       └── ippo/                 # 10 seeds
└── smoke/
    ├── betaflight_sitl_closed_loop_smoke_v19/    # 单机 SITL
    ├── betaflight_sitl_closed_loop_smoke_v19_multiseed/  # 5-seed
    ├── betaflight_sitl_multi_closed_loop_smoke_v2/       # 双机 SITL
    └── uav_imappo_paper_smoke/                   # I-MAPPO smoke

configs/research/
├── uav_imappo_main.paper.json    # 主实验配置
├── uav_imappo_ablation.pilot.json # 消融实验配置（待执行）
├── uav_imappo_paper.smoke.json   # Smoke 测试配置
├── betaflight_sitl.smoke.v19.json # SITL 单机配置
└── betaflight_sitl.multi_smoke.v2.json # SITL 双机配置

run_*.py
├── run_research_study.py         # MARL 实验执行器
├── run_betaflight_sitl_smoke.py  # 单机 SITL runner
└── run_betaflight_sitl_multi_smoke.py # 多机 SITL runner
```

---

## 5. 待执行

| 优先级 | 实验 | 配置 | 状态 |
|--------|------|------|------|
| P0 | 消融实验 | `uav_imappo_ablation.pilot.json` | ⏳ 待执行 |
| P1 | 论文图表生成 | `docs/paper/generated/uav_imappo_main/` | ✅ pilot 版完成 |
| P1 | SITL + I-MAPPO 策略部署 | 训练好的策略→SITL | ⏳ |
| P2 | 超参数敏感性 | lr, entropy, rollout 扫描 | ⏳ |
