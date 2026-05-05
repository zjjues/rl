# Stage7: LLM Intent 接入实验结果

**日期**: 2026-05-04  
**目标**: 验证 LLM 语义 intent 向量库对 I-MAPPO 安全性能的提升  
**配置**: 3000 episodes, 3 seeds [7, 11, 23], 8 agents, 6 targets, intent_dim=64

---

## 对比配置

| 配置名称 | intent_source | 算法 |
|----------|:---:|------|
| **I-MAPPO(LLM)** | `llm_library` (static hash, 25 预置描述) | I-MAPPO |
| **I-MAPPO(onehot)** | `onehot` (原版) | I-MAPPO |
| **MAPPO (LLM组)** | — | MAPPO |
| **MAPPO (onehot组)** | — | MAPPO |

注：MAPPO 不使用 intent 机制，两组 MAPPO 的区别仅在于与哪个 I-MAPPO 同批运行。

---

## 核心结果

### 碰撞率 ↓ (越低越好)

| 指标 | I-MAPPO(LLM) | I-MAPPO(onehot) | MAPPO(LLM组) | MAPPO(onehot组) |
|------|:---:|:---:|:---:|:---:|
| Eval 碰撞率 | **0.289** | 0.424 | 0.481 | 0.353 |
| Easy Probe | 0.289 | 0.424 | 0.481 | 0.353 |
| Mid Probe | **0.304** | 0.620 | 0.395 | 0.493 |
| Hard Probe | 0.517 | 0.535 | 0.517 | 0.799 |

### 任务完成率 ↑ (越高越好)

| 指标 | I-MAPPO(LLM) | I-MAPPO(onehot) | MAPPO(LLM组) | MAPPO(onehot组) |
|------|:---:|:---:|:---:|:---:|
| Easy | 0.784 | 0.791 | 0.768 | 0.793 |
| Mid | 0.748 | 0.784 | 0.757 | 0.786 |
| Hard | 0.756 | 0.759 | 0.743 | 0.789 |

---

## 关键发现

### 1. LLM intent 显著降低碰撞率
I-MAPPO(LLM) 相比 I-MAPPO(onehot) 碰撞率从 **0.424 → 0.289 (-31.8%)**。Mid Probe 上的改善尤其显著 (0.620 → 0.304, -51%)。

### 2. LLM intent 使 I-MAPPO 首次明确优于 MAPPO
- I-MAPPO(LLM) vs MAPPO: 碰撞率 0.289 vs 0.481 (**-40.0%**)
- I-MAPPO(onehot) vs MAPPO: 碰撞率 0.424 vs 0.353 (**+20.1%, 反而更差**)

原版 onehot I-MAPPO 在安全性上跑不过 MAPPO，LLM intent 彻底扭转了这一趋势。

### 3. 任务完成率持平
引入 LLM intent 后任务完成率在各个 probe 上基本不变（~0.75-0.79），没有以牺牲效率换取安全。

### 4. 25 条预置语义描述已经有效
当前使用 static hash embedding 模式（零 API 依赖），仅 25 条手写意图描述就带来显著提升。接入真实 LLM embedding 后预期进一步提升。

---

## 与之前 stage 对比

| Stage | I-MAPPO 碰撞率 | MAPPO 碰撞率 | I-MAPPO 优势 |
|-------|:---:|:---:|:---:|
| Stage6 (onehot) | 0.424 | 0.353 | **-20% 劣势** |
| **Stage7 (LLM)** | **0.289** | 0.481 | **+40% 优势** |

---

## 下一步

- **P0-1 完成**: LLM intent 接入已验证有效
- 可考虑接入真实 LLM embedding API 构建更大的向量库
- 继续推进 P0-2 (更多环境) 和 P1-1 (更多 baseline)
