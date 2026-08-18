# 项目持续执行与恢复状态

> 最后更新：2026-08-18
> 状态：**研究继续进行。40-run 架构 pilot 已修复 provenance 并通过 artifact 审计；它不是 frozen paper result。SITL 单/双机闭环 smoke 已通过。**

## 0. 当前状态

- Git 分支：`testv1`
- 环境：`D:\Programs\anaconda3\envs\rl-test\python.exe`（PyTorch 2.4.1+cu124，CUDA 12.5）
- GPU：NVIDIA RTX 3050 6GB Laptop
- 实验入口：`run_research_study.py --config <config.json> --allow-dirty`
- 当前 artifact 审计：`uav_imappo_main` valid，4 variants × 10 seeds、40 results、44 checksums、0 errors；保留历史 dirty-worktree warning。
- 2026-08-18 完整测试：99 passed、1 个可选 PettingZoo warning、4.53 s；`pip check` 无损坏依赖。

### 实验文件结构

```
experiments/pilot/uav_imappo_main/   # 主实验，4 variants × 10 seeds
experiments/smoke/betaflight_sitl_closed_loop_smoke_v19*/  # SITL 单机
experiments/smoke/betaflight_sitl_multi_closed_loop_smoke_v2/ # SITL 双机

configs/research/uav_imappo_main.paper.json    # 主实验配置
configs/research/uav_imappo_ablation.pilot.json # 消融实验（待执行）
configs/research/betaflight_sitl.smoke.v19.json # SITL 单机
configs/research/betaflight_sitl.multi_smoke.v2.json # SITL 双机

run_research_study.py                 # MARL 实验执行器
run_betaflight_sitl_smoke.py          # 单机 SITL runner
run_betaflight_sitl_multi_smoke.py    # 多机 SITL runner
```

### SITL 源码修改（WSL `/home/zhaji/rl-sitl/betaflight`）

构建命令：`make TARGET=SITL -j$(nproc) OPTIONS=SITL_ATTITUDE_DIRECT`

| 文件 | 修改 |
|------|------|
| `src/main/rx/rx.c` | `frameStatusUdp` SIMULATOR 路径（RXLOSS 修复） |
| `src/main/flight/imu.c` | `isUpright()` SIMULATOR 分支（ANGLE 修复） |
| `src/platform/SIMULATOR/sitl.c` | 包含 `fc/runtime_config.h`，每帧 FDM `ENABLE_ARMING_FLAG(ARMED)` |
| `src/platform/SIMULATOR/sitl.c` | `--port-offset <n>` 参数 |
| `src/betaflight_sitl_bridge.py` | port offset + drone_id 支持 |

## 1. 主实验结果

8 机 6 目标，3000 episodes，10 seeds，所有变体使用 onehot 意图。

| 变体 | 碰撞率(easy) | 碰撞率(hard) | 任务完成(easy) | 任务完成(hard) |
|------|-------------|-------------|---------------|---------------|
| mappo | **0.113** | **0.422** | 0.548 | 0.552 |
| imappo | 0.119 | 0.548 | 0.551 | 0.554 |
| matd3 | 0.378 | 0.564 | 0.581 | 0.586 |
| ippo | 0.303 | 0.625 | **0.648** | **0.656** |

详细结果见 `docs/paper/EXPERIMENT_RESULTS.md`。

## 2. 待执行

| 优先级 | 任务 | 命令 |
|--------|------|------|
| P0 | 消融实验 | `run_research_study.py --config configs/research/uav_imappo_ablation.pilot.json --allow-dirty` |
| P1 | 论文图表 | pilot 自动产物已生成，paper/frozen 版待 clean run |
| P1 | I-MAPPO 策略部署到 SITL | 需要策略导出 |
| P2 | 重写正文.docx 实验部分 | 用真实数据替换占位符 |

## 3. 恢复命令

```powershell
# 环境验证
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' -c "import torch; print(torch.cuda.is_available())"

# 完整测试
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' -m pytest -q

# 主 pilot artifact 审计
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' validate_research_artifact.py --study-dir experiments\pilot\uav_imappo_main --config configs\research\uav_imappo_main.paper.json

# 重建 pilot 论文表图
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' generate_paper_artifacts.py --study-dir experiments\pilot\uav_imappo_main --config configs\research\uav_imappo_main.paper.json --output-dir docs\paper\generated\uav_imappo_main

# 运行消融实验
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' run_research_study.py --config configs/research/uav_imappo_ablation.pilot.json --allow-dirty

# 运行单机 SITL（rl-pybullet 环境）
& 'D:\Programs\anaconda3\envs\rl-pybullet\python.exe' run_betaflight_sitl_smoke.py --config configs/research/betaflight_sitl.smoke.v19.json

# SITL 构建核验
wsl.exe -d Ubuntu -- bash -lc "cd /home/zhaji/rl-sitl/betaflight && test -x obj/main/betaflight_SITL.elf && echo 'SITL binary OK'"
```

## 4. 2026-08-18 精确恢复点

- 已完成：resume 协议合并、manifest run history、缓存身份检查、artifact validator、精确 paired sign-flip、Holm 校正、自动 CSV/PNG/统计报告。
- 已恢复：`src/intent_geometry.py` 与泛化 suite v1/v2/v7/v8 最小协议链；不要恢复已清理的旧实验结果目录。
- 主 pilot 可支持的结论：I-MAPPO 对 MAPPO 的 collision/task 无 Holm 显著优势；对 IPPO/MATD3 展现安全—任务权衡；不支持语义意图优势。
- 紧接着执行：完整 pytest → 消融配置 dry-run/GPU 预检 → 5-seed `uav_imappo_ablation` → artifact 审计与消融图表。
- 顶刊硬缺口：clean-commit 10+ seed/100+ eval frozen 主实验、独立人类/外部语言数据、语义因果消融、策略部署到多机冲突 SITL、HIL/受控实机。
- 当前工作树包含约 988 个既有删除和多项未提交新增/修改；它们来自本轮之前的清理与研究，禁止批量恢复、reset 或覆盖。
