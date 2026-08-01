# 项目持续执行与恢复状态

> 最后更新：2026-08-01 23:28（Asia/Shanghai）  
> 状态：**应用户要求暂停**；研究目标未完成，尚未达到可诚实宣称“顶刊投稿就绪”的门槛。  
> 用途：任务中断、上下文压缩或更换执行会话后，以本文件作为第一恢复入口。

## 0. 暂停点

- 当前没有 Betaflight SITL 或实验 runner 在后台运行；25 秒临时诊断实例已自动退出。
- 当前 Git 分支：`testv1`；该分支用于保存本次全部工作区内容并推送到 `zjjues/rl`。
- 最后完成的动作：v3 单机 Betaflight SITL × PyBullet 闭环 smoke 运行结束并写入带校验和的失败实验目录。
- 最后确认的问题：UDP 双向通信已经成立，但 Betaflight 没有解锁，收到的所有电机值均为 0。
- 尚未执行：arming disable flags/RC 模式诊断、v4 配置与运行、合并后的完整 pytest、论文方法/结果台账中的 SITL 段落更新。
- 恢复任务时，从第 4 节“恢复后的立即操作”继续；不要重跑或覆盖 v1–v3。

## 1. 当前结论

- Git 提交基线与远端 `zjjues/rl` 的 `origin/main` 一致，二者均为 `f0b0d8186e648d8971e785874c1681407d17b6ae`。
- 工作区并不与远端内容完全一致：存在大量未提交的研究代码、配置、测试、论文文档、实验产物，以及原工作区已有的删除/修改。所有这些内容都必须保留，禁止用 `git reset --hard`、`git checkout --` 或清理命令覆盖。
- 工程成熟度的保守判断：工程实现与可复现基础约 85%；足以支撑顶刊主张的证据约 50%–55%。该百分比仅用于项目管理，不是审稿结论。
- 已形成可运行的语义意图、多智能体强化学习、规则/优化安全基线、PyBullet 高保真迁移和统计分析骨架；仍缺独立人类偏好数据、冻结的大样本主实验、多实例飞控闭环、HIL/实机验证和最终论文级审计。
- 当前不可使用“顶刊要求已经达到”“论文结果已冻结”等表述。

## 2. 不可丢失的环境与版本

### 主研究环境

- Python：`D:\Programs\anaconda3\envs\rl-test\python.exe`
- Python 版本：3.10.20
- NumPy：1.26.4
- PyTorch：CPU 构建
- 2026-08-01 暂停快照完整测试：86 passed，1 个 PettingZoo 可选依赖 warning，耗时 4.48 s。
- 2026-08-01 暂停快照 `pip check`：`No broken requirements found.`

### PyBullet 高保真环境

- Python：`D:\Programs\anaconda3\envs\rl-pybullet\python.exe`
- Python 版本：3.10.20
- 环境清单：`environment-rl-pybullet.yml`
- `gym-pybullet-drones`：2.1.0，来源提交 `e712698a05a80728b06572819dcf044596707754`
- `pybullet` Python distribution：3.2.5；conda 包元数据显示为 3.25。
- NumPy：2.2.6（OpenBLAS）。该选择规避了本机 conda MKL NumPy 在 `numpy.linalg.inv` 上挂起的问题。
- 隔离环境为最小运行环境，未安装未使用的 Stable-Baselines3/pytest，因此上游包元数据的 `pip check` 警告不能与主环境健康状态混淆。

### Betaflight SITL / WSL2

- WSL：Ubuntu 26.04 LTS，用户 `zhaji`
- 仓库：`/home/zhaji/rl-sitl/betaflight`
- Betaflight：`b41431ae981ced5086a63a89e8217fe6da02df33`
- config 子模块：`749fff19942fd7b44fa8020a086e1b566054cae9`
- libcanard 子模块：`601ed35467e0ac38819df17cd7c918de19f62d58`
- 构建产物：`/home/zhaji/rl-sitl/betaflight/obj/main/betaflight_SITL.elf`
- 官方构建命令：`make TARGET=SITL -j2`
- 当前飞控 EEPROM：来自 `gym-pybullet-drones` 的 Betaflight 资产，SHA-256 为 `006572e58a7cb196698016d1834e1c37a3819362143836e9960f406accfd2f49`。
- 自动生成的默认 EEPROM 备份：`/home/zhaji/rl-sitl/betaflight/eeprom.generated-default.bin`，SHA-256 为 `5e9d4df63deb85205c34dab16db6af14d1b113ddbaeab6af6077efd638f08a28`。
- WSL2 地址是动态的；恢复时必须重新解析 guest IP 与 Windows host gateway，不能硬编码当前地址。
- 官方参考：<https://betaflight.com/docs/development/SITL>、<https://betaflight.com/docs/development/building/Building-in-Ubuntu>。

## 3. 已完成的关键研究增量

### 语义意图与偏好解码

- 多种语义解码器与固定 MiniLM/NLI revision。
- `src/imappo.py` 中实现同观测反事实评估，避免把环境变化误认为语义可控性。
- v8 协商偏好覆盖 distance、energy、safety、task、time、threat；collision 被定义为不可放松硬约束。
- 已实现独立人类 JSONL 标注协议与数据管线：`src/preference_dataset.py`、`train_preference_decoder.py`、`docs/paper/PREFERENCE_ANNOTATION_PROTOCOL.md`。
- 硬缺口：尚未获得独立受试者的人类偏好数据，不能由合成标签替代外部有效性证据。

### 安全控制与低保真证据

- `src/rule_based_baseline.py` 包含 cyclic 与 QP-CBF，并在 study runner 中记录求解器和约束审计。
- `experiments/pilot/uav_qp_cbf_pilot_v1`（5 seeds）：QP 对抗碰撞率 0.0144，none 为 0.0184；优化耗时约 1 ms。

### PyBullet 高保真迁移

- 核心实现：`src/pybullet_transfer.py`、`run_pybullet_transfer_study.py`、`tests/test_pybullet_transfer.py`。
- 初始高保真 smoke 暴露对称死锁；随后加入与场景几何无关的 agent-index 高度通道。
- `experiments/pilot/pybullet_transfer_lanes_pilot_v1`（5 seeds）：QP 成功率 0.9917、碰撞步率 0、最终 RMSE 0.0425 m、求解时间 0.4335 ms。
- `experiments/pilot/pybullet_robust_qp_pilot_v1`（5 seeds，含 ground/drag/downwash 与 80 ms 延迟安全裕量）：最小间距 0.258493 m、碰撞率 0、安全违规率 0.000556、目标成功率 0.916667、RMSE 0.08153 m、求解成功率 1、最大约束违规 0、耗时 0.4117 ms。
- robust QP 相对基础 QP：最小间距 +0.011734（95% CI [0.010580, 0.013284]）、安全违规 -0.003778（[-0.006111, -0.001444]）、成功率 +0.050000（[0.008333, 0.091667]）。
- 两个 pilot 的校验和已验证；但 5 seeds 仍是先导证据，不是最终冻结论文实验。

## 4. 当前正在处理：Betaflight SITL 闭环

### 已完成

- WSL 安装 `build-essential`、clang-18、gcc/g++、make、curl、git、`python-is-python3`。
- 官方 Betaflight SITL 已成功编译并完成最小启动验收：TCP 5761、UDP 9003/9004 监听，输出端口为 9002/9001。
- 新增 `src/betaflight_sitl_bridge.py`：状态包 `@18d`、RC 包 `@d16H`、电机包 `@4f`、四元数旋转、CTBR→RC、电机推力→RPM、UDP bridge、审计与 WSL 网络解析。
- 新增 4 个 bridge 单元测试，定向运行已通过。
- 新增 `run_betaflight_sitl_smoke.py` 与 smoke 配置。

### 必须保留的失败历史

1. `experiments/smoke/betaflight_sitl_closed_loop_smoke_v1`
   - 失败原因：当前 Betaflight 不再接受位置式 IP 参数，必须使用 `--ip <address>`。
   - runner 已修复为显式 `--ip`。
2. `experiments/smoke/betaflight_sitl_closed_loop_smoke_v2`
   - 失败原因：`gym-pybullet-drones` 枚举名是 `DroneModel.RACE`，runner 错写为 `DroneModel.RACER`。
   - 已在 runner 中修复为 `DroneModel.RACE`，并使用新的 v3 study ID 重跑；v2 保持原样。
3. `experiments/smoke/betaflight_sitl_closed_loop_smoke_v3`
   - 网络/协议验收通过：发送 3000 个状态包和 3000 个 RC 包，收到 2972 个有效电机包，接收率 0.990667，无无效电机包。
   - 控制验收失败：`motor_output_max=0`，飞控未解锁；高度由 0.099961 m 降至 0.013490 m，未产生爬升响应。
   - 飞控进程退出码 124 来自预设 `timeout`，不是崩溃证据；轨迹和数据均为有限值。
   - 该结果将故障边界从网络/包格式收窄到 RC 解锁序列、AUX 模式或 Betaflight arming disable flags。
   - 目录包含 config、manifest、result、result card、SITL stdout 和 `checksums.sha256`，必须保留。

### 恢复后的立即操作

1. 启动短时 Betaflight 诊断实例，通过 TCP 5761/MSP CLI 或等价只读接口读取 `status`、arming disable flags、AUX/BOXARM 映射和接收通道值；不要先猜测或改 EEPROM。
2. 对照上游 `BetaAviary` 的 RC 顺序 `[roll, pitch, throttle, yaw, aux1, ...]`。当前 bridge 已匹配该顺序，AUX1 在 1 秒后由 1000 切到 1500，因此重点核验当前 Betaflight revision 与导入 EEPROM 的模式配置兼容性。
3. 根据诊断结果做最小修复，并新建 `configs/research/betaflight_sitl.smoke.v4.json`；禁止覆盖 v1–v3。
4. v4 必须继续记录包接收率、非零电机输出、解锁状态（新增审计字段）、高度响应、进程日志和校验和。
5. 单机闭环通过后，再实现端口偏移、多 Betaflight 实例和多无人机隔离。
6. 运行主环境完整测试与 `pip check`，更新本文档、方法草稿、结果台账和研究变更日志。

## 5. 精确恢复命令

在 Windows PowerShell、工作目录 `D:\seu\p\rl`：

```powershell
git status --short --branch
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' -m pytest -q
& 'D:\Programs\anaconda3\envs\rl-test\python.exe' -m pip check
& 'D:\Programs\anaconda3\envs\rl-pybullet\python.exe' -c "from gym_pybullet_drones.utils.enums import DroneModel; print(list(DroneModel))"
```

SITL 构建/版本核验：

```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /home/zhaji/rl-sitl/betaflight && git status --short --branch && git rev-parse HEAD && test -x obj/main/betaflight_SITL.elf"
```

已完成的 v3 命令（仅用于溯源，禁止再次执行覆盖）：

```powershell
& 'D:\Programs\anaconda3\envs\rl-pybullet\python.exe' run_betaflight_sitl_smoke.py --config configs\research\betaflight_sitl.smoke.v3.json
```

恢复时先完成只读 arming 诊断，再创建并运行 v4。WSL 命令可能需要 Codex 的沙箱提升权限。

## 6. 后续论文级路线与硬门槛

1. **单机飞控闭环**：Betaflight SITL × PyBullet 稳定闭环，包时序、丢包、超时、控制饱和和飞控进程状态均可审计。
2. **多机闭环**：多实例端口隔离、多无人机冲突场景、安全层对实际飞控输出的改善，至少跨多个随机种子。
3. **语言外部有效性**：独立人类标注数据、盲测划分、标注者一致性、置信区间与错误类型分析。
4. **鲁棒与校准**：延迟、丢包、传感噪声、模型失配、风扰和物理参数扫掠；对安全概率或风险指标做校准分析。
5. **冻结论文实验**：预先固定配置与评价规则，关键比较至少 10 seeds，报告效应量、置信区间、多重比较策略和失败案例。
6. **复现资产**：环境锁定、版本清单、单命令入口、结果校验和、表图生成脚本、原始与汇总数据分离。
7. **HIL/实机证据**：至少完成 HIL；若论文声称真实部署能力，则必须补充受控实飞，记录安全审批和中止条件。
8. **投稿审计**：贡献—证据矩阵、相关工作、威胁与限制、统计审计、匿名复现包和论文全文一致性检查全部通过。

## 7. 每次大改动的记录规则

每个大改动完成后必须同步更新：

- 本文件：状态、版本、失败、下一命令；
- `docs/paper/RESEARCH_CHANGELOG.md`：为什么改、改了什么、对论文方法/主张的影响；
- `docs/paper/METHODS_DRAFT.md`：可进入论文的方法定义、假设、算法和实现细节；
- `docs/paper/RESULTS_LEDGER.md`：仅记录有完整配置、原始数据和统计依据的结果；
- `docs/paper/PUBLICATION_READINESS.md`：证据门槛与尚未满足项；
- 对应实验目录：冻结配置、stdout/stderr、原始逐 seed 数据、汇总、校验和和失败原因。

若发生中断，恢复顺序固定为：先读本文件，再看 `git status`，再验证环境/测试，最后从“当前正在处理”和“恢复后的立即操作”继续。不得从头重复已经完成的实验，也不得删除失败实验。
