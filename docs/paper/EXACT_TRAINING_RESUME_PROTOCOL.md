# Episode 边界精确训练恢复协议

## 研究动机

论文级多种子 MARL 运行的最小原子单元不能只停留在“算法 × 种子”。VMAS 100-episode calibration 的首个 attention-PPO 运行记录了 16,556 s 墙钟，而中断的 MAPPO 在结果 JSON 写入前丢失了整个运行进度。宿主挂起会污染墙钟，但不改变一个事实：只在完整 seed 结束后落盘不足以支持可审计长实验。

## 状态合同

运行器在注册的 episode 间隔边界原子写入 `training_checkpoint.pt.tmp`，写完后以同目录 replace 提升为 `training_checkpoint.pt`。最终 episode 无条件保存；正式长配置使用 `checkpoint_interval_episodes=50`，默认值为 1。MATD3 calibration 的 checkpoint 随 replay 从 8.33 MiB（3,400 transitions）增至 18.22 MiB（10,000 transitions），据此将 paper cadence 从初拟 10 上调为 50，避免数十 GB 的重复 I/O。最终 `result.json` 原子写入成功后才删除训练检查点，因此训练完成后若在评测、泛化诊断或结果落盘阶段失败，恢复时不会重新训练。

on-policy IMAPPO/MAPPO/IPPO/HAPPO 检查点包含：actor/critic/potential、全部优化器、势函数更新游标、当前调度系数、未满 rollout buffer、累计日志和下一 episode。HAPPO 额外逐 actor 保存独立网络和优化器以及顺序更新 RNG。MATD3 检查点包含在线与目标 actor、在线与目标双 critic、两个优化器、完整 replay buffer、critic/actor 更新计数、最近 actor loss、累计 environment steps、日志和下一 episode。

所有算法同时保存私有 NumPy generator，以及 Python `random`、全局 NumPy、Torch CPU 和全部 CUDA RNG state。恢复加载模型后才恢复全局 RNG，避免构造网络消耗的随机数改变后续轨迹。检查点身份绑定完整注册 spec、variant、seed，以及 `run_research_study.py` 和 `src/**/*.py` 的 SHA-256 实现指纹；协议或代码变化会硬拒绝恢复，不能把跨实现拼接伪装成同一 seed。

## 等价性证据

`tests/test_training_checkpoint_resume.py` 在带真实梯度更新的确定性双 UAV 环境中比较两条路径：(a) 四个 episode 连续训练；(b) 第二个 episode 后模拟进程异常、磁盘恢复并完成训练。IMAPPO、独立 actor HAPPO 与 MATD3 的最终网络逐 tensor `torch.equal`；日志完全相等；MATD3 的目标网络、replay transition、总更新数和 delayed actor update 游标完全相等。独立测试还验证同 seed 意图序列、IMAPPO/HAPPO 私有 RNG checkpoint、身份不匹配拒绝和临时文件清理。2026-08-24 全量回归为 **178 passed, 14 warnings**。

该结果证明 CPU 测试路径在 episode 边界可 bitwise 恢复；它不等于宣称任意 CUDA kernel 跨驱动/硬件 bitwise deterministic。正式运行仍固定硬件、软件版本和代码指纹，并报告任何 CUDA 非确定性设置。注册间隔为 50 时，最坏只丢失并确定性重算最近 49 个已完成 episode，而不是声称每个 episode 都永久落盘。

## 论文可用表述

“We used atomic episode-boundary training checkpoints that captured model and target parameters, optimizer states, on-policy rollout or off-policy replay state, update cursors, accumulated logs, and all Python/NumPy/Torch RNG streams. Checkpoints were bound to the preregistered protocol and a source-code fingerprint; mismatched resumes were rejected. An interruption test established bitwise equality on the deterministic CPU test path for IMAPPO, independent-actor HAPPO, and MATD3. Paper runs checkpointed every fifty episodes and always at the final episode.”

## 剩余限制

- 旧 VMAS attention calibration 结果产生于本协议之前，没有 process CPU time 字段，16,556 s 墙钟受到宿主挂起污染；它已保留为 legacy 文件并由同配置复跑替代。
- navigation 与 dispersion 五算法 calibration 均已完成，但每场景仍只有 seed 7。
- 检查点提高实验可靠性，不增加统计证据等级；单 seed calibration 与 smoke 仍不可用于算法排序。
