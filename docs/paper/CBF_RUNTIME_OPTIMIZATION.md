# Pairwise CBF 运行时优化与论文披露

## 问题

8 UAV 产生 28 个 pair。旧 cyclic projection 每个 pair/iteration 都用 `.item()` 判断距离和约束激活，CUDA 因而发生同步；每步 diagnostics 再次逐 pair 同步。中断 smoke 中含 CBF 的变体约 95–116 s，而 `no_cbf` 约 18 s。

## 实现

- 数学约束、pair 顺序、projection iterations 和 action clipping 不变。
- 小规模数组一次性在 host float32 上执行顺序投影；投影后就地生成相同 diagnostics。
- 策略侧使用 fused filter+diagnostics；资源评估复用该结果。
- 旧实现被冻结在 `benchmark_cbf_runtime.py` 和单元测试 reference 中，防止“只测新实现自己”。

## 数值与性能证据

固定 seed `20260820`、8 agents、28 pairs、4 iterations、100 repeats：

| Device | Legacy ms | Optimized ms | Speedup | Max action error | Max diagnostic error |
|---|---:|---:|---:|---:|---:|
| RTX 3050 6GB | 54.798 | 0.827 | 66.237× | 4.25e-7 | 2.03e-8 |
| CPU | 7.615 | 0.756 | 10.076× | 3.80e-7 | 2.03e-8 |

原始 JSON：

- `docs/paper/audits/cbf_runtime_benchmark_cuda.json`
- `docs/paper/audits/cbf_runtime_benchmark_cpu.json`

完整 10-variant smoke 的 full 冷启动为 68.65 s；其余缓存变体平均 12.56 s，`no_cbf` 为 13.14 s。由此只能判断 CBF 不再是该 smoke 的主导耗时。

## 论文表述边界

允许表述：在注册容差内复现旧 cyclic projection，并显著减少本机上的运行时同步开销。

禁止表述：bitwise 完全相同、硬实时保证、形式化安全、连续时间前向不变性，或基于单台机器的普遍速度结论。paper 版应报告硬件、软件版本、warmup/repeats 和原始 JSON。
