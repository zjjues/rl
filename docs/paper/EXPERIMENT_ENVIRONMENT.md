# 论文实验环境

## 当前本地环境

- Conda 环境：`rl-test`
- 环境路径：`D:\Programs\anaconda3\envs\rl-test`
- 操作系统：Windows 10（构建号由运行 manifest 记录）
- Python：3.10.20（Anaconda）
- 计算设备：CPU；PyTorch 报告 `cuda_available=False`

关键依赖：

| Package | Version |
|---|---:|
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| Gymnasium | 1.1.1 |
| PyTorch | 2.4.1+cpu |
| TorchVision | 0.19.1+cpu |
| sentence-transformers | 3.2.1 |
| Transformers | 4.57.6 |
| SentencePiece | 0.2.2 |
| VMAS | 1.5.2 |
| Gym (VMAS compatibility) | 0.26.2 |
| Shimmy | 2.0.1 |

## UAV 环境版本

- `uav-scheduling-v0`：历史兼容版本，局部观测 30 维；任务进展与动作幅值耦合，威胁区位置不可观测。仅用于复现既有结果。
- `uav-scheduling-v2`：当前论文候选，4-UAV 局部观测 33 维，新增最近威胁区相对位置；任务进展由目标接近度和控制方向一致性决定；新增连续最近威胁距离。
- 两个版本使用不同 Gym ID、独立 study ID 和 manifest；禁止跨版本合并均值或进行未标注的数值比较。
- v2 仍是简化的点质量速度模型，不等价于六自由度飞控或 SITL/HIL，这一外部有效性缺口尚未关闭。

`pip check` 于 2026-08-01 返回 `No broken requirements found`。完成动态意图、循环/QP-CBF 违例审计、独立偏好数据验证、NLI/原型解码、MATD3、扰动、规模、VMAS 1.5.2、PyBullet 纯函数适配、延迟预算 margin 和向量化统计后，主环境的 82 项单元测试全部通过。`pytest.ini` 将测试发现限制在 `tests/`，避免历史绘图脚本依赖用户级 Matplotlib 缓存。

## 隔离的 PyBullet 环境

- Conda 环境：`rl-pybullet`；锁文件：`environment-rl-pybullet.yml`。
- `gym-pybullet-drones==2.1.0` 固定到提交
  `e712698a05a80728b06572819dcf044596707754`。
- Windows 可用的 conda-forge PyBullet 为 3.2.5，低于上游元数据声明的 3.2.7；该偏差
  写入每次 manifest。核心 `VelocityAviary` 多机路径已实际运行。
- conda-forge NumPy/MKL 在审计主机的 `numpy.linalg.inv` 卡死，故使用 PyPI
  `numpy==2.2.6` 的 OpenBLAS wheel；`setuptools==80.9.0` 用于兼容上游仍使用的
  `pkg_resources`。
- 上游包把未使用的 RL examples 依赖 `stable-baselines3` 和 `pytest` 声明为必需项；当前
  最小运行环境未安装它们，因此隔离环境的 `pip check` 会报告这两项以及 PyBullet 元数据
  偏差。主研究环境 `rl-test` 的 `pip check` 仍完全通过。

重建与执行：

```powershell
conda env create -f environment-rl-pybullet.yml
conda activate rl-pybullet
python run_pybullet_transfer_study.py --config configs/research/pybullet_transfer_lanes.pilot.json
```

## 重建方式

在 Anaconda PowerShell 中执行：

```powershell
conda env create -f environment-rl-test.yml
conda activate rl-test
python -m pip check
python -m pytest -q
```

该环境文件面向当前 UAV 意图表示研究，并已锁定 VMAS 1.5.2 用于公开连续多智能体 cross-environment benchmark。PettingZoo 仍为可选依赖，当前只有其缺失警告，不影响 UAV 或 VMAS 研究。

## 语义模型冻结

- 模型：`sentence-transformers/all-MiniLM-L6-v2`
- Hugging Face revision：`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- 原始表示维度：384
- 当前 smoke 投影维度：16
- paper 配置投影维度：64
- 投影随机种子：17

NLI 极性模型为 `cross-encoder/nli-deberta-v3-small`，固定 Hugging Face revision
`e9890682d9e4279b7ae6d0fcfb435a43206280ec`。其 tokenizer 依赖
`sentencepiece==0.2.2`。manifest 必须同时记录嵌入模型与 NLI 模型 revision；缺少
NLI revision 的 paper 配置会被运行器拒绝。

Windows 未启用 Hugging Face 缓存符号链接优化，因此模型缓存可能占用更多磁盘，但模型 revision、权重内容和实验计算不受影响。

## 已验证命令

```powershell
python run_research_study.py --config configs/research/uav_intent_representation.smoke.json
python run_research_study.py --config configs/research/uav_intent_pretrained_semantic.smoke.json
python run_research_study.py --config configs/research/uav_intent_generalization.smoke.json
python run_research_study.py --config configs/research/uav_semantic_residual.smoke.json
```

上述运行均生成 `status=complete` 的 manifest、逐 seed 原始结果、统计汇总、结果卡和校验和。它们只证明环境及研究管线可执行，不能支持论文效果结论。
