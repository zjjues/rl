# 候选贡献证据台账

| ID | 候选贡献 | 当前状态 | 必需代码 | 必需证据 | 论文风险 |
|---|---|---|---|---|---|
| C1 | 目标落地的语义意图表示 | 编码、目标适配器、true one-hot、未见/改写协议已实现 | 组合意图、paper 运行 | semantic > random/hash/one-hot；同义和组合泛化 | 5-seed feasibility 只有混合信号，尚无稳定 unseen 行为优势 |
| C2 | 意图查询 centralized critic | 已有原型 | attention/uniform/MLP 一致接口 | 完整 critic 消融、注意力解释和稳定性 | 可能只是常规注意力组合 |
| C3 | 状态—意图势函数奖励 | 已有原型 | 切换边界、更新策略、日志 | 样本效率、最终性能、最优性影响 | 理论和因果证据不足 |
| C4 | 动态意图重规划 | 初步指标 | 标准化切换场景、连续延迟度量 | 多意图、多风险、多 seed 延迟分布 | 当前延迟度量过于离散 |
| C5 | UAV 安全协同应用 | 简化环境 | 高保真动力学、通信和扰动 | SITL/HIL/真机、传统规划基线 | 当前环境不足以支撑系统顶刊 |
| C6 | 可复现 MARL 评估 | 运行/统计、校验和、确定性配对 seed、真实 IPPO/MATD3 已实现 | 容器与跨环境配置 | 一键复现、全量结果与 CI | 尚未完成首次 frozen paper 运行 |
| C7 | 经典导航先验 + 语义条件残差 MARL | 共享 rule prior、零残差初始化、rollout 基准动作与 factor smoke 已实现 | 多 seed residual pilot、动态意图与鲁棒性场景 | full > nonsemantic residual > rule-only，并保持安全下界；相对 direct 提高样本效率 | 当前 smoke 只证明 prior 有效，语义残差与无语义残差几乎相同 |

## 使用说明

只有当“必需代码”和“必需证据”均完成时，候选贡献才可以写入论文摘要或贡献列表。否则只能放在方法动机、探索性结果或未来工作中。
