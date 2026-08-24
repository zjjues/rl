# 方法章节持续草稿

## 20. Episode 边界精确恢复与实现身份绑定

长实验按注册的 episode 间隔原子保存完整训练态，而非只保存推理权重。IMAPPO/HAPPO 保存未消费 rollout，MATD3 保存 replay、目标网络和 delayed-update 游标；三者均保存优化器、日志、下一 episode、私有与全局 Python/NumPy/Torch RNG。检查点同时绑定完整实验 spec 与全部研究 Python 源码的 SHA-256 指纹，协议或实现变化时拒绝拼接恢复。正式 paper 配置根据 replay I/O calibration 每 50 episode 保存，最终 episode 强制保存，`result.json` 成功替换后才移除 checkpoint。确定性 CPU 中断测试对 IMAPPO、HAPPO、MATD3 得到逐 tensor 相等；该证据不外推为跨 CUDA 驱动的 bitwise 确定性。完整合同见 `EXACT_TRAINING_RESUME_PROTOCOL.md`。

## 21. 训练监控与最终统计样本量解耦

最终 paper evaluation 固定为每 seed、每 risk tier 100 episodes，不因预算调整而缩减。训练期定期监控只用于故障发现和学习曲线，独立注册 `monitor_eval_episodes=20`；UAV on-policy 监控后另执行同样 20 episodes 的 collision-probe，MATD3 没有训练期 evaluator。预算器逐算法计入 training、periodic monitoring、collision probe 与 final tier evaluation，避免把 MAPPO/HAPPO 的评估结构错误套给 MATD3。UAV v3 calibration 的 reference workload 为 on-policy 20,000、MATD3 16,000 最大 environment steps；paper workload 分别为 900,000 与 660,000。

## 22. 链式单因素消融与运行时同构

正式消融由十个变体和九条有向对照构成。每个非根变体恰有一个父对照，注册的 `changed_fields` 必须等于两份变体执行字典的真实逐字段差异；因此 attention、mask、intent shaping、CBF、NLI prototype gate、learned residual、semantic rule prior、semantic representation 和 intent channel 分别具有唯一可审计对照。主要统计族固定为九条对照乘 hard-tier 的 collision rate 与 task completion，共 18 个假设，使用共同随机数种子、配对效应、bootstrap 置信区间、exact test 与 Holm 校正。

运行时 calibration 与 paper 配置不仅要求相同键名和顺序，还要求每个变体字典逐字段完全相同；算法、critic、语义源或安全层的任何漂移都会拒绝外推。单 seed calibration 使用 16,000 最大 environment steps；正式每 run 为 860,000 steps，含训练、30 次 monitor、30 次 collision probe 与 hard-tier 100-episode final evaluation。由 process CPU 外推的 100-run 预算为 70.65–78.87 active CPU-hours。该 calibration 只验证管线和预算，不进入 18 个正式假设的效果证据。

## 23. 分块结果的内容寻址 provenance

每个正式 result 保存两个内容指纹：完整注册 `spec+variant+seed` 的 canonical JSON SHA-256，以及 runner 与全部研究源码的 implementation SHA-256。study manifest 和每次 resume history 同时记录完整 spec hash 与 implementation hash。Git commit 可因新增实验文件变化，但所有 chunk 的 implementation hash 必须一致；否则 resume 在读取缓存结果或开始新训练前拒绝。partial artifact 只在 manifest 的  missing pairs 与文件系统精确一致、已有结果身份/checksum 有效且不存在 summary 时标记 `valid_partial`。默认最终审计仍拒绝 partial。

## 18. 偏好相关性拒答与一次性外部终测

自由文本首先经过冻结 MiniLM embedding 上的 logistic relevance gate。gate 只决定“是否允许进入六轴 profile decoder”，拒绝样本的六个目标倍率全部回到 1.0；它不改变不可协商 collision 约束。阈值仅由 AerialVLN `val_seen` 负样本上限校准，开发版 gate SHA-256 为 `8518d9be...87fd`，阈值为 `0.0244081132`。AerialVLN `val_unseen` 已在设计阶段查看，因此只作为 development evaluation，不作为最终盲测。

CityNav 在接触文本前完成 v2 预注册：冻结源 commit、190,078,685-byte archive SHA-256、四个 canonical split、gate 哈希/阈值、总体 FAR、Wilson 95% CI 和 5%/10% pass-caution-fail 边界。运行器在模型推理前写 one-shot attempt marker，已有 attempt 或结果时拒绝重跑。32,637 条真实城市目标描述的最终 FAR 为 **0.961516**（95% CI **[0.959374, 0.963549]**），预注册结论为 fail。该语料禁止用于修改当前 gate 或阈值。

这个负结果否定了“冻结 embedding 线性 gate 已学到可迁移偏好相关性”的主张。更合理的解释是分类器区分了开发者模板与 AerialVLN 风格；CityNav 的目标描述产生严重域偏移。后续方法必须使用来源独立、writer-disjoint 的人工偏好 train/dev/test，并把多来源导航/任务语言作为训练负例；CityNav 只能保留为已消耗的最终外部测试。

## 19. VMAS 跨场景架构复现边界

VMAS navigation 与 dispersion 只检验 MARL 架构外部有效性。所有变体统一使用 `intent_source=none`、`intent_profile_decoder=none`、`eta=0`、无 action mask、无 safety filter、direct policy；协议验证器拒绝把 UAV 语言、奖励画像或安全语义带入 VMAS。比较对象为 attention centralized PPO、MAPPO、IPPO、独立 actor 顺序 HAPPO 和 MATD3，唯一有效主指标为场景原生 episode return。

VMASAdapter 固定 v1.5.2，强制注册 horizon，并仅映射原生 reward 与显式 collision diagnostics；`pos_rew` 不会被重命名为 UAV task completion。两个 5-algorithm/1-seed smoke 均通过 checksum/artifact 审计；它们只证明管线，不支持排序。navigation 与 dispersion 的五算法、单 seed、100-training-episode calibration 均已完成。最终统计固定 100 episodes；训练期每次监控独立注册为 20 episodes，避免把最终样本量重复执行 40 次。按 process CPU active-time 和逐算法实际 workload 外推，50-run paper 计划分别为 33.42–44.36 与 34.09–45.47 active CPU-hours；该量排除宿主挂起但不是 GPU device-hours，且单 calibration seed 仍不足以估计跨 seed 时间方差。

## 0.1 Betaflight SITL 高保真迁移

### 架构

单机闭环由三部分组成：
1. **PyBullet 物理引擎**（Windows, rl-pybullet conda env）：运行刚体动力学，RACE 机型 830g
2. **Betaflight SITL**（WSL2 Ubuntu）：运行真实飞控固件（commit `b41431ae`），含 PID 控制器、混控器、滤波器
3. **UDP Bridge**（`src/betaflight_sitl_bridge.py`）：双向转发
   - Bridge→SITL: 状态包 `@18d`（姿态/角速度/加速度/位置）→ port 9003；RC 包 `@d16H`（摇杆通道）→ port 9004
   - SITL→Bridge: 电机包 `@4f`（归一化推力 [0,1]）→ port 9002

### SITL 定制

为在 WSL2 网络 NAT 下实现解锁与持续闭环，对 Betaflight 源码做以下修改：
- `SITL_ATTITUDE_DIRECT`：跳过 Mahony 姿态估计，直接使用 FDM 四元数
- `frameStatusUdp` SIMULATOR 路径：RC 帧间隔不触发 RXLOSS
- `isUpright()` SIMULATOR 分支：绕过倾角解锁检查
- 每帧 FDM 持续 `ENABLE_ARMING_FLAG(ARMED)`：抵抗内部 disarm()

### 多机扩展

- `--port-offset <n>` CLI 参数：SITL 端口 = base + n（bf0:9002-9004, bf1:9012-9014）
- 每机独立 EEPROM、SITL 实例目录 (`bf0/`, `bf1/`)
- Bridge `drone_id` 参数 + `ports_for_drone()`

### 当前状态

单机闭环 smoke v19 通过，双机 smoke multi v2 通过。尚未达到冻结论文实验标准（需 10+ seeds，冲突场景，安全层验证）。

## 0. 当前主方法候选：文本目标概念瓶颈与残差控制

冻结文本编码器先将指令 (g) 映射为句向量 (e)。目标解码器输出七维可解释画像

\[
\hat\rho(g)=(\hat w_d,\hat w_e,\hat w_c,\hat w_s,\hat w_q,\hat w_t,\hat w_h).
\]

`prototype_ridge` 使用 19 个训练意图及每个目标的一对正/反文本原型拟合岭回归；原型目标只改变对应维度，其余维度保持 1。评估 query 的 canonical label 和真实画像不进入编码器或策略，只在推理后计算画像误差。连续画像调节目标吸引、邻机排斥、速度阻尼和动作上限，PPO actor 只学习有界残差。`dual_ridge`、三姿态检索、零残差、无语义残差和真实画像 oracle 构成机制消融。

反事实评估对每个目标构造 low/neutral/high 三条指令，仅改变一个注册画像维度。三条轨迹采用 common random numbers：同一训练 seed、风险层和 episode 共享 reset seed，从而把行为差异归因于文本干预而非场景变化。

### 0.1 UAV benchmark v2

为避免意图遵循被不真实任务定义主导，v2 将每机任务进展写为

\[
\Delta q_i=\kappa\frac{1}{1+d_i/r}\left(0.2+0.8\max(0,\langle \hat a_i,\hat d_i\rangle)\right),
\]

其中 (d_i) 是到分配目标的距离。观测加入最近威胁区的相对三维位置，使 threat avoidance 不再依赖不可观测变量。二值碰撞/威胁违规与连续最小机间距/最近威胁距离同时报告，以区分零事件退化和真实安全裕量变化。`uav-scheduling-v0` 保留用于历史复现，v2 结果不得与 v0 数值直接合并。

## 1. 问题定义

考虑由 \(N\) 个 UAV 构成的协作式部分可观测马尔可夫博弈。智能体 \(i\) 在时刻 \(t\) 接收局部观测 \(o_t^i\)，共享高层意图描述 \(g\)，并输出连续控制动作 \(a_t^i\)。高层意图由文本编码器 \(E_\psi\) 映射为向量：

\[
z = P(E_\psi(g)),
\]

其中 \(E_\psi\) 在主实验中为冻结的预训练文本编码器，\(P\) 为可选的固定或可训练投影。one-hot、随机稠密码和 legacy hash 均作为表示消融，不作为语义模型。

### 1.1 表示组定义

- `pretrained_semantic`：冻结并锁定 revision 的 `sentence-transformers` 文本编码器；当原始维度大于策略意图维度时，使用固定种子的高斯随机投影并重新单位化。随机投影只用于降维，不根据实验回报拟合。
- `random_dense`：每个意图对应独立高斯随机单位向量，向量与文本内容无关，用于控制“稠密高维表示”本身的影响。
- `legacy_hash`：复现历史 Stage7 的 SHA256→PRNG 确定性单位向量，不保留文本间语义邻域。
- `onehot`：离散意图身份基线。

真实语义编码路径在模型或依赖不可用时必须失败，不能静默退回 hash/random 表示。

### 1.3 目标落地的语义适配器

冻结语言模型的相似度未必与 UAV 控制目标一致。当前候选适配器只使用训练意图，将文本向量通过 dual ridge 映射到标准化的七维奖励画像；held-out 查询在评估时只输入文本，未见标签的真实画像仅用于事后诊断，不参与编码。几何 pilot 选择的探索性设置为 ridge=0.01、semantic weight=0、objective weight=1。该选择尚未由 paper 级行为结果确认，必须与 raw pretrained 和无适配器版本共同报告。

### 1.2 身份基线与查询协议

`onehot` 现在表示完整意图 catalog 的身份矩阵，而不是 attack/stealth/frozen 三模式。若 catalog 含 (K) 个意图，则 one-hot 维度至少为 (K)。训练 held-out 实验仅激活训练标签对应坐标；未见标签在评估时激活训练中从未出现的坐标。

对 paraphrase 查询，one-hot 与 random-dense 使用 canonical label 查表，因此是“已知身份 oracle”控制；pretrained semantic 与 legacy hash 只消费新的查询文本。该设计分别回答：文本编码器是否保持同义结构，以及即使给出完美身份，策略是否能执行相同任务。

## 2. 意图条件 Actor

所有同质 UAV 共享策略参数。分散执行时，actor 仅使用局部观测和共享意图：

\[
\pi_\theta(a_t^i\mid o_t^i,z,m_t^i),
\]

其中 \(m_t^i\) 为由任务意图导出的动作可行性掩码。连续动作由高斯策略采样并经 `tanh` 映射到环境动作范围。

### 2.1 语义条件残差策略

5-seed 可行性实验显示，从零学习的策略明显弱于简单目标跟踪/势场控制器。因此主方法候选使用相同的经典控制器产生导航先验 \(a_{0,t}^i=f_{rule}(o_t^i,p)\)，其中 \(p\in\{attack,stealth,neutral\}\) 决定目标吸引、邻机排斥和速度阻尼增益。actor 学习受限残差：

\[
u_t^i=\tanh(\Delta_\theta(o_t^i,z)+\sigma_\theta\epsilon),
\qquad
h(u,a_0)=\begin{cases}1-a_0,&u\ge0\\1+a_0,&u<0,\end{cases}
\qquad
a_t^i=a_{0,t}^i+\lambda u_t^i h(u_t^i,a_{0,t}^i).
\]

残差均值头初始化为零，因此初始确定性策略等于 rule-only 控制器。分段 headroom 映射在规则动作饱和到 \(\pm1\) 时仍保留向动作空间内部的梯度；v1 使用 inverse-tanh 中心，因饱和梯度过小而被保留为失败版本。当前探索性尺度为 \(\lambda=0.5\)。每个 rollout transition 保存采样残差潜变量，PPO 概率比在潜变量分布上计算；最终动作是潜变量的确定性映射，因此无需对饱和动作求逆。核心因子比较为 full semantic residual、nonsemantic residual、semantic direct policy 和 rule-only。

规则先验上下文有三种显式口径：`neutral` 不接收任务信息；`oracle_posture` 接收任务姿态，只用于传统规划上界；`intent_retrieval` 仅用文本编码向量检索训练意图并取得增益姿态。主语义方法采用 `intent_retrieval`，无语义 residual 采用 `neutral`，从而避免 suite 的 canonical posture 旁路泄漏给学习方法。

## 3. 意图查询的集中式 Critic

训练阶段允许 critic 访问所有智能体观测。每个智能体观测先编码为特征 \(h_i\)，意图向量生成 query：

\[
q=W_qz,\quad k_i=W_kh_i,\quad
\alpha_i=\operatorname{softmax}(q^\top k_i/\sqrt{d}),
\]

\[
c=\sum_i\alpha_iW_vh_i,\quad V_\phi(s,z)=f_\phi([s;c]).
\]

注意力用于在不同意图下重新分配对智能体状态的关注。`uniform` 与纯 MLP critic 作为消融。

## 4. 意图势函数奖励

状态—意图势函数 \(\Phi_\omega(s,z)\) 产生势差奖励：

\[
r_t^{\mathrm{int}}=\gamma\Phi_\omega(s_{t+1},z)-\Phi_\omega(s_t,z),
\]

\[
\tilde r_t=r_t^{\mathrm{env}}+\eta_t r_t^{\mathrm{int}}.
\]

需要在实验中验证：势函数是否提高样本效率、是否保持任务最优性、不同更新频率的稳定性，以及当意图在 episode 内切换时是否需要分段势函数边界处理。

## 5. 意图—任务一致性

主方法要求文本意图、战术姿态、动作约束和环境奖惩共享同一份结构化意图元数据。例如 `stealth_approach` 必须对应避开威胁区的姿态，而 `aggressive_pursuit` 对应允许进入威胁区的姿态。训练不得独立随机生成互相冲突的意图和环境目标。

当前实现将意图分为 `attack`、`stealth` 和 `neutral`。给定 episode 战术姿态时，只从同姿态或 neutral 意图中采样；动作掩码也由同一姿态元数据生成。该设计消除了旧版中随机文本意图与环境奖励姿态可能相反的混杂因素。

### 5.1 细粒度意图奖励画像

为避免 25 条文本在环境中退化为二元姿态，任务定义为每个 canonical intent 注册七维奖励倍率：

\[
\rho_g=(w_d,w_e,w_c,w_s,w_q,w_t,w_h),
\]

分别作用于距离进展、能耗、碰撞、安全间距、任务进展、时间和威胁区分量：

\[
r_t^{env}(g)=w_dr_t^d+w_er_t^e+w_cr_t^c+w_sr_t^s+w_qr_t^q+w_tr_t^t+w_hr_t^h.
\]

倍率由 `intent_objectives.py` 显式注册并随配置开关；历史实验默认关闭。该画像属于任务语义定义，因此正式实验必须加入关闭画像和倍率扰动消融。

## 6. 未见意图泛化

训练集合包含 19 个 canonical intents。评估查询分为：

- `seen`：训练文本校准点；
- `paraphrase`：标签已见但描述从未用于训练；
- `unseen`：标签和描述均未用于训练。

查询向量只在训练结束后生成，不加入 rollout 或优化器。表示诊断报告 top-1 retrieval、姿态 retrieval、matched cosine 与 semantic margin；行为评估报告安全、任务完成和回报。文本 query 先在每个 seed 内平均，再以训练 seed 为统计单位。

## 7. 配对随机性

所有表示变体使用相同研究 seed。环境 reset seed 由研究 seed、训练 episode、风险档和评估 episode 确定性派生。反事实 query 不再进入 seed 公式，因此同一风险层内所有文本干预面对相同初态、目标和随机扰动。该规则写入 manifest，使 paired difference 同时配对训练种子和评估场景。旧版包含 query index 的结果只作非因果探索证据。

### 7.1 多目标意图遵循指标

每个 query 除回报、碰撞和完成度外，还记录平均剩余能量、动作强度、速度、目标距离、最小邻机距离和威胁区违规率。每个训练 seed 内计算 profile preference 与相应行为的 tie-aware Spearman：energy→剩余能量、distance→负目标距离、time→速度、safety→最小邻机距离。查询不是独立训练样本，相关系数必须先按 seed 计算，再跨 seed bootstrap。

### 7.2 冻结 NLI 极性与目标类别门控

当前候选文本解码器将“目标类别”和“高/低极性”分开。冻结 MiniLM 只用于目标相关性；冻结 DeBERTa-v3 NLI 对每个目标的直接陈述假设计算

\[
s_k(x)=P(E\mid x,h_k)-P(C\mid x,h_k),\qquad
\hat w_k=1+\begin{cases}0.7s_k,&s_k\ge0\\0.5s_k,&s_k<0.\end{cases}
\]

元话语假设（例如“instruction assigns high priority”）在命令式文本上产生系统性矛盾误判，因此改用目标特定的直接陈述。`nli_prototype_gated` 的相关性门控只在训练侧独立原型上建立：每个目标包含四种高权重和四种低权重表述，另有中性类；冻结嵌入的类内均值归一化为八个质心，查询只允许最近的非中性质心对应目标进入 NLI 极性判定。该结构在定义上将每条最小反事实指令限制为至多一个非中性目标，避免同时改变多个控制增益。v4/v5 已用于方法诊断，后续不得作为无偏确认集。

### 7.3 逐步 CBF 审计与主动安全间距

联合动作通过循环半空间投影满足一步线性分离约束。对智能体对 \((i,j)\)，审计量为

\[
\nu_{ij}=\max\{0,d_{min}-d_{ij}-\Delta t\,\beta\,\hat r_{ij}^{\top}(v_i-v_j)-\Delta t\,g\,\hat r_{ij}^{\top}(a_i-a_j)\}.
\]

每步记录最大/平均 \(\nu_{ij}\)、违例对比例和预测最小成对距离。该离散一步约束不得表述为连续时间前向不变性证明。为提高目标可辨识性，`safety` 调整提前避让半径，`collision` 调整近程响应和屏障距离，两者不再共用同一固定 2 m 势场。

安全合同进一步规定：自然语言只能提高 CBF 最小距离，不能将其降低到 `base_min_distance` 以下；collision 不再作为可被用户降权的偏好。低 safety 只允许收紧或放松屏障外的主动编队间距。后续正式语言目标集合应移除 collision-low/high，将物理接触作为约束违例和结果指标，而非偏好维度。

除快速循环投影外，`pairwise_qp` 显式求解

\[
\min_{a\in[-1,1]^{3N}}\frac12\lVert a-a_{nom}\rVert_2^2
\quad\text{s.t.}\quad A_ta\ge b_t,
\]

其中每行对应一个成对一步分离约束。实现使用固定 SciPy 版本的 SLSQP；只有审计最大违例不超过预注册 tolerance 时才记为成功。数值终止失败时运行较长循环投影并返回残余违例更小的候选。每步记录求解器报告状态、审计成功率、迭代数、fallback 和墙钟延迟。该 QP 仍基于离散点质量局部线性化，不构成连续时间或高保真动力学的形式化安全证明。

正式语言空间从七维调整为六个可协商偏好 `distance/energy/safety/task/time/threat`；`collision` 只作为不可放宽的约束与结果指标。v8 suite 从继承链中显式排除 collision-low/mid/high，避免训练或评估用户“要求容忍碰撞”的不合规能力。

### 7.4 episode 内动态意图干预

在固定步切换盲文本意图。对每个切换后状态，同时计算新意图动作与“保持旧意图”的反事实动作；两者使用完全相同的观测，因此动作响应延迟不受轨迹分叉混杂。报告首步动作差、切换后平均动作差、超过预注册阈值的响应率/删失延迟，以及切换前后能量、目标距离、最小邻机距离、任务和威胁区指标。不同 transition 在同一风险层共享 reset seed。

## 8. 尚待补充的理论与实现说明

### 8.1 连续动作 off-policy 强基线

MATD3 基线使用同质 UAV 共享的确定性局部 actor，训练时 twin centralized critics 接收全局状态与联合动作。target action 加入截断高斯平滑噪声，critic 使用两目标 Q 的较小值，actor 每两次 critic 更新一次并以 Polyak 系数更新 target 网络。该策略不接收文本或 canonical label；标签只定义环境任务奖励，因此不能产生语言条件行为。比较时同时报告环境 transition 数、replay warmup 和 actor/critic 更新次数。

### 8.2 扰动与规模外推

鲁棒性协议分别控制零均值高斯风加速度、局部空间观测噪声、离散动作时延以及最近邻通信特征丢包。零扰动不消耗额外随机数，以保持历史 seed 轨迹逐步等价。所有非零扰动只使用环境内部 RNG，因此不同算法在相同 reset seed 下面对同一随机过程。

规模外推固定每机最近 3 个邻居，使 4/8/12/16 UAV 的局部观测均为 30 维。共享 actor 在 4 UAV 训练，评估时按实际 agent batch 分散执行；集中式 critic 不参与扩大规模的推理。

### 8.3 Crazyflie 刚体动力学迁移

高保真补充实验使用固定提交 `e712698a05a80728b06572819dcf044596707754` 的
`gym-pybullet-drones`，通过 `VelocityAviary` 的内置 PID 将物理速度指令转换为四旋翼
RPM。控制器与模拟器以 30 Hz 交互，PyBullet 以 240 Hz 积分。该实验验证高层控制器
和安全层面对旋翼刚体闭环时的迁移，不属于 Betaflight/PX4 SITL、HIL 或实机试飞。

针对互换目标导致的局部屏障死锁，冻结控制器采用与随机场景无关的 agent-index 高度
通道和统一环流作为对称破缺。速度空间 QP 为：

\[
\min_{v\in[-v_{max},v_{max}]^{3N}}\tfrac12\lVert v-v_{nom}\rVert_2^2,
\quad
\hat r_{ij}^{\top}(\alpha(v_i-v_j)+(1-\alpha)(\dot p_i-\dot p_j))
\ge (d_{min}-d_{ij})/H.
\]

报告物理最小间距、碰撞步比例、安全距离违规步比例、目标成功、最终 RMSE、控制能量、
命令空间约束残差和延迟。命令空间残差为零不代表刚体轨迹始终满足安全距离；两者差异
显式作为模型失配报告。

鲁棒版本不从评估轨迹拟合 margin，而是预注册跟踪延迟预算 \(\tau=80\) ms。两架速度
受限飞行器在该时段的最坏相对闭合距离为
\(m_\tau=2v_{max}\tau=0.04\) m，QP 使用 \(d_{min}+m_\tau\) 作为约束距离，结果仍按原始
\(d_{min}\) 统计安全违规。该构造只覆盖速度上界与延迟预算内的闭合误差，不覆盖状态
估计偏差、碰撞几何误差、未建模执行器故障或超过预算的网络延迟。

- 预训练文本编码器及投影方法；
- 组合意图构造；
- episode 内动态意图切换；
- 可行性掩码是否属于策略的一部分；
- 复杂度与智能体数量的关系；
- 与 MAPPO、IPPO、HAPPO、MADDPG/MATD3 的关系。

## 9. 分块运行的 provenance 与结果冻结

一个 study 可以为控制 GPU 时长而按变体分块执行，但 `--resume` 只能追加变体，不能改变 seeds、环境、训练超参数、意图定义、评估风险层或泛化 suite。同名变体必须与已有完整定义一致；缓存结果必须同时匹配 seed、variant key 和完整变体字段。若任一条件不满足，runner 终止而不是复用结果。

Composite manifest 使用 schema v2。顶层 `config` 表示合并后的完整研究协议，`run_history` 对每次调用保存命令、内嵌子配置、Git commit、dirty-worktree 状态、Python/依赖和起止时间。逐次记录不可由后一次 resume 覆盖。Artifact validator 独立核对：

1. 外部预注册配置与顶层协议字段、变体定义；
2. manifest 内嵌配置与 run-history 变体覆盖；
3. 每个 variant/seed 的文件存在性、seed 和变体身份；
4. summary 中 primary metric 的 raw 数组与逐 seed 原值；
5. 除 checksum 文件自身外的完整 SHA-256 集合。

`valid` 仅表示 artifact 内部一致，不自动提升证据等级。若训练时 Git 工作区非 clean，validator 保留 warning；`paper` 级 artifact 将该 warning 视为错误。

## 10. 配对推断与多重比较

所有算法按研究 seed 配对。对 treatment 与 baseline 的差值 \(d_s\) 报告均值、median、win rate、paired standardized effect \(d_z=\bar d/s_d\) 和 seed-level bootstrap 95% CI。缺失/非有限观测只按完整 pair 同步排除，禁止两组分别删值后重新配对。

均值差的双侧零假设使用 paired sign-flip randomization test。seed 数不超过 16 时枚举全部 \(2^n\) 个符号排列；更大样本使用固定 seed 的 Monte Carlo 近似。预注册主 family 包含所有 baseline、risk tier 上的 collision rate 与 task completion。本轮 3 baselines × 3 tiers × 2 metrics 共 18 个假设，使用 Holm step-down 控制 family-wise error rate 0.05。未校正 CI 与 Holm 结论冲突时，主张以 Holm 结果为准并同时报告两者。

## 11. 架构 pilot 的解释边界

`uav_imappo_main` 的四个变体都使用 25 维真实 one-hot identity code，并设置 `disable_intent_reward=true`。2026-08-20 的实现审计进一步发现，名为 MAPPO 的变体实际仍使用 `algorithm="imappo"`，且未实现的 `critic_mode="concat"` 执行为 attention。故该实验连“attention + mask 相对 concat MAPPO”也不能回答；历史数值只保留为 protocol-identity 失败案例。

修正协议 `uav_marl_architecture_v2_paper` 使用五个显式计算路径：identity-conditioned attention I-MAPPO、其 no-mask 版本、全局状态 centralized-MLP MAPPO、local-critic IPPO 和 centralized-twin-critic MATD3。运行前 semantic validator 检查保留 baseline key 与 algorithm 一致，拒绝任何未知 critic mode，并禁止 IPPO/MAPPO 的静默 critic 覆盖。正式比较必须从 clean commit 重新训练，不能重标旧 checkpoint 或逐 seed JSON。

## 12. 链式消融契约

语义系统的多个机制存在依赖关系，因此全部相对 full 的平行消融会把表示、解码器和规则先验混在一起。预注册消融改为以 full 为根的有向树。每条边定义 reference、variant、唯一 factor、精确 changed fields、primary tiers、primary metrics 与双侧可证伪假设。验证器要求每个非 full 变体恰有一个父节点、整张图从 full 可达、无环，并比较完整 variant 字典确认没有未声明漂移。

主链为：full → no-profile-prior → identity-oracle → no-intent。第一条隔离解码目标画像进入经典控制先验的影响；第二条在 neutral prior 下比较冻结语义几何与 canonical-label identity code；第三条移除 actor/critic intent 输入及 intent-potential shaping。其余 mask、critic attention、intent shaping、CBF、NLI gate 和 learned residual 均直接相对 full 单因素改变。

`intent_source="none"` 不改变任务生成：标签仍从同一 catalog、同一 posture 和共同随机数协议采样，环境仍使用同一标签奖励画像；actor 与 critic 接收全零 64 维向量。为保持与 full 相同的动作可行域，posture-derived mask 仍生效，因此它是“无 actor/critic intent、保留 mask 侧信道”的对照，而非信息论意义的零任务信息。metadata 与报告必须披露该边界。

## 13. 资源与模型缓存审计

MiniLM 和 NLI CrossEncoder 在单个 study 进程内按 `(model, revision, device)` 复用。缓存对象固定为 eval mode 且所有参数 `requires_grad=false`；每个变体仍独立拟合轻量 objective adapter、维护 profile cache 并初始化 MARL 网络。该优化不改变模型权重或预测函数，只消除重复磁盘加载与显存构造。

每个逐 seed result 记录总墙钟时间、CUDA 峰值、actor/critic/potential 的总参数和可训练参数、以及当时的冻结文本模型缓存键。paper 报告同时给出 transition 数、训练更新次数与这些系统资源，防止只报告任务指标而隐藏推理/训练代价。

## 14. 小规模 CBF 的等价执行优化

8 UAV 只有 28 条成对约束。原 cyclic projection 在 CUDA 上对距离、violation 和 diagnostics 逐标量 `.item()`，导致每个控制步发生数百次同步；其成本来自执行方式而非约束规模。优化实现仍按固定 lexicographic pair 顺序执行四轮 Gauss--Seidel 投影，并在每个 active constraint 后对完整联合动作执行 `[-1,1]` clipping。不同之处是把观测和候选动作一次性转为 host float32 数组，在同一事务中完成投影与诊断，再把结果返回原设备。CBF 本来就通过离散 active-set 分支不可微，因此该路径不撤销受支持的梯度合同。

可复现实验 `benchmark_cbf_runtime.py` 同时运行冻结的旧实现 oracle 与新实现。8 agents、28 pairs、4 iterations、100 repeats 下，RTX 3050 的 filter+diagnostics 平均延迟为 54.80/0.83 ms，CPU 为 7.61/0.76 ms；最大动作/诊断误差分别为 `4.25e-7`/`2.03e-8`。随机 CPU/CUDA 张量测试还覆盖重合位置的默认方向。该证据支持“注册约束在数值容差内等价且运行更快”，不支持严格 bitwise identity、连续时间安全或硬实时截止保证。

## 15. HAPPO 强基线

HAPPO 基线遵循原论文的 sequential policy update，并以 PKU-MARL/HARL `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2` 为计算协议参照。与共享 actor 的 MAPPO 不同，8 个 UAV 各自拥有独立连续动作 actor π_i；参数存储和 optimizer state 均不共享。每次 rollout 更新抽取一个随机排列 \((i_1,\ldots,i_N)\)，初始 factor \(M=1\)。训练 agent \(i_m\) 时 PPO clipped surrogate 乘以固定的前序 factor；该 agent 完成所有 epoch 后，在整段 rollout 上重算动作对数概率，并更新

\[
M \leftarrow M\,\exp\{\log \pi_{i_m}^{new}(a_{i_m}|o_{i_m})-\log \pi_{i_m}^{old}(a_{i_m}|o_{i_m})\}.
\]

所有 actor 更新结束后才训练 centralized MLP value critic。HAPPO 不接收 intent vector、task-derived mask、规则先验或 CBF，故比较的是纯 MARL 强基线。每个 result 记录 independent actor count、sequential factor scheme、actor/critic 参数量；训练日志记录 factor mean/absolute max 和当次第一个 agent。该本地实现经过公式与官方 runner 协议交叉检查，但不是官方 HARL 源码的 vendored copy；正式论文仍需独立的官方框架交叉运行。

## 16. 外部语言数据与构念边界

语言证据分成两类。第一类是独立采集的六维 UAV 操作偏好，直接监督 distance、energy、safety、task、time、threat 的 low/high 与 neutral；writer 按身份隔离 train/dev/test，第二人盲审，分歧由独立第三人裁决。所有阈值在 test SHA-256 冻结前确定，报告裁决前一致率、Cohen's kappa、macro-F1、逐类 recall、校准和拒答率。

第二类是公开 UAV 导航语言，仅用于 out-of-distribution negative control。AerialVLN 指令由人类根据参考飞行视频撰写，具备真实 UAV 措辞，但其标签是路径/目标，不是操作员对多目标 reward 的偏好。导入器因此不输出 objective 或 polarity；机器 manifest 强制 `label_compatibility=navigation_instruction_not_preference` 只能搭配 `usage=ood_negative_control`。在该集合上只报告偏好解码器的误接收/拒答，不报告偏好分类准确率。这一约束阻止通过研究者后验映射制造虚假的外部泛化。

实现中 preference decoder 只输出六轴 profile；collision reward weight 对全部 canonical label 固定为 1.0，环境拒绝语言 profile 对 collision 的非单位 override。该合同的含义是“语言不能放宽接触惩罚”，不是“策略形式上保证永不碰撞”；物理安全仍需独立 CBF/备用控制与系统验证。AerialVLN 128 条 OOD smoke 显示未校准 0.20 profile 偏移阈值仍有 39.06% 激活，故当前不设 production threshold，必须等待独立 preference dev 校准。

## 17. 长实验预算与原子恢复

正式运行仍注册完整 variants、seeds、训练 episode、评测 episode 和风险档。执行时可以选择 variant×seed 子集；每个逐 seed result 原子写入独立目录。若尚有注册 pair 缺失，manifest 为 `partial`，列出 missing pairs，且不生成 summary、显著性检验或 `complete` 标志。后续 `--resume` 首先核验已有 result 的 seed、variant key 和完整 variant definition，所有 pair 齐全后重新从磁盘载入全集并一次性聚合。因此中断和调用顺序不改变统计样本。

预算器 schema v2 按训练与评测 environment-step 数外推，并强制声明参考来源与 timing field。wall time 仅作为独占加速器占用代理，可能含宿主挂起；process CPU time 排除挂起但不是 GPU device time，二者不得互换命名。总 workload 比率给出低估计，训练/评测最大单项比率给出保守估计；同构算法 study 中唯一极端首轮耗时可拆为一次性 cold start，混合算法 study 不作该假设。VMAS 采用 100-training-episode calibration 的 process CPU time，预算不再使用受挂起污染的 smoke wall time。
