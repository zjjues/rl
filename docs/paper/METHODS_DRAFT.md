# 方法章节持续草稿

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
