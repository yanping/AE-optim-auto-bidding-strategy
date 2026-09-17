# AlphaEvolve 驱动的广告买方侧（Advertiser-Side）Target-CPC 自动出价策略优化 Demo 规划方案

---

## 1. 项目背景与定位 (Executive Summary & Problem Statement)

### 1.1 业务与技术背景
在计算广告实时竞价（RTB）领域，广告主通常希望在满足特定商业约束的前提下最大化业务成果。目前工业界与学术界主流关注的典型任务之一，是在**目标点击成本约束（Target-CPC Constraint）和总体预算约束（Budget Constraint）**下，尽可能多地获取点击量（Click Maximization）。该任务直接对应阿里巴巴等机构在 **BAT (Benchmark for Auto-bidding Task)** 论文中所定义的**实验 2（CPC-Constrained Bidding）**。

然而，现有学术 Benchmark（如 BAT）大多站在**广告平台（Ad Exchange / Ad Platform）**的全知视角建模：
* 平台拥有所有竞价者的全局出价记录，可以通过历史全量回放（Full Market Replay）直接评估任何新出价策略；
* **但现实中的广告买方（Advertiser / DSP）天然处于“部分可观测（Partially Observable）”环境**：广告主看不到竞争对手的具体出价，也看不到全市场的清算底价。当广告主输掉拍卖时，只能获得**右删失反馈（Right-Censored Feedback）**——仅知道“市场赢标价高于我当时的出价”，但无法直接获知真实市场价。

现阶段，多数广告买方依赖人工经验规则（如基于误差反馈的固定步长调节器、经验启发式阈值规则等）。这些规则难以自适应动态变化的市场环境，而直接引入黑盒深度强化学习（Deep RL）不仅面临巨大的样本外延安全风险，还面临“黑盒不可审计、不可解释、难以直接合规部署”等工业痛点。

### 1.2 核心方案定位：AlphaEvolve + Advertiser-Side Sandbox
本项目旨在构建一个高说服力、可交互的算法 Demo，向客户证明：**即便在不掌握竞争对手出价、仅依赖广告主自身历史删失日志的严苛现实条件下，仍然能够借助 Google DeepMind 的 AlphaEvolve（基于大语言模型驱动的代码进化算法框架），自动进化并发现优于人类专家基线的符号化、可解释、可直接部署的自动出价程序（Auditable Bidding Code）。**

本方案的核心故事线与技术亮点包括：
1. **真实买方约束建模（Censored Market Response Modeling）**：利用生存分析（Survival Analysis / Kaplan-Meier）从买方历史输赢日志中学习连续市场响应曲线 $P(\text{win} \mid \text{bid}, \text{context})$，构建离线仿真代理评测器（Learned Simulator）；
2. **代码级算法进化（Programmatic Code Evolution）**：利用 AlphaEvolve，以人类专家启发式规则为种子，以 LLM 作为变异/重组算子，搜索可解释的 Python 调价函数 $m(state)$，发现诸如非线性误差反馈、价值依赖调价（Value-Dependent Bidding）、自适应分时阻尼等高级策略结构；
3. **双重视角沙盒评估（Dual-View Sandbox & Oracle Offline A/B）**：利用公开真实数据集（以 iPinYou 为 MVP，预留 AuctionNet 接口），构建“研究者上帝视角 vs 广告主受限视角”的沙盒机制。AlphaEvolve 在搜索期间绝对看不到隐藏的市场真实清算价，最终通过未公开的真实清算价进行离线“上帝回放”（Oracle Replay），以硬核数据证明：**在近似模拟器中进化出的出价策略，在真实物理市场依然产生显著正向 Uplift**；
4. **与前沿研究呼应**：
   - 吸收 **Meta KDD 2024**（*Offline Reinforcement Learning for Optimizing Production Bidding Policies*）的核心理念：立足于已有生产规则，通过离线学习优化，产物必须是可审计、高可靠的规则体系；
   - 继承 **Google DeepMind AlphaEvolve**（2025）的代码演化框架与 FunSearch 思想，利用代码符号化演化彻底消除黑盒神经网络策略的上线隐患；
   - 验证 **BAT (Benchmark for Auto-bidding Task)** 与 **NeurIPS 2024 AuctionNet** 在买方侧的适配闭环。

---

## 2. 问题数学形式化与核心技术攻坚

### 2.1 优化问题定义 (Optimization Formulation)
考虑广告活动（Campaign）周期内的连续 $T$ 次展示竞价机会（Auction Impressions），广告主面临如下带约束的序列决策优化问题：

$$\max_{\pi} \sum_{t=1}^T \text{click}_t$$

受限于两个硬性约束：
1. **Target-CPC 成本约束**：
   $$\text{CPC}_{\text{real}} = \frac{\sum_{t=1}^T \text{cost}_t}{\sum_{t=1}^T \text{click}_t} \le \text{CPC}_{\text{target}}$$
2. **预算约束**：
   $$\sum_{t=1}^T \text{cost}_t \le \text{Budget}$$

其中：
* $\text{click}_t \in \{0, 1\}$ 表示第 $t$ 次展示是否产生点击；
* $\text{cost}_t$ 表示第 $t$ 次展示广告主的实际支付金额（二价拍卖下为竞争对手最高出价，即市场清算价 $z_t$；若未中标则 $\text{cost}_t = 0$）；
* 策略 $\pi(state_t)$ 输出出价 $b_t$。

### 2.2 买方侧数据删失问题（Censored Feedback Dilemma）
广告主在真实日志中对于单次竞价的观测数据为 $(x_t, b_t, y_t, c_t)$，其中：
* $x_t$：请求上下文特征（展示位、时间段、用户群体标签、预估 pCTR 等）；
* $b_t$：广告主本次给出的出价；
* $y_t \in \{0, 1\}$：是否赢标（$y_t = \mathbb{I}(b_t > z_t)$）；
* $c_t$：结算金额。若 $y_t = 1$，观测到真实结算价 $c_t = z_t$；**若 $y_t = 0$，真实市场价 $z_t$ 被右删失（Right-Censored）**，广告主只获得隐式不等式：
  $$z_t > b_t$$

### 2.3 核心技术攻坚：如何构建不会被 AlphaEvolve “欺骗”的 Evaluator
AlphaEvolve 具备极强的探索与优化能力。如果离线模拟器（Learned Simulator）在历史数据覆盖较少的极端出价区间存在预测失真（例如对 $b \gg b_{\text{hist}}$ 错误预估高赢率且低 CPC），AlphaEvolve 极易搜出利用模拟器 Bug 的外推解（OOD Exploitation）。

为解决这一问题，本 Demo 确立了**四重安全与保真防护网**：
1. **出价乘子搜索范围约束（Bounded Multiplier Space）**：
   出价形式固定为价值基线乘动态调整因子：
   $$b_t = p\text{CTR}_t \times \text{CPC}_{\text{target}} \times m_t$$
   AlphaEvolve 仅负责进化 $m(state_t)$，且在系统层施加物理截断：$m_t \in [m_{\min}, m_{\max}]$（例如 MVP 设为 $[0.8, 1.2]$），确保出价始终落在历史观测支撑集（Data Support）内部；
2. **保守评估准则（Conservative Lower/Upper Bounds）**：
   在评估候选规则时，采用悲观估计准则（LCB / UCB），惩罚高不确定性区域：
   $$\text{Fitness} = \text{LCB}(\text{Clicks}) - \lambda_1 \max(0, \text{UCB}(\text{CPC}) - \text{CPC}_{\text{target}}) - \lambda_2 \text{OOD\_Ratio}$$
3. **生存分析严谨建模（Kaplan-Meier Non-parametric Estimation）**：
   利用生存分析理论，将未中标的删失样本作为正规生存时间（Survial Duration）参与对全概率分布 $P(Z \le b \mid x)$ 的无偏或渐进无偏估计；
4. **两阶段沙盒隔离（Air-Gapped Oracle Verification）**：
   搜索阶段完全处于买方受限视图（Censored View），评测通过的 Top 候选代码再进入测试集（Held-out Test Period），通过底层真实市场价进行 Oracle Replay。

---

## 3. Demo 整体架构设计 (System Architecture)

整个 Demo 系统由五个模块构成闭环，架构拓扑如下图所示：

```
                      +---------------------------------------+
                      |         Raw Benchmark Dataset         |
                      |        (iPinYou / AuctionNet)         |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |      Data Isolation & Preprocessing   |
                      +---------+-------------------+---------+
                                |                   |
             (Training & Validation Split)     (Held-out Test Split)
                                |                   |
         +----------------------v-------+           |
         | Advertiser-Side Censored View|           |
         |  - Win: Observe Price & Click|           |
         |  - Loss: Hide Market Price   |           |
         +--------------+---------------+           |
                        |                           |
         +--------------v---------------+           |
         | Market Response & Value Model|           |
         |  - CTR Predictor (pCTR)      |           |
         |  - Win Probability P(win|b)  |           |
         |  - Cost Expectation E[cost]  |           |
         +--------------+---------------+           |
                        |                           |
         +--------------v---------------+           |
         |   Learned Offline Simulator  |           |
         +--------------+---------------+           |
                        |                           |
                        | (Proxy Evaluator Feedback)|
                        v                           |
         +------------------------------+           |
         |      AlphaEvolve Engine      |           |
         |  - Population / Island Elite |           |
         |  - LLM Mutator (Gemini)      |           |
         |  - Sandbox Code Execution    |           |
         +--------------+---------------+           |
                        |                           |
               (Best Candidate Code)                |
                        |                           |
                        +------------+   +----------+
                                     |   |
                        +------------v---v--------------+
                        |      Oracle Offline A/B       |
                        |   (Replay vs Ground-Truth Z)  |
                        +---------------+---------------+
                                        |
                        +---------------v---------------+
                        |   Interactive Streamlit Demo  |
                        |  - Response Curve Comparison  |
                        |  - Evolution Trajectory       |
                        |  - Human vs Evolved Code Diff |
                        |  - Hard-Metric Head-to-Head   |
                        +-------------------------------+
```

### 3.1 模块 1：数据分流与掩蔽沙盒（Data Masking Sandbox）
* **训练集 / 演化验证集**：
  - 对数据集按时间序列切分（如前 5 天训练模型，第 6 天演化验证，第 7 天终极盲测）；
  - 对训练集和演化集进行信息脱敏与删失处理：所有赢标记录保留清算价，所有输标记录将 `market_price` 字段严格置空/删除，仅保留 `win=0`；
* **测试集（Oracle Ground Truth）**：
  - 封存第 7 天全部数据（含真实 $z_t$），AlphaEvolve 演化流程不可见。

### 3.2 模块 2：买方市场响应与价值建模（Market Response & Value Models）
* **CTR 预估模型**：
  - 输入展示特征 $x$，输出 $p\text{CTR} \in (0, 1)$。采用经典的 Logistic Regression 或 LightGBM，直接复现 iPinYou 基线，保证评估速度轻量高效；
* **市场赢标与成本响应模型（Market Response Model）**：
  - **MVP 方案（Kaplan-Meier 离散分桶估计）**：
    将流量按粗粒度 Context（如时段、广告位类型）聚类成 $K$ 个 Segment。在每个 Segment 内，统计出价阶梯的赢标与删失计数，利用 Kaplan-Meier 乘积限估计法（Product-Limit Estimator）拟合经验分布：
    $$\hat{S}(b) = \prod_{b_i \le b} \left(1 - \frac{d_i}{n_i}\right)$$
    其中 $d_i$ 为在价格 $b_i$ 处赢标的样本数，$n_i$ 为出价不低于 $b_i$ 的在险样本数（Risk Set）；从而计算赢标概率 $P(\text{win} \mid b) = 1 - \hat{S}(b)$ 以及期望二价清算成本 $E[\text{cost} \mid \text{win}, b]$；
  - **增强方案（Deep Landscape Forecasting, DLF）**：
    利用 RNN/MLP 结合生存分析损失函数（Negative Log-Likelihood with Censoring），输出每个具体请求在任意出价 $b$ 下的连续概率密度，作为高精度选配模块。

### 3.3 模块 3：AlphaEvolve 代码演化引擎 (AlphaEvolve Engine)
AlphaEvolve 不搜索神经网络权重，而是专门进化**符号化 Python 代码**。
* **演化目标**：
  搜索函数 `def get_multiplier(state: BidState) -> float` 的完整代码实现。
* **核心组成部件**：
  1. **Prompt 模板设计**：包含 RTB 领域背景、输入状态字段定义、安全乘子阈值 $[0.8, 1.2]$、演化目标函数、上一代优秀代码片段及其评测指标反馈（Clicks, CPC, Spend, Constraint Status）；
  2. **LLM 演化算子（Gemini Flash + Gemini Pro）**：
     - Gemini Flash 负责广度探索（快速产生小突变、逻辑条件组合）；
     - Gemini Pro 负责深度重构（从经济学与控制理论角度对低效代码进行结构性重写）；
  3. **种群管理（Elite Archive / MAP-Elites）**：
     - 维护容量为 $N$ 的优秀程序代码库，记录代码 AST 特征多样性（如是否包含时序统计、非线性激活、分段条件等），防止种群早熟收敛；
  4. **沙盒安全评测器（Sandbox Executor & Evaluator）**：
     - 在内存受限的隔离环境中执行生成的 Python 函数，过滤语法错误（SyntaxError）、超时、除零异常；
     - 在验证集上下文流上批量运行该函数，通过市场响应模型计算预期效果并返回 Fitness。

### 3.4 模块 4：离线 A/B 回放评测器（Oracle Offline A/B Replay）
* 将演化出的 Top 策略与基线策略置于第 7 天的真实测试集上运行；
* 对于每次展示机会，策略输出出价 $b_t$。比对真实记录中的 $z_t$：
  - 若 $b_t > z_t$：判定为赢标，累加实际点击（若原日志中该展示被点击），支付金额按二价计费即 $z_t$；
  - 若 $b_t \le z_t$：判定为未中标，不产生点击，花费为 0；
  - 实时更新累计消耗，若超过 Budget 则提前停止竞价；
* 最终输出各项指标的真实真实表现（Hard Metrics）。

### 3.5 模块 5：交互式双语演化报告（Interactive Bilingual HTML Report）
* 提供完全自包含的现代化可视化 HTML 报告，开箱即用，支持 5 页商业汇报提纲、动态演化动力学历程回放、真实盲测横向对比及代码深度机理剖析。

---

## 4. 出价策略基线与 AlphaEvolve 搜索空间

### 4.1 候选基线组合（Baselines）
为兼顾对比的严谨性与 Demo 的简洁明了，选取以下 4 个基线：
1. **Mcpc (Maximum Cost-Per-Click Heuristic)**：
   最经典的买方基准。按照期望点击价值出价：
   $$b_t = p\text{CTR}_t \times \text{CPC}_{\text{target}}$$
2. **Linear Bidding (LIN)**：
   学术界与工业界最通用的出价基准，根据相对点击率出价：
   $$b_t = b_0 \times \frac{p\text{CTR}_t}{\overline{p\text{CTR}}}$$
   其中 $b_0$ 经由验证集网格搜索选取最优参数；
3. **Human Heuristic Feedback Controller (人工反馈规则基准)**：
   模拟有经验的广告买手编写的直觉反馈逻辑（也是 AlphaEvolve 的演化 Seed）：
   ```python
   def get_multiplier(state: BidState) -> float:
       m = 1.0
       # CPC 超标则降价
       if state.current_cpc > state.target_cpc:
           m *= 0.90
       # CPC 充裕且近期赢率偏低，适当抬价冲量
       elif state.current_cpc < 0.85 * state.target_cpc and state.recent_win_rate < 0.20:
           m *= 1.08
       # 预算消耗落后于时间进度，小幅加速
       if state.budget_spend_ratio < state.time_progress_ratio - 0.10:
           m *= 1.05
       return max(0.8, min(1.2, m))
   ```
4. **Offline RL (参考 Meta KDD 2024 / IQL)**（可选高阶对照）：
   作为学术高阶对照，展示纯黑盒 RL 在有限历史样本上的效果。

### 4.2 AlphaEvolve 接口与输入状态定义
AlphaEvolve 演化的目标函数签名与输入特征严格限定在广告主可观测的物理量之内：

```python
from dataclasses import dataclass

@dataclass
class BidState:
    # 当前请求特征
    pctr: float                    # 当前展示的预估点击率
    avg_pctr: float                # 历史平均点击率
    
    # 宏观目标与进度
    target_cpc: float              # 目标 CPC 上限
    budget: float                  # 总预算
    remaining_budget_ratio: float  # 剩余预算比例 [0, 1]
    time_progress_ratio: float     # 时间进度比例 [0, 1]
    
    # 累计与微观滑动窗口指标
    current_cpc: float             # 累计至今的实际平均 CPC
    cpc_ratio: float               # current_cpc / target_cpc
    recent_win_rate: float         # 最近 N 次竞价的滑动平均赢率 (如 N=100)
    recent_cpc: float              # 最近 N 次获胜展示的平均 CPC
    spend_velocity: float          # 当前消耗速率对比理论均匀消耗速率
    last_multiplier: float         # 上一次计算出的出价乘子
```

### 4.3 演化空间与变异方向
AlphaEvolve 被允许并鼓励在以下方向进行程序发现：
* **非线性误差响应**：使用 `math.tanh`、平滑 Sigmoid 或多项式替代突兀的 `if/else` 硬阈值切换；
* **非对称调节机制（Asymmetric Response）**：对 CPC 溢出风险采取激进刹车，对 CPC 结余采用稳健缓释抬价；
* **价值敏感度门控（Value-Dependent Bidding）**：区分高价值流量（$p\text{CTR} \gg \overline{p\text{CTR}}$）与长尾流量，高价值流量给予更高的提价容忍度；
* **时序平滑与阻尼因子**：自动引入低通滤波或动量平滑机制，避免高频竞价震荡；
* **时间进度与预算消耗的动态耦合**：在投放末期自适应收窄探索区间，确保不超支。

---

## 5. 评测指标与目标函数设计 (Evaluation & Fitness Metrics)

### 5.1 适应度评估函数（Evaluator Fitness Design）
针对每次演化出的候选代码，Evaluator 计算预期收益。为彻底杜绝 CPC 违规与预算超支，采用**带罚项的分段约束函数**：

```python
def compute_fitness(metrics: dict, target_cpc: float, budget: float) -> float:
    expected_clicks = metrics["expected_clicks"]
    expected_cpc = metrics["expected_cpc"]
    expected_spend = metrics["expected_spend"]
    ood_ratio = metrics["ood_ratio"]  # 落在历史不可信区间的决策比例
    
    # 1. 预算严重超支硬淘汰
    if expected_spend > budget * 1.02:
        return -1000.0 - (expected_spend - budget)
    
    # 2. CPC 严重超标硬淘汰 (允许 1% 数值波动)
    if expected_cpc > target_cpc * 1.01:
        excess = (expected_cpc - target_cpc) / target_cpc
        return -500.0 - 1000.0 * excess
    
    # 3. OOD 样本过多硬淘汰 (防止欺骗模型)
    if ood_ratio > 0.08:
        return -200.0 - 500.0 * ood_ratio
        
    # 4. 合格策略评分：主导项为点击量，次导项鼓励预算充分利用且贴近目标 CPC
    # 注：在 CPC <= target 条件下，越充分利用预算获取点击，得分越高
    cpc_slack = max(0.0, (target_cpc - expected_cpc) / target_cpc)
    fitness = expected_clicks - 5.0 * cpc_slack  # 适度惩罚过于保守导致买量严重不足
    return fitness
```

### 5.2 核心对比与交付指标 (Evaluation Scorecard)
在最终的测试集上，所有策略对比以下 5 项核心指标：
1. **Total Clicks (总点击量)**：主要优化目标，以基线 Mcpc 为 100% 归一化；
2. **Empirical CPC (实际 CPC)**：硬约束，要求 $\le \text{CPC}_{\text{target}}$；
3. **Budget Utilization (预算利用率)**：$\frac{\sum \text{cost}_t}{\text{Budget}}$，衡量买量充分性；
4. **Constraint Violation Rate (违约率)**：若出现 $\text{CPC} > \text{CPC}_{\text{target}}$ 标记为 Failed；
5. **Simulated vs. Oracle Consistency (模拟与真值一致性)**：
   $$\Delta_{\text{sim}} = \frac{\text{Clicks}_{\text{sim}}(\text{AE}) - \text{Clicks}_{\text{sim}}(\text{Base})}{\text{Clicks}_{\text{sim}}(\text{Base})}, \quad \Delta_{\text{oracle}} = \frac{\text{Clicks}_{\text{oracle}}(\text{AE}) - \text{Clicks}_{\text{oracle}}(\text{Base})}{\text{Clicks}_{\text{oracle}}(\text{Base})}$$
   若 $\Delta_{\text{sim}} > 0$ 且 $\Delta_{\text{oracle}} > 0$，强力证明 AlphaEvolve 的真实泛化能力。

---

## 6. 数据集选取与沙盒构建细节

### 6.1 主选数据集：iPinYou RTB Benchmark (MVP 首选)
* **选用理由**：
  1. **数据体量适中**：单 Campaign 包含数十万至数百万竞价日志，训练与评估秒级/分钟级完成，非常适合流畅的实时 Demo 演示；
  2. **字段完整吻合**：原始数据包含 `bid`、`win/loss`、`click`、`paying price (z)`，天然支持构建“故意隐藏 $z$ 供买方学习，保留 $z$ 供上帝评测”的对照沙盒；
  3. **业界公认**：自 2014 年以来，作为买方侧 RTB 出价优化的金标准数据集，已被数百篇顶级学术论文（WSDM, KDD, CIKM）检验。
* **推荐 Campaign 实例**：
  选取如 `1458` 或 `3386`（典型的游戏与电商买量投放，流量充足且点击率分布合理）。

### 6.2 扩展验证集：AuctionNet (Alibaba NeurIPS 2024, Phase 2 扩展)
* **定位**：作为第二阶段扩展验证。在 iPinYou 验证完毕后，提供 AuctionNet 的 Adapter，接入其 48 个智能体环境，进一步展现算法在大规模多智能体博弈环境下的稳健性。

---

## 7. 交互式双语演化报告设计 (Interactive HTML Report Design)

采用轻量化、自包含的现代化单文件 HTML 报告架构（无需额外安装前端或后端运行环境，生成即落盘归档，并可一键托管至 GCS），划分 4 个关键视图模块：

```
+-------------------------------------------------------------------------------+
|  AlphaEvolve 广告买方侧 Target-CPC 智能出价策略优化报告                       |
+-------------------------------------------------------------------------------+
| [商业汇报提纲] 5 页交互式 Pitch Deck 幻灯片（业务痛点 / 范式革新 / 防线 / 成果 / 路线）|
+-------------------------------------------------------------------------------+
| 模块 1：演化动力学历程动态回放               | 模块 2：Day 7 真实拍卖盲测与基线对比  |
|   - 200 代时序播放控制 (从第 1 代启动)         |   - 真实点击量 (Clicks) 柱状图对比    |
|   - 散点分布与最优前沿 (Best Frontier) 跃迁   |   - 结算 CPC vs 120 元安全红线基准    |
|   - 实时候选代码探测器卡片 (Inspector)        |   - 4-Card 基线策略公式与优缺点全景拆解|
+-------------------------------------------------------------------------------+
| 模块 3：核心代码突变与 Gemini 3.8 Flash (global) 深度机理解析                 |
|   - Unified Diff 语法高亮代码差异对比视窗                                      |
|   - 非对称对数阻尼、抛物线安全闸门、三次双曲紧急制动、高 pCTR 价值收割等机制溯源|
+-------------------------------------------------------------------------------+
```

### 7.1 核心图表详细说明
1. **商业汇报提纲（Executive Pitch Deck）**：
   - 内置 5 页投影视角幻灯片，支持上一页/下一页、页码指示、缩略标签与键盘方向键翻页。
2. **演化动力学历程演示动画（Optimization Trajectory Replay）**：
   - 包含 ▶ 播放 / ⏸ 暂停、↺ 重置（严格从第 1 代启动）、单步步进、直达冠军、全景总览与 1x~10x 速度调节；
   - 随代际推进展示候选代码探索得分与历史最优前沿攀升过程。
3. **策略横向对比图表（Head-to-Head Comparison Chart）**：
   - 点击量（Clicks）与点击成本（CPC）双视图自由切换；
   - 醒目标注 120.00 RMB 目标 CPC 红色虚线硬上限基准线。
4. **代码突变与机理剖析（Code Diff & Gemini Mechanism Analysis）**：
   - 侧重展示 AlphaEvolve 的算法发现能力与白盒可解释性。

---

## 8. 分阶段落地实施路径 (Implementation Roadmap)

整个项目规划为 5 个清晰阶段（MileStones），当前处于阶段 0：

```
[Phase 0: 方案设计与环境准备] ---> [Phase 1: 数据流与响应建模] ---> [Phase 2: AlphaEvolve 闭环打通]
       (当前阶段完成)                    (离线数据与 KM 拟合)             (Prompt/LLM/Evaluator)
                                                                                  |
[Phase 4: 交互报告与客户交付] <--- [Phase 3: 演化实验与 Oracle 对比] <----------+
      (自包含HTML双语报告)            (iPinYou 批量评测与验证)
```

### Phase 0：方案制定与架构对齐（当前完成）
* 产出完整设计规范文档（`PLAN.md`）；
* 明确买方删失数据、Target-CPC 优化目标、评估隔离沙盒与演化机制。

### Phase 1：数据管道与市场响应建模
1. 下载并预处理 iPinYou 代表性 Campaign 数据（如 1458）；
2. 构建时间切分（Train / Val / Held-out Test）；
3. 实现数据掩蔽器（Data Masker），构建买方受限视图；
4. 实现轻量级 pCTR 预测器（LightGBM / LogisticRegression）；
5. 实现基于生存分析（Kaplan-Meier）的离散响应模型 $P(\text{win} \mid b, context)$ 和成本预估函数；
6. 单元测试验证：对比模拟器估算出的胜率曲线与测试集真实胜率曲线的拟合优度（$R^2$ 或 RMSE）。

### Phase 2：AlphaEvolve 演化模块研发
1. 封装 `BidState` 数据结构与出价接口规范；
2. 搭建 Python 内存沙盒执行器，支持超时保护与语法安全检查；
3. 构建基于 Learned Simulator 的离线 Evaluator；
4. 配置 Gemini API 接口，设计高信息密度的 Prompt 模板（包含前代代码、指标表现与改进建议）；
5. 实现初代种子策略（Human Rule）及基础 MAP-Elites / 种群管理池；
6. 跑通第一轮自动化“代码生成 -> 离线评估 -> 挑选优胜 -> 变异迭代”演化循环（先跑 10~20 代验证）。

### Phase 3：全量基线对比与 Oracle 真实验证
1. 实现 Mcpc、Linear、Human Controller 3 个标准基线；
2. 运行 AlphaEvolve 完整演化实验（30~50 代，生成 200~500 个候选程序）；
3. 提取 Top 3 优胜程序；
4. 在 Held-out Test 数据集上执行真实 Oracle Offline A/B 回放；
5. 验证核心假设：AlphaEvolve 选出的策略在真实市场结算下是否依然保持 $\text{CPC} \le \text{CPC}_{\text{target}}$ 且带来点击量正增长。

### Phase 4：自包含交互式演化报告交付
1. 搭建双语自包含 HTML 报告渲染核心引擎；
2. 集成离线实验结果与 200 代时序动态回放功能（原生 SVG + 硬件加速，严格从第 1 代平滑步进）；
3. 嵌入 5 页高管商业汇报提纲（Pitch Deck Mode）与 4-Card 基准策略横向对比；
4. 撰写商业汇报演讲提纲（Pitch Deck Outline）与配套白皮书；
5. 云端 GCS 自动化发布与公开只读托管。

---

## 9. 风险识别与应对策略 (Risk Analysis & Mitigation)

| 风险点 | 表现与危害 | 应对与缓解策略 |
| :--- | :--- | :--- |
| **模拟器被漏洞利用 (Simulator Exploitation)** | LLM 找到模拟器在稀疏数据区的数学漏洞，产生虚高的预测点击量 | ① 严格限制乘子范围 $m \in [0.8, 1.2]$；<br>② 评测器增加 OOD 样本惩罚，超出支持集的策略直接负分淘汰；<br>③ 采用 LCB 保守下界作为主分数。 |
| **真实市场 A/B 结果不及预期** | 离线模拟器选出的最佳策略在 Oracle 回放中 CPC 溢出或点击量下降 | ① 引入双重鲁棒估计（Doubly Robust）校准模拟偏差；<br>② 在验证集上加入模拟器与历史真实样本的一致性正则项；<br>③ 限制单次变异幅度。 |
| **LLM 演化速度与 API 延迟** | 实时生成和评测几百个候选程序耗时较长，现场演示易卡顿 | ① 演示采用“预计算实验数据回放 + 现场实时微调体验”结合的架构；<br>② 使用 Gemini Flash 进行高并发轻量探索，Pro 进行收敛精修。 |
| **客户质疑真实落地可行性** | 客户质疑“模拟再好，我线上真敢用吗？” | ① 强调输出物为**完全白盒可审计的代码**，可由买手随时人工审查或介入；<br>② 演示方案包含 1%~5% 小流量安全灰度与连续学习闭环（Model-based Continuous Learning Loop），完全契合企业安全上线规范。 |

---

## 10. 总结与下一步行动

本方案完整梳理了从广告买方现实视角出发，利用大语言模型驱动的代码进化算法（AlphaEvolve）优化 Target-CPC 自动出价策略的全套技术路径。

方案通过 **“买方删失数据学习 + 严谨受限空间演化 + 上帝视角双重盲测”** 的闭环设计，既攻克了买方无法获取对手出价的本质痛点，又展示了 AI 自动发现高级出价程序的能力。

**下一步建议行动（待用户确认后开展）：**
1. 确认上述规划细节与演化设定是否符合预期；
2. 用户确认后，启动 **Phase 1**：准备 iPinYou 样例数据集并搭建买方删失数据掩蔽与 Kaplan-Meier 市场响应模型。
