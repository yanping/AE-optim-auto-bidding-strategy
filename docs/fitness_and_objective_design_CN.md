# AlphaEvolve 广告自动出价：目标函数设计与适应度计算原理
# (Fitness Function & Optimization Objective Methodology)

[English](fitness_and_objective_design.md) | **简体中文**

> **文档定位**：本文档详细记录了本项目中**“适应度得分（Fitness Score）”的完整计算链路**，并从运筹优化、计算广告（Computational Advertising）、博弈论及大模型代码演化的视角，深度论证**“为什么不能直接以点击数作为目标函数”**。供后续复盘、学术交流及向管理层/技术评审撰写汇报材料时直接引用。

---

## 目录
- [一、 业务场景与数学形式化表达](#一-业务场景与数学形式化表达)
- [二、 适应度得分（Fitness Score）的完整计算流程](#二-适应度得分fitness-score的完整计算流程)
  - [1. 逐曝光时序仿真环境（Simulation Horizon）](#1-逐曝光时序仿真环境simulation-horizon)
  - [2. 市场响应与二阶清算结算](#2-市场响应与二阶清算结算)
  - [3. 分段带罚项的适应度函数（Fitness Piecewise Formula）](#3-分段带罚项的适应度函数fitness-piecewise-formula)
- [三、 深度辨析：为什么不能直接以“点击数”作为目标函数？](#三-深度辨析为什么不能直接以点击数作为目标函数)
  - [维度 1：有约束优化现实 vs. 无约束目标陷阱（防止流氓策略天价出价）](#维度-1有约束优化现实-vs-无约束目标陷阱防止流氓策略天价出价)
  - [维度 2：古德哈特定律与仿真器漏洞利用（防止 OOD 虚假高分）](#维度-2古德哈特定律与仿真器漏洞利用防止-ood-虚假高分)
  - [维度 3：资本边际效率——平衡“防超支”与“防保守未消耗（Under-delivery）”](#维度-3资本边际效率平衡防超支与防保守未消耗under-delivery)
  - [维度 4：运筹学外点罚函数法（Penalty Method）引导大模型搜索收敛](#维度-4运筹学外点罚函数法penalty-method引导大模型搜索收敛)
- [四、 汇报材料速查：核心对比矩阵与决策逻辑](#四-汇报材料速查核心对比矩阵与决策逻辑)
- [五、 代码实现与溯源映射](#五-代码实现与溯源映射)

---

## 一、 业务场景与数学形式化表达

在实时竞价（Real-Time Bidding, RTB）广告投放中，广告主通过 DSP 平台参与毫秒级的二阶密封拍卖（Generalized Second Price Auction）。广告主通常设定固定的预算周期（如单日预算 $`B`$）以及期望达成的目标转化成本（如单次点击成本上限 $`\mathrm{CPC}_{\mathrm{target}}`$）。

因此，自动出价代理（Auto-Bidding Agent）的核心优化任务在数学上是一个**带严格不等式约束的随机动态运筹优化问题（Constrained Stochastic Optimization Problem）**：

$$
\begin{aligned}
\max_{\pi} \quad & \sum_{t=1}^{T} \mathrm{Click}_t(b_t) \\
\text{s.t.} \quad & \sum_{t=1}^{T} \mathrm{Cost}_t(b_t) \le B \quad \text{（全局预算硬约束）} \\
& \frac{\sum_{t=1}^{T} \mathrm{Cost}_t(b_t)}{\sum_{t=1}^{T} \mathrm{Click}_t(b_t)} \le \mathrm{CPC}_{\mathrm{target}} \quad \text{（目标成本硬约束）} \\
& b_t \in \mathrm{Support}(\mathcal{D}_{\mathrm{history}}) \quad \text{（数据有效性与分布可信约束）}
\end{aligned}
$$

其中：
* $`\pi`$ 为待演化的自适应出价乘子策略函数 `get_multiplier(state) -> float`；
* $`b_t = \mathrm{CPC}_{\mathrm{target}} \times p\mathrm{CTR}_t \times 1000 \times m_t`$ 为实际竞价请求时的出价（以 CPM 为单位）；
* $`\mathrm{Click}_t \in \{0, 1\}`$，$`\mathrm{Cost}_t`$ 为第二名出价出清价格。

---

## 二、 适应度得分（Fitness Score）的完整计算流程

评估过程完全在沙盒沙箱中由 [`src/evaluate.py`](../src/evaluate.py) 与 [`src/program.py`](../src/program.py) 自动化驱动，全过程分为三个阶段：

### 1. 逐曝光时序仿真环境（Simulation Horizon）
* **评估数据集**：采用 Day 6 验证集（抽样 $`N = 5,000`$ 条独立拍卖请求），严格按时间序列流式推进；
* **物理状态构建（`BidState`）**：在每个竞价请求 $`t`$，提取广告主侧真实可观测的物理特征：
  * **请求级特征**：当前预估点击率 $`p\mathrm{CTR}_t`$、全盘历史平均点击率 $`\overline{p\mathrm{CTR}}`$；
  * **宏观进度特征**：剩余预算比例 `rem_budget_ratio` ($`\in [0, 1]`$)、时间进度比例 `time_progress_ratio` ($`\in [0, 1]`$)；
  * **消耗反馈特征**：消耗速率比值 $`\mathrm{spend\_velocity} = \frac{\mathrm{Spend}_{\mathrm{actual}}}{\mathrm{Spend}_{\mathrm{ideal}}}`$；
  * **成本控制特征**：累计点击成本 `current_cpc`、成本比例 $`\mathrm{cpc\_ratio} = \frac{\mathrm{CPC}_{\mathrm{current}}}{\mathrm{CPC}_{\mathrm{target}}}`$；
  * **滑动窗口短期统计（$`W=100`$）**：最近竞价胜率 `recent_win_rate`、最近成交均价 `recent_cpc`；
  * **上一周期动作**：上一轮出价乘子 `last_multiplier`。

### 2. 市场响应与二阶清算结算
候选策略代码 `get_multiplier(state)` 计算出乘子 $`m_t \in [0.5, 2.0]`$，折算为单次出价 $`b_t`$。随后通过离线拟合的买方市场响应模型评估：
* **胜标概率**：基于 Kaplan-Meier 乘积限生存分析模型（考虑右删失）预测 $`P(\mathrm{win} \mid b_t)`$；
* **清算成本**：二阶定价期望扣费 $`\mathbb{E}[\mathrm{Cost} \mid \mathrm{win}, b_t]`$；
* **期望新增消耗**：$`\Delta \mathrm{Spend} = \frac{\mathbb{E}[\mathrm{Cost}]}{1000} \times P(\mathrm{win} \mid b_t)`$；
* **期望新增点击**：$`\Delta \mathrm{Clicks} = p\mathrm{CTR}_t \times P(\mathrm{win} \mid b_t)`$；
* **分布外标记（OOD）**：出价 $`b_t`$ 超出历史可信出价支撑区间时计入 `ood_count`。

循环递推直到全部数据处理完毕，或因预算彻底耗尽触发提前截断（Early Termination）。

### 3. 分段带罚项的适应度函数（Fitness Piecewise Formula）
回放结束后，计算累计指标：`cum_spend`、`cum_clicks`、`final_cpc = cum_spend / cum_clicks` 以及 `ood_ratio = ood_count / N`。适应度得分按以下分段函数严格执行：

$$
\mathrm{Fitness} = 
\begin{cases} 
-1000.0 - (\mathrm{Spend}_{\mathrm{cum}} - B) & \text{若 } \mathrm{Spend}_{\mathrm{cum}} > 1.02 \cdot B \quad \text{【预算超支淘汰】} \\
-500.0 - 1000.0 \times \frac{\mathrm{CPC}_{\mathrm{final}} - \mathrm{CPC}_{\mathrm{target}}}{\mathrm{CPC}_{\mathrm{target}}} & \text{若 } \mathrm{CPC}_{\mathrm{final}} > 1.01 \cdot \mathrm{CPC}_{\mathrm{target}} \quad \text{【CPC超标淘汰】} \\
-200.0 - 500.0 \times \mathrm{Ratio}_{\mathrm{ood}} & \text{若 } \mathrm{Ratio}_{\mathrm{ood}} > 0.08 \quad \text{【OOD越界淘汰】} \\
\mathrm{Clicks}_{\mathrm{cum}} - 5.0 \times \frac{\mathrm{CPC}_{\mathrm{target}} - \mathrm{CPC}_{\mathrm{final}}}{\mathrm{CPC}_{\mathrm{target}}} & \text{其它（所有硬约束合规）} \quad \text{【合格策略得分】}
\end{cases}
$$

#### 源码映射（`src/program.py` 第 195~206 行）：
```python
# Fitness computation with hard constraint penalties (Section 5.1 of PLAN.md)
if cum_spend > budget * 1.02:
    fitness = -1000.0 - (cum_spend - budget)
elif final_cpc > target_cpc * 1.01:
    excess = (final_cpc - target_cpc) / target_cpc
    fitness = -500.0 - 1000.0 * excess
elif ood_ratio > 0.08:
    fitness = -200.0 - 500.0 * ood_ratio
else:
    cpc_slack = max(0.0, (target_cpc - final_cpc) / target_cpc)
    fitness = cum_clicks - 5.0 * cpc_slack
```

---

## 三、 深度辨析：为什么不能直接以“点击数”作为目标函数？

这是许多初涉计算广告或强化学习优化的工程师最常提出的疑问：**“广告主终极目标不就是为了买更多点击吗？为什么不直接把目标函数写成 $`\mathrm{Fitness} = \mathrm{Clicks}_{\mathrm{cum}}`$？”**

若单纯使用点击数，将直接导致四大系统性灾难：

### 维度 1：有约束优化现实 vs. 无约束目标陷阱（防止流氓策略天价出价）
* **流氓策略的形成**：如果优化器只看点击数，LLM 代码生成引擎会迅速发现一条**作弊捷径**：
  * 在程序中无视任何约束，将所有曝光的乘子死死固定在最大上限 $`m=2.0`$ 甚至更高；
  * 在前 100~200 个请求中，以全场最高价碾压所有竞对，以近乎 100% 的极高胜率迅速拿下 3~5 个点击；
  * 然后，全部预算瞬间耗尽，被迫停止竞价。
* **业务后果**：
  * 表面上看它迅速斩获了几个点击，但单次点击成本（CPC）暴涨至 **300~500 元**（目标上限仅 120 元）；
  * 这种策略若投入线上，广告主将在数小时内面临预算烧穿、ROI 击穿破产的灾难性后果。

### 维度 2：古德哈特定律与仿真器漏洞利用（防止 OOD 虚假高分）
> **古德哈特定律（Goodhart's Law）**：*“当一个指标变成目标时，它就不再是一个好指标。”*

* **离线模拟器的外推局限**：买方所依赖的 Kaplan-Meier 市场响应模型是基于历史投放日志（含右删失数据）构建的。其有效性仅存在于历史出价的经验支撑区间内；
* **代码演化对抗攻击**：若纯粹最大化点击数，大模型演化算子会像寻找“越狱漏洞”一样，生成特殊的出价公式，专门寻找离线模拟器在极值盲区处的数学漏洞（如某些高价区间的概率外推溢出）；
* **虚假繁荣与真实暴跌**：该策略会在仿真器中骗取虚假的超高“期望点击数”，但在进入真实未脱敏的 Day 7 真实拍卖市场（Oracle Ground Truth）时，真实出清价格会立刻重罚这种盲目出价，导致实际效果大幅崩盘；
* **防作弊屏障**：必须加入 `ood_ratio > 0.08` 的断崖扣分（`-200` 分），切断 LLM 寻找模型死角的投机空间，迫使其在真实可信的价格区间内探寻运筹解。

### 维度 3：资本边际效率——平衡“防超支”与“防保守未消耗（Under-delivery）”
在真实的广告平台运营中，**“超支是事故，严重欠消耗（Under-delivery）同样是不可接受的事故”**：
* 如果只有硬惩罚而无引导，策略很容易演化成“惊弓之鸟”——为了绝对不超标，将出价乘子全面压到最低（$`m=0.5`$），只买最底层、最便宜的长尾流量；
* 最终全天仅花掉 5% 的预算，实际 CPC 极低（如 10 元），但全天只买到了 0.2 个点击，导致广告主计划投放的促销拉新目标全部泡汤；
* **松弛度项的精妙调节**：

$$
\mathrm{Slack}_{\mathrm{cpc}} = \frac{\mathrm{CPC}_{\mathrm{target}} - \mathrm{CPC}_{\mathrm{final}}}{\mathrm{CPC}_{\mathrm{target}}}
$$

  公式中引入 $`-5.0 \times \mathrm{Slack}_{\mathrm{cpc}}`$，相当于在经济学上引入了**“资本闲置机会成本”**：
  * 当策略合规且保本时，若 CPC 远低于目标（表明安全裕量极为充裕），松弛惩罚会驱使策略适度激进放量；
  * 推动出价乘子主动向目标 CPC 靠拢，把预算花在刀刃上，换取更多的总点击规模。

### 维度 4：运筹学外点罚函数法（Penalty Method）引导大模型搜索收敛
AlphaEvolve 背后的演化循环与大语言模型符号变异，本质上属于**离散代码空间中的无约束全局搜索器**。
* **罚函数法的运筹学数学根基**：在非凸、无梯度的代码空间中，处理复杂的强约束优化问题，最成熟稳健的手段正是**外点罚函数法（Penalty Function Method）**；
* **惩罚绝壁（Penalty Cliffs）的塑造**：
  * 通过在可行域外设置大梯度负分（`-1000`、`-500`、`-200`），我们在适应度地形图边缘构筑了高耸入云的“惩罚绝壁”；
  * **第一阶段（代际 1 ~ 20）**：LLM 迅速收到诊断洞察（如 `Budget Overrun`、`Target CPC Exceeded`）与巨额负分，迅速剔除所有不守规矩的粗糙代码，强迫种群迅速向合规可行域（Feasible Region）收敛；
  * **第二阶段（代际 20 ~ 200）**：种群一旦进入可行域，负罚项归零，目标函数平滑切换为以点击量为主导、CPC 充分度为辅的精细优化，引导冠军策略向帕累托最优前沿（Pareto Frontier）稳步演进。

---

## 四、 汇报材料速查：核心对比矩阵与决策逻辑

在向上汇报或撰写分析报告时，可直接引用下述表格与要点：

### 1. 目标函数方案选型对比表

| 候选目标方案 | 目标函数数学形式 | 演化收敛表征 | 真实测试集（Day 7 Oracle）落地表现 | 结论 |
| :--- | :--- | :--- | :--- | :--- |
| **方案 A：朴素点击量目标** | $`\mathrm{Fitness} = \mathrm{Clicks}_{\mathrm{cum}}`$ | 策略无脑顶格提价，利用模拟器极值盲区刷分 | 预算数小时烧空，实际 CPC 严重超标（>300 RMB），线上不可用 | ❌ **严重不可行** |
| **方案 B：仅成本达标目标** | $`\mathrm{Fitness} = - \lvert \mathrm{CPC}_{\mathrm{final}} - \mathrm{CPC}_{\mathrm{target}} \rvert`$ | 策略过度保守，只买极低成本流量 | 预算消耗不足 10%（严重欠量），点击数极低，商业目标破产 | ❌ **业务不可用** |
| **方案 C：分段罚函数复合适应度（本项目）** | **分段绝壁惩罚 + 点击主导 + 松弛激励** | 前 20 代迅速进入合规域，后 180 代实现非线性控制平滑跃升 | **Oracle 真实点击提升 +16.6%，CPC 78.76 RMB 严守 120 RMB 上限，100% 履约** |  **最优工业落地解** |

### 2. 最终收益与汇报亮点
* **适应度演化成效**：从初始人工经验种子的 **0.2088** 跃升至 **1.3520**（提升 **+547%**）；
* **真实 Oracle 商业成果**：
  * 真实点击量从 251 跃升至 **288**（较人工规则提升 **+14.7%**，较调优线性基准提升 **+2.5%**）；
  * 实际 CPC 控制在 **78.76 RMB**，远低于 **120.00 RMB** 的安全上限；
  * 真正实现了在严苛物理与商业约束下**多拿流量、不超预算、守住成本**的多目标协同优化。

---

## 五、 代码实现与溯源映射

如需在代码中查阅或复现具体逻辑，可参考以下相对路径文件：

1. **核心适应度计算实现**：
   * 仿真与分段惩罚：[`src/program.py` (第 85-214 行)](../src/program.py#L85-L214)
   * 适应度函数封装与诊断信息输出：[`src/evaluate.py` (第 134-258 行)](../src/evaluate.py#L134-L258)
2. **自动化测试与断言验证**：
   * 候选策略执行与异常打分用例：[`tests/test_evolution.py`](../tests/test_evolution.py)
   * 出价状态与乘子边界用例：[`tests/test_bidding.py`](../tests/test_bidding.py)
3. **真实测试集 Oracle 验证实现**：
   * 二阶出清与基准对比：[`src/bidding/benchmark.py`](../src/bidding/benchmark.py)
   * 自动化报表渲染（含 Gemini 机理剖析）：[`src/report.py`](../src/report.py)
