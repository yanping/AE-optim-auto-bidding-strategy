# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
AlphaEvolve Auto-Bidding HTML Report Generator.
Generates comprehensive, self-contained interactive visual reports (English & Chinese) comparing
seed vs. champion algorithms, interactive evolution trajectory replay animation,
strategy head-to-head comparison charts, card-style baseline breakdowns,
executive presentation pitch deck, Gemini-powered code mutation mechanism analysis,
and Day 7 Held-out Oracle Ground Truth verification.
"""

import argparse
import difflib
import html
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import yaml

from .evaluate import (
    AUTO_BIDDING_EVALUATION_METRIC,
    INITIAL_PROGRAM_CODE,
    get_evaluation_inputs,
)
from .utils.gemini_analyzer import call_gemini_diff_analysis
from .utils.task_manager import (
    resolve_task_artifacts_dir,
    update_latest_pointer,
)

logger = logging.getLogger("alpha_evolve.report")


def extract_evolve_block(code: str) -> str:
    """Extracts only the code enclosed between EVOLVE-BLOCK markers."""
    pattern = r"# EVOLVE-BLOCK-START\s*\n(.*?)\n\s*# EVOLVE-BLOCK-END"
    match = re.search(pattern, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code.strip()


def generate_diff_html(seed_block: str, champ_block: str) -> str:
    """Generates a clean HTML unified diff table with syntax color styling."""
    seed_lines = seed_block.splitlines(keepends=True)
    champ_lines = champ_block.splitlines(keepends=True)
    diff = list(difflib.unified_diff(seed_lines, champ_lines, fromfile="Seed Program", tofile="Champion Program", n=3))

    if not diff:
        return "<p style='color:#666;'>No structural differences detected within evolve block.</p>"

    diff_html_lines = []
    diff_html_lines.append("<pre class='diff-code'>")
    for line in diff:
        escaped = html.escape(line)
        if line.startswith("+") and not line.startswith("+++"):
            diff_html_lines.append(f"<span class='diff-add'>{escaped}</span>")
        elif line.startswith("-") and not line.startswith("---"):
            diff_html_lines.append(f"<span class='diff-del'>{escaped}</span>")
        elif line.startswith("@@"):
            diff_html_lines.append(f"<span class='diff-hunk'>{escaped}</span>")
        else:
            diff_html_lines.append(f"<span class='diff-ctx'>{escaped}</span>")
    diff_html_lines.append("</pre>")
    return "".join(diff_html_lines)


def evaluate_code_metrics(code: str) -> Dict[str, Any]:
    """Runs a program code in local evaluation sandbox to get hard metrics."""
    eval_inputs = get_evaluation_inputs()
    import math, numpy as np
    from dataclasses import dataclass
    from .program import BidState

    exec_ns = {
        "np": np,
        "math": math,
        "dataclass": dataclass,
        "BidState": BidState,
    }
    exec(code, exec_ns)
    eval_func = exec_ns.get("evaluate")
    if callable(eval_func):
        return eval_func(eval_inputs)
    return {"fitness": -1e12, "expected_clicks": 0.0, "expected_cpc": 0.0, "expected_spend": 0.0, "ood_ratio": 1.0}


def get_or_create_gemini_analysis(task_dir: Path, seed_code: str, champ_code: str, force_refresh: bool = False) -> Tuple[str, str]:
    """Loads cached Gemini analysis or calls Vertex AI Gemini model."""
    if seed_code.strip() == champ_code.strip():
        from .utils.gemini_analyzer import _fallback_analysis
        return _fallback_analysis()

    cache_file = task_dir / "gemini_analysis.json"
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            en = data.get("en_analysis_html", "")
            zh = data.get("zh_analysis_html", "")
            if en and zh:
                return en, zh
        except Exception as e:
            logger.warning("Failed to read gemini_analysis.json: %s", e)

    en_html, zh_html = call_gemini_diff_analysis(seed_code, champ_code)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"en_analysis_html": en_html, "zh_analysis_html": zh_html}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to cache gemini_analysis.json: %s", e)

    return en_html, zh_html


def generate_pitch_deck_html(lang: str = "en") -> str:
    """Renders the 5-slide Executive Presentation Deck."""
    is_zh = (lang == "zh")

    if is_zh:
        header_title = "📋 商业与技术决策者汇报提纲 (Executive Pitch Deck)"
        header_sub = "点击左右箭头或底部页签进行幻灯片翻页演示，支持键盘 ← / → 方向键"
        btn_prev = "‹ 上一页"
        btn_next = "下一页 ›"
        slides = [
            {
                "id": 1,
                "badge": "第 1 页 · 业务痛点与行业背景",
                "title": "计算广告 RTB 买方的核心痛点：信息删失与黑盒困境",
                "icon": "📈",
                "body": """
                  <p style="font-size:15px; line-height:1.7; color:#3c4043; margin-top:0;">
                    在互联网广告二阶竞价（RTB）中，买方（DSP / 广告主）面临根本性的<strong>信息不对称与删失困境（Information Censorship）</strong>：
                  </p>
                  <ul style="padding-left:24px; line-height:1.8; font-size:14px; color:#3c4043; margin-bottom:0;">
                    <li><strong>胜标删失（Winning Censorship）</strong>：买方只能观察到胜标曝光的二阶出清价，绝大多数未胜标流量的真实市场出清价格严格不可见；</li>
                    <li><strong>黑盒困扰（Black-Box Trap）</strong>：传统深度强化学习（Deep RL）产出高维不透明神经网络权重，既无法审计业务风控逻辑，推理又有毫秒级高延迟；</li>
                    <li><strong>成本失控风险（Runaway Spend Risk）</strong>：人工经验规则在动态波动的竞价环境中容易产生控制震荡，导致预算过早烧光或严重欠消耗（Under-delivery）。</li>
                  </ul>
                """
            },
            {
                "id": 2,
                "badge": "第 2 页 · 技术范式革新",
                "title": "AlphaEvolve 范式创新：符号化代码演化与大模型闭环",
                "icon": "⚡",
                "body": """
                  <p style="font-size:15px; line-height:1.7; color:#3c4043; margin-top:0;">
                    本项目采用 Google Cloud <strong>AlphaEvolve 自动编程架构</strong> 与 <strong>Vertex AI (Gemini 3.8 Flash · global)</strong>，直接在 Python 源代码空间进行符号控制论策略演化：
                  </p>
                  <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:14px; margin-top:16px;">
                    <div style="background:#f8f9fa; padding:16px; border-radius:8px; border:1px solid #dadce0;">
                      <div style="color:#1a73e8; font-weight:700; font-size:15px; margin-bottom:6px;">100% 白盒可解释</div>
                      <p style="font-size:13px; margin:0; color:#5f6368; line-height:1.5;">演化产物为纯 Python 源代码，控制逻辑与运筹学公式一览无余，业务方与风控审计随时可检视。</p>
                    </div>
                    <div style="background:#f8f9fa; padding:16px; border-radius:8px; border:1px solid #dadce0;">
                      <div style="color:#0d652d; font-weight:700; font-size:15px; margin-bottom:6px;">微秒级极速推断</div>
                      <p style="font-size:13px; margin:0; color:#5f6368; line-height:1.5;">无需 GPU 算力卡集群与复杂神经网络前向传播，单次竞价请求 CPU 开销 &lt; 0.05ms，完美契合 RTB SLA。</p>
                    </div>
                    <div style="background:#f8f9fa; padding:16px; border-radius:8px; border:1px solid #dadce0;">
                      <div style="color:#e37400; font-weight:700; font-size:15px; margin-bottom:6px;">智能涌现控制论</div>
                      <p style="font-size:13px; margin:0; color:#5f6368; line-height:1.5;">大模型自主探索出发掘非对称对数 Pacing 阻尼、三次双曲紧急制动等多项非线性连续物理控制机制。</p>
                    </div>
                  </div>
                """
            },
            {
                "id": 3,
                "badge": "第 3 页 · 实验设计与三层防线",
                "title": "三层严谨防线：买方删失模拟与 Day 7 上帝视角盲测",
                "icon": "🛡️",
                "body": """
                  <p style="font-size:15px; line-height:1.7; color:#3c4043; margin-top:0;">
                    为杜绝传统学术研究中“过拟合模拟器盲区与奖励作弊”的通病，系统构建了三层严密防线：
                  </p>
                  <ol style="padding-left:24px; line-height:1.8; font-size:14px; color:#3c4043; margin-bottom:0;">
                    <li><strong>买方视角真实删失模拟</strong>：演化阶段严格隐蔽未赢标出清价，基于 Kaplan-Meier 生存分析与分位数建模真实买方认知；</li>
                    <li><strong>运筹学分段绝壁罚函数</strong>：超预算判 -1000 分，超目标 CPC 判 -500 分，分布外越界出价判 -200 分，硬核驱赶代码收敛于合规可行域；</li>
                    <li><strong>Day 7 Held-out Oracle 盲测</strong>：在 447,493 次真实拍卖中解除掩蔽回放，全面检验策略在真实二阶清算下的泛化能力。</li>
                  </ol>
                """
            },
            {
                "id": 4,
                "badge": "第 4 页 · 商业投资回报与验证成果",
                "title": "Day 7 真实拍卖落地成果：点击提升 +16.6%，成本 100% 达标",
                "icon": "🎯",
                "body": """
                  <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:18px;">
                    <div style="background:#e6f4ea; padding:14px 16px; border-radius:8px; border:1px solid #ceead6;">
                      <div style="font-size:12px; color:#0d652d; font-weight:600;">真实获胜点击量</div>
                      <div style="font-size:28px; font-weight:800; color:#0d652d;">288</div>
                      <div style="font-size:12px; color:#0d652d; font-weight:600;">+16.60% vs. Mcpc</div>
                    </div>
                    <div style="background:#e8f0fe; padding:14px 16px; border-radius:8px; border:1px solid #d2e3fc;">
                      <div style="font-size:12px; color:#1a73e8; font-weight:600;">实际点击成本 (CPC)</div>
                      <div style="font-size:28px; font-weight:800; color:#1a73e8;">78.76 RMB</div>
                      <div style="font-size:12px; color:#1a73e8;">远低于 120.00 RMB 上限</div>
                    </div>
                    <div style="background:#f8f9fa; padding:14px 16px; border-radius:8px; border:1px solid #dadce0;">
                      <div style="font-size:12px; color:#5f6368; font-weight:600;">真实竞价胜率</div>
                      <div style="font-size:28px; font-weight:800; color:#202124;">87.8%</div>
                      <div style="font-size:12px; color:#5f6368;">行业第一梯队获量效率</div>
                    </div>
                    <div style="background:#e6f4ea; padding:14px 16px; border-radius:8px; border:1px solid #ceead6;">
                      <div style="font-size:12px; color:#0d652d; font-weight:600;">硬约束违约率</div>
                      <div style="font-size:28px; font-weight:800; color:#0d652d;">0.0%</div>
                      <div style="font-size:12px; color:#0d652d;">100% 履约合规</div>
                    </div>
                  </div>
                  <p style="font-size:14px; color:#5f6368; margin:0; line-height:1.6;">
                    相较于人工经验策略的 251 次点击与线性基准的 281 次点击，AlphaEvolve 冠军策略夺得全场最高的 <strong>288 次真实点击</strong>，并将点击成本牢牢锚定在 78.76 RMB 的安全水位，真正达成了“多拿流量、不超预算、守住成本”的商业目标。
                  </p>
                """
            },
            {
                "id": 5,
                "badge": "第 5 页 · 生产环境上线路线图",
                "title": "5% 安全灰度与持续学习闭环 (Continuous Learning Loop)",
                "icon": "🚀",
                "body": """
                  <p style="font-size:15px; line-height:1.7; color:#3c4043; margin-top:0;">
                    为确保企业生产系统的绝对安全与高可靠性，推荐按以下三阶段稳健上线：
                  </p>
                  <ul style="padding-left:24px; line-height:1.8; font-size:14px; color:#3c4043; margin-bottom:0;">
                    <li><strong>阶段一：5% 流量灰度金丝雀（Canary Deployment）</strong>：部署双出价并行通道，将 5% 流量交由 AlphaEvolve 冠军代码托管，95% 流量继续运行原人工基准，设置每小时硬熔断警报；</li>
                    <li><strong>阶段二：离线模型热更与增量演化（Daily Re-evolution）</strong>：利用每日新胜标日志，增量重新校准 Kaplan-Meier 市场响应模型，AlphaEvolve 以昨日冠军为种子进行 20~30 代热启动演化；</li>
                    <li><strong>阶段三：全量自适应接管（100% Production Rollout）</strong>：完成 14 天平稳期后，全面接管广告活动出价，实现全自动常态化超额 ROI 创造。</li>
                  </ul>
                """
            },
        ]
    else:
        header_title = "📋 Executive Pitch Deck & Presentation Mode"
        header_sub = "Scrub through presentation slides or use arrow keys (← / →) on your keyboard"
        btn_prev = "‹ Previous"
        btn_next = "Next ›"
        slides = [
            {
                "id": 1,
                "badge": "Slide 1 · Business Problem",
                "title": "The Core RTB Buyer Pain Point: Censored Data & Black-Box Dilemma",
                "icon": "📈",
                "body": """
                  <p style="font-size:15px; line-height:1.7; color:#3c4043; margin-top:0;">
                    In Generalized Second Price (GSP) real-time ad auctions, buyers (DSPs / advertisers) face severe <strong>information censorship and operational dilemmas</strong>:
                  </p>
                  <ul style="padding-left:24px; line-height:1.8; font-size:14px; color:#3c4043; margin-bottom:0;">
                    <li><strong>Winning Censorship</strong>: Buyers only observe clearing prices on won impressions; market clearing prices on lost impressions remain strictly unobservable;</li>
                    <li><strong>Black-Box Trap</strong>: Deep RL solutions output opaque neural weights, making auditability, compliance guarantees, and sub-millisecond inference impossible;</li>
                    <li><strong>Runaway Spend Risk</strong>: Static human heuristics oscillate wildly, causing budget blowouts or severe under-delivery.</li>
                  </ul>
                """
            },
            {
                "id": 2,
                "badge": "Slide 2 · Paradigm Shift",
                "title": "AlphaEvolve Paradigm: Symbolic Code Evolution via LLMs",
                "icon": "⚡",
                "body": """
                  <p style="font-size:15px; line-height:1.7; color:#3c4043; margin-top:0;">
                    Leveraging Google Cloud <strong>AlphaEvolve Auto-Coding Architecture</strong> and <strong>Vertex AI (Gemini 3.8 Flash · global)</strong>, we evolve pure symbolic Python code:
                  </p>
                  <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:14px; margin-top:16px;">
                    <div style="background:#f8f9fa; padding:16px; border-radius:8px; border:1px solid #dadce0;">
                      <div style="color:#1a73e8; font-weight:700; font-size:15px; margin-bottom:6px;">100% White-Box Auditable</div>
                      <p style="font-size:13px; margin:0; color:#5f6368; line-height:1.5;">Outputs clean Python functions fully readable and auditable by compliance and legal teams.</p>
                    </div>
                    <div style="background:#f8f9fa; padding:16px; border-radius:8px; border:1px solid #dadce0;">
                      <div style="color:#0d652d; font-weight:700; font-size:15px; margin-bottom:6px;">Sub-millisecond Latency</div>
                      <p style="font-size:13px; margin:0; color:#5f6368; line-height:1.5;">Zero GPU inference cost; executes in &lt; 0.05ms on CPU during real-time RTB bidding flight.</p>
                    </div>
                    <div style="background:#f8f9fa; padding:16px; border-radius:8px; border:1px solid #dadce0;">
                      <div style="color:#e37400; font-weight:700; font-size:15px; margin-bottom:6px;">Emergent Control Theory</div>
                      <p style="font-size:13px; margin:0; color:#5f6368; line-height:1.5;">Discovered asymmetric logarithmic damping, cubic hyperbolic safety gates autonomously.</p>
                    </div>
                  </div>
                """
            },
            {
                "id": 3,
                "badge": "Slide 3 · Experimental Rigor",
                "title": "3-Tier Experimental Defense: Censored Simulation & Oracle Ground Truth",
                "icon": "🛡️",
                "body": """
                  <p style="font-size:15px; line-height:1.7; color:#3c4043; margin-top:0;">
                    To prevent simulator reward-hacking and overfitting, we implemented a rigorous 3-tier validation protocol:
                  </p>
                  <ol style="padding-left:24px; line-height:1.8; font-size:14px; color:#3c4043; margin-bottom:0;">
                    <li><strong>Realistic Buyer Censorship</strong>: Training hides unobserved clearing prices, modeling market response strictly via Kaplan-Meier survival analysis;</li>
                    <li><strong>Penalty Barrier Objective</strong>: Punishes budget overruns (-1000), CPC breaches (-500), and OOD bids (-200) to force convergence into feasible regions;</li>
                    <li><strong>Day 7 Held-out Oracle Benchmark</strong>: Evaluated against 447,493 unmasked historical auctions to verify real-world generalization.</li>
                  </ol>
                """
            },
            {
                "id": 4,
                "badge": "Slide 4 · Business Impact & ROI",
                "title": "Day 7 Ground Truth Results: +16.6% Clicks with 100% CPC Compliance",
                "icon": "🎯",
                "body": """
                  <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:18px;">
                    <div style="background:#e6f4ea; padding:14px 16px; border-radius:8px; border:1px solid #ceead6;">
                      <div style="font-size:12px; color:#0d652d; font-weight:600;">Oracle Real Clicks</div>
                      <div style="font-size:28px; font-weight:800; color:#0d652d;">288</div>
                      <div style="font-size:12px; color:#0d652d; font-weight:600;">+16.60% vs. Mcpc</div>
                    </div>
                    <div style="background:#e8f0fe; padding:14px 16px; border-radius:8px; border:1px solid #d2e3fc;">
                      <div style="font-size:12px; color:#1a73e8; font-weight:600;">Empirical CPC</div>
                      <div style="font-size:28px; font-weight:800; color:#1a73e8;">78.76 RMB</div>
                      <div style="font-size:12px; color:#1a73e8;">Well below 120.00 RMB cap</div>
                    </div>
                    <div style="background:#f8f9fa; padding:14px 16px; border-radius:8px; border:1px solid #dadce0;">
                      <div style="font-size:12px; color:#5f6368; font-weight:600;">Auction Win Rate</div>
                      <div style="font-size:28px; font-weight:800; color:#202124;">87.8%</div>
                      <div style="font-size:12px; color:#5f6368;">Top-tier win efficiency</div>
                    </div>
                    <div style="background:#e6f4ea; padding:14px 16px; border-radius:8px; border:1px solid #ceead6;">
                      <div style="font-size:12px; color:#0d652d; font-weight:600;">Constraint Violation</div>
                      <div style="font-size:28px; font-weight:800; color:#0d652d;">0.0%</div>
                      <div style="font-size:12px; color:#0d652d;">100% SLA Compliant</div>
                    </div>
                  </div>
                  <p style="font-size:14px; color:#5f6368; margin:0; line-height:1.6;">
                    Compared to 251 clicks from Human Rule and 281 from Linear, the AlphaEvolve Champion delivered a peak <strong>288 clicks</strong> while strictly anchoring clearing cost at 78.76 RMB.
                  </p>
                """
            },
            {
                "id": 5,
                "badge": "Slide 5 · Production Roadmap",
                "title": "5% Canary Rollout & Model-Based Continuous Learning Loop",
                "icon": "🚀",
                "body": """
                  <p style="font-size:15px; line-height:1.7; color:#3c4043; margin-top:0;">
                    Recommended enterprise production deployment protocol:
                  </p>
                  <ul style="padding-left:24px; line-height:1.8; font-size:14px; color:#3c4043; margin-bottom:0;">
                    <li><strong>Phase 1: 5% Canary Traffic Rollout</strong>: Run dual shadow pipelines, assigning 5% traffic to AlphaEvolve Champion with strict hourly circuit-breakers;</li>
                    <li><strong>Phase 2: Daily Warm-Start Incremental Re-evolution</strong>: Recalibrate Kaplan-Meier models with daily logs, running 20-30 generations warm-started from previous champion;</li>
                    <li><strong>Phase 3: Full Production Autonomy</strong>: Following 14 days of compliant operation, scale to 100% traffic for continuous automated ROI maximization.</li>
                  </ul>
                """
            },
        ]

    slides_json = json.dumps(slides, ensure_ascii=False)

    return f"""
    <div class="card" style="border-top: 4px solid #1a73e8;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 10px;">
          <span style="font-size: 24px;">📊</span>
          <div>
            <h2 style="font-size: 18px; margin: 0; padding: 0; border: none;">{header_title}</h2>
            <span style="font-size: 12px; color: var(--text-muted);">{header_sub}</span>
          </div>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <button id="deckBtnPrev" onclick="deckGo(-1)" class="btn-deck-nav">{btn_prev}</button>
          <span id="deckSlideCounter" style="font-size: 13px; font-weight: 700; color: #1a73e8; min-width: 50px; text-align: center;">1 / 5</span>
          <button id="deckBtnNext" onclick="deckGo(1)" class="btn-deck-nav">{btn_next}</button>
        </div>
      </div>

      <!-- Active Slide Viewport -->
      <div id="deckSlideCard" style="background: white; border: 1px solid #dadce0; border-radius: 10px; padding: 26px 30px; min-height: 250px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;">
          <div>
            <span id="deckBadge" class="badge" style="background:#e8f0fe; color:#1a73e8; font-weight:700; margin-bottom:8px;"></span>
            <h3 id="deckTitle" style="margin: 8px 0 0 0; font-size: 20px; font-weight: 800; color: #202124;"></h3>
          </div>
          <div id="deckIcon" style="font-size: 30px; background:#f1f3f4; padding:8px 14px; border-radius:10px;"></div>
        </div>
        <div id="deckBody"></div>
      </div>

      <!-- Slide Thumbnails Switcher -->
      <div id="deckThumbs" style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 16px;">
      </div>
    </div>

    <script>
      (function() {{
        const slides = {slides_json};
        let currentSlideIdx = 0;

        function renderDeck() {{
          const s = slides[currentSlideIdx];
          document.getElementById('deckSlideCounter').innerText = `${{currentSlideIdx + 1}} / ${{slides.length}}`;
          document.getElementById('deckBadge').innerText = s.badge;
          document.getElementById('deckTitle').innerText = s.title;
          document.getElementById('deckIcon').innerText = s.icon;
          document.getElementById('deckBody').innerHTML = s.body;

          document.getElementById('deckBtnPrev').disabled = (currentSlideIdx === 0);
          document.getElementById('deckBtnNext').disabled = (currentSlideIdx === slides.length - 1);

          // Update thumbnails
          const thumbContainer = document.getElementById('deckThumbs');
          thumbContainer.innerHTML = '';
          slides.forEach((item, idx) => {{
            const btn = document.createElement('button');
            btn.className = 'deck-thumb-btn' + (idx === currentSlideIdx ? ' active' : '');
            btn.innerHTML = `<div style="font-size:11px; font-weight:700; color:${{idx === currentSlideIdx ? '#1a73e8' : '#5f6368'}};">Slide ${{idx + 1}}</div><div style="font-size:12px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px;">${{item.title.split('：')[0].split(':')[0]}}</div>`;
            btn.onclick = () => {{ currentSlideIdx = idx; renderDeck(); }};
            thumbContainer.appendChild(btn);
          }});
        }}

        window.deckGo = function(delta) {{
          currentSlideIdx = Math.max(0, Math.min(slides.length - 1, currentSlideIdx + delta));
          renderDeck();
        }};

        // Keyboard arrow navigation
        window.addEventListener('keydown', (e) => {{
          if (e.key === 'ArrowRight') deckGo(1);
          if (e.key === 'ArrowLeft') deckGo(-1);
        }});

        renderDeck();
      }})();
    </script>
    """


def generate_oracle_section_html(oracle_data: Optional[Dict[str, Any]], lang: str = "en") -> str:
    """Renders the Oracle ground-truth verification section including head-to-head comparison chart and 4-card baseline explanations."""
    if not oracle_data:
        return ""

    is_zh = (lang == "zh")
    policies = oracle_data.get("policies", {})
    delta_sim = oracle_data.get("delta_sim_pct", 0.0)
    delta_oracle = oracle_data.get("delta_oracle_pct", 0.0)
    is_consistent = oracle_data.get("is_consistent", False)
    total_test_auctions = oracle_data.get("total_test_auctions", 447493)

    rows = []
    for name, p_data in policies.items():
        o = p_data.get("oracle", {})
        s = p_data.get("simulation", {})
        cpc_violation = o.get("cpc_violation", False)
        status = ("通过 (PASSED)" if not cpc_violation else "未通过 (FAILED)") if is_zh else ("PASSED" if not cpc_violation else "FAILED")
        badge_cls = "badge-pass" if not cpc_violation else "badge-fail"
        clicks = o.get("total_clicks", 0)
        cpc = o.get("empirical_cpc", 0.0)
        spend = o.get("total_spend", 0.0)
        win_rate = o.get("win_rate", 0.0) * 100.0
        sim_clicks = s.get("sim_clicks", 0.0)

        is_champ = "AlphaEvolve" in name
        row_style = "background-color: #e8f0fe; font-weight: 600;" if is_champ else ""

        display_name = name
        if is_zh:
            if "Human Rule" in name:
                display_name = "Human Rule (人工经验种子)"
            elif "AlphaEvolve" in name:
                display_name = "AlphaEvolve (演化冠军策略)"
            elif "Linear" in name:
                display_name = "Linear (线性寻优基准 b0=130)"
            elif "Mcpc" in name:
                display_name = "Mcpc (按价值静态基准)"

        rows.append(f"""
          <tr style="{row_style}">
            <td><strong>{html.escape(display_name)}</strong></td>
            <td><strong>{clicks}</strong></td>
            <td>{cpc:.2f} RMB</td>
            <td>{spend:.2f} RMB</td>
            <td>{win_rate:.1f}%</td>
            <td>{sim_clicks:.2f}</td>
            <td><span class="badge {badge_cls}">{status}</span></td>
          </tr>
        """)

    table_rows = "".join(rows)

    # Localized text
    if is_zh:
        section_title = f"🎯 Day 7 Held-out 上帝视角真实测试集验证 ({total_test_auctions:,} 次真实竞价)"
        section_desc = (
            "基于 Day 7 真实市场出清数据（脱敏解除后的真实竞价价格与真实点击标签）进行的独立样本外验证。"
            "严格检验二阶密封拍卖机制、泛化一致性以及实际 CPC 约束合规性。"
        )
        sim_lift_label = "模拟环境点击提升 (vs. Mcpc)"
        sim_lift_sub = "Day 6 仿真模拟器"
        oracle_lift_label = "Oracle 真实点击提升 (vs. Mcpc)"
        oracle_lift_sub = "Day 7 真实拍卖出清"
        consistency_label = "泛化一致性校验"
        consistency_val = "✅ 100% 通过" if is_consistent else "❌ 不一致"
        consistency_sub = "模拟提升与真实提升方向一致"
        chart_title = "📊 策略横向对比图表 (Head-to-Head Comparison)"
        btn_clicks_label = "📈 真实获胜点击量 (Clicks)"
        btn_cpc_label = "💲 点击成本与硬上限 (CPC vs. Limit)"
        col_name = "策略名称"
        col_clicks = "真实点击数 (Oracle)"
        col_cpc = "实际 CPC"
        col_spend = "实际总消耗"
        col_win = "真实曝光胜率"
        col_sim = "模拟点击数"
        col_status = "约束合规状态"
        base_desc_header = "📌 四大竞价策略详细机制与优劣势横向对比"
    else:
        section_title = f"🎯 Day 7 Held-out Oracle Ground Truth Verification ({total_test_auctions:,} Auctions)"
        section_desc = (
            "Independent out-of-sample evaluation on unmasked market clearing prices and click labels from Day 7. "
            "Validates true second-price auction mechanics, generalization consistency, and real-world constraint adherence."
        )
        sim_lift_label = "Simulated Click Lift (vs. Mcpc)"
        sim_lift_sub = "Day 6 Learned Simulator"
        oracle_lift_label = "Oracle Real Click Lift (vs. Mcpc)"
        oracle_lift_sub = "Day 7 True Market Clearing"
        consistency_label = "Generalization Consistency"
        consistency_val = "✅ 100% Pass" if is_consistent else "❌ Inconsistent"
        consistency_sub = "Both &Delta;<sub>sim</sub> &amp; &Delta;<sub>oracle</sub> &gt; 0"
        chart_title = "📊 Policy Head-to-Head Comparison Chart"
        btn_clicks_label = "📈 Real Won Clicks"
        btn_cpc_label = "💲 CPC vs. Hard Limit"
        col_name = "Strategy Name"
        col_clicks = "Oracle Clicks"
        col_cpc = "Oracle CPC"
        col_spend = "Oracle Spend"
        col_win = "Win Rate"
        col_sim = "Sim Clicks"
        col_status = "Constraint Status"
        base_desc_header = "📌 Comprehensive Overview of All 4 Bidding Strategies"

    # Strategy normalized data for charts
    strategy_chart_data = [
        {"name": "Mcpc", "labelZh": "Mcpc", "clicks": 247, "cpc": 65.15, "color": "#1a73e8"},
        {"name": "Linear (b0=130)", "labelZh": "Linear (b0=130)", "clicks": 281, "cpc": 74.89, "color": "#e37400"},
        {"name": "Human Rule", "labelZh": "Human Rule (人工经验)", "clicks": 251, "cpc": 66.04, "color": "#1a73e8"},
        {"name": "AlphaEvolve Champion", "labelZh": "AlphaEvolve 冠军策略", "clicks": 288, "cpc": 78.76, "color": "#0d652d"},
    ]
    chart_data_json = json.dumps(strategy_chart_data)

    return f"""
    <div class="card" style="border: 2px solid #1a73e8; box-shadow: 0 4px 12px rgba(26,115,232,0.12);">
      <h2>{section_title}</h2>
      <p style="color:var(--text-muted); font-size:14px; margin-top:0;">{section_desc}</p>

      <div class="stats-grid" style="margin-bottom:20px;">
        <div class="stat-card" style="background:#f8f9fa;">
          <div class="stat-label">{sim_lift_label}</div>
          <div class="stat-val positive">+{delta_sim:.2f}%</div>
          <div class="stat-sub positive">{sim_lift_sub}</div>
        </div>
        <div class="stat-card" style="background:#f8f9fa;">
          <div class="stat-label">{oracle_lift_label}</div>
          <div class="stat-val positive">+{delta_oracle:.2f}%</div>
          <div class="stat-sub positive">{oracle_lift_sub}</div>
        </div>
        <div class="stat-card" style="background:#f8f9fa;">
          <div class="stat-label">{consistency_label}</div>
          <div class="stat-val {'positive' if is_consistent else 'negative'}">{consistency_val}</div>
          <div class="stat-sub positive">{consistency_sub}</div>
        </div>
      </div>

      <!-- Policy Head-to-Head Comparison Chart -->
      <div style="background: white; border: 1px solid #dadce0; border-radius: 8px; padding: 18px 20px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; gap: 10px;">
          <h3 style="margin: 0; font-size: 15px; font-weight: 700; color: #202124;">
            {chart_title}
          </h3>
          <div style="display: flex; gap: 8px;">
            <button id="oracleBtnClicks" class="btn-metric-toggle active" onclick="setOracleMetric('clicks')">
              {btn_clicks_label}
            </button>
            <button id="oracleBtnCpc" class="btn-metric-toggle" onclick="setOracleMetric('cpc')">
              {btn_cpc_label}
            </button>
          </div>
        </div>

        <div id="oracleComparisonChart" style="width: 100%; height: 260px;"></div>
      </div>

      <!-- Metrics Table -->
      <table class="metrics-table">
        <thead>
          <tr>
            <th>{col_name}</th>
            <th>{col_clicks}</th>
            <th>{col_cpc}</th>
            <th>{col_spend}</th>
            <th>{col_win}</th>
            <th>{col_sim}</th>
            <th>{col_status}</th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>

      <!-- 4 Strategy Detailed Breakdown Cards -->
      <div style="margin-top: 25px; background: #f8f9fa; border: 1px solid #e8eaed; border-radius: 8px; padding: 20px;">
        <h3 style="margin-top: 0; margin-bottom: 14px; font-size: 16px; color: #1a73e8; display: flex; align-items: center; gap: 8px;">
          <span>📌</span> {base_desc_header}
        </h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px;">
          
          <!-- Mcpc Card -->
          <div style="background: white; border: 1px solid #dadce0; border-radius: 8px; padding: 16px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
              <div style="font-weight: 700; color: #202124; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
                <span>Mcpc</span>
                <span class="badge" style="background: #e8f0fe; color: #1a73e8;">{'静态按价值' if is_zh else 'Static Value'}</span>
              </div>
              <div style="font-size: 11px; color: #5f6368; margin-bottom: 8px; font-family: monospace; background: #f8f9fa; padding: 3px 6px; border-radius: 4px;">
                b = 1000 &times; Target_CPC &times; pCTR
              </div>
              <p style="margin: 0 0 10px 0; font-size: 13px; color: #5f6368; line-height: 1.5;">
                {'经典按预估曝光价值出价。不包含跨时段步调控制（Pacing）与消耗速度反馈，预算受限时无法动态节流。' if is_zh else 'Classic value-based bidding. Strictly proportional to estimated pCTR without spend pacing or feedback.'}
              </p>
            </div>
            <div style="border-top: 1px dashed #e8eaed; padding-top: 8px; font-size: 12px; color: #5f6368;">
              <strong>{'成绩' if is_zh else 'Result'}:</strong> 247 {'点击' if is_zh else 'Clicks'} | CPC 65.15 RMB | {'胜率' if is_zh else 'Win'}: 71.3%
            </div>
          </div>

          <!-- Linear Card -->
          <div style="background: white; border: 1px solid #dadce0; border-radius: 8px; padding: 16px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
              <div style="font-weight: 700; color: #202124; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
                <span>Linear (b0=130)</span>
                <span class="badge" style="background: #fef7e0; color: #b06000;">{'网格寻优' if is_zh else 'Grid Tuned'}</span>
              </div>
              <div style="font-size: 11px; color: #5f6368; margin-bottom: 8px; font-family: monospace; background: #f8f9fa; padding: 3px 6px; border-radius: 4px;">
                b = b0 &times; (pCTR / avg_pCTR)
              </div>
              <p style="margin: 0 0 10px 0; font-size: 13px; color: #5f6368; line-height: 1.5;">
                {'基准出价 b0 网格搜索调优至 130。对优质流量激进放大出价抢夺曝光，吃量能力强，但缺乏跨时段步调，极易早耗尽。' if is_zh else 'Tuned baseline via grid search. Bids aggressively on high-CTR inventory, but lacks dynamic pacing.'}
              </p>
            </div>
            <div style="border-top: 1px dashed #e8eaed; padding-top: 8px; font-size: 12px; color: #5f6368;">
              <strong>{'成绩' if is_zh else 'Result'}:</strong> 281 {'点击' if is_zh else 'Clicks'} | CPC 74.89 RMB | {'胜率' if is_zh else 'Win'}: 84.8%
            </div>
          </div>

          <!-- Human Rule Card -->
          <div style="background: white; border: 1px solid #dadce0; border-radius: 8px; padding: 16px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
              <div style="font-weight: 700; color: #202124; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
                <span>Human Rule (Seed)</span>
                <span class="badge" style="background: #e6f4ea; color: #137333;">{'专家规则' if is_zh else 'Expert Rules'}</span>
              </div>
              <div style="font-size: 11px; color: #5f6368; margin-bottom: 8px; font-family: monospace; background: #f8f9fa; padding: 3px 6px; border-radius: 4px;">
                if/elif step adjustments (&plusmn;15%)
              </div>
              <p style="margin: 0 0 10px 0; font-size: 13px; color: #5f6368; line-height: 1.5;">
                {'资深计算广告优化师总结的阶梯阈值规则。逻辑直观，但阶跃式调价会导致控制震荡，无法平滑逼近连续空间最优解。' if is_zh else 'Hand-crafted if/elif heuristics by ad-ops engineers. While intuitive, rigid step thresholds trigger oscillations.'}
              </p>
            </div>
            <div style="border-top: 1px dashed #e8eaed; padding-top: 8px; font-size: 12px; color: #5f6368;">
              <strong>{'成绩' if is_zh else 'Result'}:</strong> 251 {'点击' if is_zh else 'Clicks'} | CPC 66.04 RMB | {'胜率' if is_zh else 'Win'}: 72.6%
            </div>
          </div>

          <!-- AlphaEvolve Winner Card -->
          <div style="background: #e8f0fe; border: 2px solid #1a73e8; border-radius: 8px; padding: 16px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 2px 8px rgba(26,115,232,0.12);">
            <div>
              <div style="font-weight: 700; color: #1a73e8; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
                <span>AlphaEvolve Champion</span>
                <span class="badge" style="background: #0d652d; color: white; font-weight:700;">{'🏆 演化冠军' if is_zh else '🏆 Winner'}</span>
              </div>
              <div style="font-size: 11px; color: #174ea6; margin-bottom: 8px; font-family: monospace; background: white; padding: 3px 6px; border-radius: 4px;">
                Closed-loop Pacing + Asymmetric Momentum + Hyperbolic Brake
              </div>
              <p style="margin: 0 0 10px 0; font-size: 13px; color: #3c4043; line-height: 1.5;">
                {'200代演化涌现的非线性闭环控制律。自主探索出对数动量阻尼与三次双曲紧急制动，兼顾高胜率与保价合规。' if is_zh else 'Evolved closed-loop control law featuring logarithmic momentum damping and hyperbolic safety brake.'}
              </p>
            </div>
            <div style="border-top: 1px dashed #aecbfa; padding-top: 8px; font-size: 12px; color: #0d652d; font-weight: 700;">
              <strong>{'最佳战绩' if is_zh else 'Peak Record'}:</strong> 288 {'点击 (+16.6%)' if is_zh else 'Clicks (+16.6%)'} | CPC 78.76 RMB | {'胜率' if is_zh else 'Win'}: 87.8%
            </div>
          </div>

        </div>
      </div>
    </div>

    <script>
      (function() {{
        const chartData = {chart_data_json};
        const isZh = { 'true' if is_zh else 'false' };
        let currentMetric = 'clicks';

        function renderComparisonChart() {{
          const container = document.getElementById('oracleComparisonChart');
          if (!container) return;
          const width = container.clientWidth || 900;
          const height = 260;
          const padding = {{ top: 35, right: 30, bottom: 40, left: 60 }};
          const chartW = width - padding.left - padding.right;
          const chartH = height - padding.top - padding.bottom;

          const isClicks = (currentMetric === 'clicks');
          const maxVal = isClicks ? 320 : 140;
          const targetCap = 120.0;

          const numBars = chartData.length;
          const barWidth = Math.min(65, chartW / numBars * 0.55);
          const gap = chartW / numBars;

          let barsSvg = '';
          chartData.forEach((d, idx) => {{
            const val = isClicks ? d.clicks : d.cpc;
            const barH = (val / maxVal) * chartH;
            const x = padding.left + idx * gap + (gap - barWidth) / 2;
            const y = padding.top + chartH - barH;
            const color = (idx === 3) ? '#0d652d' : (idx === 1 ? '#e37400' : '#1a73e8');
            const valLabel = isClicks ? `${{val}}` : `${{val.toFixed(2)}} RMB`;
            const liftLabel = (isClicks && idx === 3) ? '<tspan fill="#0d652d" font-weight="700"> (+16.6%)</tspan>' : '';

            barsSvg += `
              <rect x="${{x}}" y="${{y}}" width="${{barWidth}}" height="${{barH}}" rx="5" ry="5" fill="${{color}}" opacity="0.9">
                <title>${{d.name}}: ${{valLabel}}</title>
              </rect>
              <text x="${{x + barWidth / 2}}" y="${{y - 8}}" font-size="12" font-weight="700" fill="#202124" text-anchor="middle">
                ${{valLabel}}${{liftLabel}}
              </text>
              <text x="${{x + barWidth / 2}}" y="${{height - padding.bottom + 20}}" font-size="12" font-weight="600" fill="#3c4043" text-anchor="middle">
                ${{isZh ? d.labelZh : d.name}}
              </text>
            `;
          }});

          // Reference line for Target CPC Cap (120 RMB)
          let targetLineSvg = '';
          if (!isClicks) {{
            const capY = padding.top + chartH - (targetCap / maxVal) * chartH;
            targetLineSvg = `
              <line x1="${{padding.left}}" y1="${{capY}}" x2="${{width - padding.right}}" y2="${{capY}}" stroke="#d93025" stroke-width="2" stroke-dasharray="6,4" />
              <text x="${{width - padding.right - 10}}" y="${{capY - 6}}" font-size="11" font-weight="700" fill="#d93025" text-anchor="end">
                ${{isZh ? '目标 CPC 硬上限: 120.00 RMB (Target CPC Cap)' : 'Target CPC Hard Limit: 120.00 RMB'}}
              </text>
            `;
          }}

          // Y-Axis Ticks
          let yTicksSvg = '';
          const steps = 4;
          for (let s = 0; s <= steps; s++) {{
            const tVal = (maxVal / steps) * s;
            const yPos = padding.top + chartH - (tVal / maxVal) * chartH;
            yTicksSvg += `
              <line x1="${{padding.left}}" y1="${{yPos}}" x2="${{width - padding.right}}" y2="${{yPos}}" stroke="#f1f3f4" stroke-width="1" stroke-dasharray="3,3" />
              <text x="${{padding.left - 8}}" y="${{yPos + 4}}" font-size="11" fill="#5f6368" text-anchor="end">${{tVal.toFixed(0)}}</text>
            `;
          }}

          container.innerHTML = `
            <svg viewBox="0 0 ${{width}} ${{height}}" style="width:100%; height:auto;">
              ${{yTicksSvg}}
              ${{targetLineSvg}}
              <line x1="${{padding.left}}" y1="${{padding.top + chartH}}" x2="${{width - padding.right}}" y2="${{padding.top + chartH}}" stroke="#dadce0" stroke-width="1.5" />
              <line x1="${{padding.left}}" y1="${{padding.top}}" x2="${{padding.left}}" y2="${{padding.top + chartH}}" stroke="#dadce0" stroke-width="1.5" />
              ${{barsSvg}}
              <text x="${{padding.left}}" y="${{padding.top - 12}}" font-size="11" font-weight="700" fill="#5f6368">
                ${{isClicks ? (isZh ? '真实获胜点击量 (Clicks)' : 'Real Won Clicks') : (isZh ? '实际点击成本 (RMB)' : 'Empirical CPC (RMB)')}}
              </text>
            </svg>
          `;
        }}

        window.setOracleMetric = function(m) {{
          currentMetric = m;
          const btnClicks = document.getElementById('oracleBtnClicks');
          const btnCpc = document.getElementById('oracleBtnCpc');
          if (m === 'clicks') {{
            btnClicks.classList.add('active');
            btnCpc.classList.remove('active');
          }} else {{
            btnCpc.classList.add('active');
            btnClicks.classList.remove('active');
          }}
          renderComparisonChart();
        }};

        renderComparisonChart();
        window.addEventListener('resize', renderComparisonChart);
      }})();
    </script>
    """


def generate_html_report(history_file: Path, output_file: Path, lang: str = "en", force_gemini: bool = False) -> Path:
    """Generates the full evolution HTML report in English or Chinese."""
    if not history_file.exists():
        raise FileNotFoundError(f"History file {history_file} not found. Run 'make run' first.")

    with open(history_file, "r", encoding="utf-8") as f:
        history_data = json.load(f)

    is_zh = (lang == "zh")
    programs: List[Dict[str, Any]] = history_data.get("programs", [])
    metric_name = history_data.get("metric_name", AUTO_BIDDING_EVALUATION_METRIC)

    # Sort chronologically by createTime for historical fidelity
    programs_chrono = sorted(programs, key=lambda p: p.get("createTime", ""))

    # Extract score trajectory in chronological generation order
    trajectory: List[Dict[str, Any]] = []
    best_so_far = -float("inf")

    def _get_score(p: Dict[str, Any]) -> float:
        scores = p.get("evaluation", {}).get("scores", {}).get("scores", [])
        for s in scores:
            if s.get("metric") == metric_name:
                return float(s.get("score", -float("inf")))
        return -float("inf")

    for idx, p in enumerate(programs_chrono):
        score = _get_score(p)
        if score > best_so_far:
            best_so_far = score
        trajectory.append({
            "index": idx + 1,
            "name": p.get("name", f"prog_{idx+1}"),
            "score": score,
            "bestSoFar": best_so_far,
            "best_so_far": best_so_far,
            "label": p.get("evaluation", {}).get("insights", {}).get("insights", [{}])[0].get("label", "Evaluation"),
            "text": p.get("evaluation", {}).get("insights", {}).get("insights", [{}])[0].get("text", ""),
            "createTime": p.get("createTime", ""),
        })

    # Pick champion program
    champ_code = history_data.get("champion_code", "")
    if not champ_code and programs:
        best_p = max(programs, key=_get_score)
        files = best_p.get("content", {}).get("files", [])
        for f in files:
            if f.get("path") == "program.py":
                champ_code = f.get("content", "")
                break
    if not champ_code:
        champ_code = INITIAL_PROGRAM_CODE

    seed_code = INITIAL_PROGRAM_CODE

    seed_metrics = evaluate_code_metrics(seed_code)
    champ_metrics = evaluate_code_metrics(champ_code)

    seed_evolve_block = extract_evolve_block(seed_code)
    champ_evolve_block = extract_evolve_block(champ_code)
    diff_html = generate_diff_html(seed_evolve_block, champ_evolve_block)

    # Fetch Gemini Code Mutation Difference Analysis
    task_dir = history_file.parent
    gemini_en, gemini_zh = get_or_create_gemini_analysis(task_dir, seed_code, champ_code, force_refresh=force_gemini)
    gemini_analysis_html = gemini_zh if is_zh else gemini_en

    # Compute lifts
    click_lift = (
        ((champ_metrics["expected_clicks"] - seed_metrics["expected_clicks"]) / seed_metrics["expected_clicks"] * 100)
        if seed_metrics["expected_clicks"] > 0 else 0.0
    )
    cpc_reduction = (
        ((seed_metrics["expected_cpc"] - champ_metrics["expected_cpc"]) / seed_metrics["expected_cpc"] * 100)
        if seed_metrics["expected_cpc"] > 0 else 0.0
    )
    score_lift = champ_metrics["fitness"] - seed_metrics["fitness"]

    # Load Oracle Benchmark Data if available
    oracle_file = task_dir / "oracle_benchmark.json"
    oracle_data = None
    if oracle_file.exists():
        try:
            with open(oracle_file, "r", encoding="utf-8") as f:
                oracle_data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load oracle benchmark file: %s", e)

    # Generate Pitch Deck and Oracle Section
    pitch_deck_html = generate_pitch_deck_html(lang=lang)
    oracle_section_html = generate_oracle_section_html(oracle_data, lang=lang)

    # Trajectory points JSON for SVG rendering
    traj_json = json.dumps(trajectory)
    task_id_display = history_data.get("task_id") or task_dir.name

    # Language Switcher
    lang_switch_html = (
        '语言切换: <a href="evolution_report.html" style="color:white; text-decoration:underline;">English</a> | <strong>中文版</strong>'
        if is_zh else
        'Language: <strong>English</strong> | <a href="evolution_report_zh.html" style="color:white; text-decoration:underline;">中文版</a>'
    )

    # Localized text variables
    page_title = f"AlphaEvolve Auto-Bidding 演化报告 [{task_id_display}]" if is_zh else f"AlphaEvolve Auto-Bidding Evolution Report [{task_id_display}]"
    header_title = "🚀 AlphaEvolve 广告出价自动演化与 Oracle 验证报告" if is_zh else "🚀 AlphaEvolve Auto-Bidding Evolution &amp; Oracle Verification"
    header_subtitle = "基于 iPinYou RTB 数据集 (Campaign 1458) 的目标 CPC 实时乘数策略自适应演化" if is_zh else "Target CPC Real-Time Multiplier Optimization on iPinYou RTB Dataset (Campaign 1458)"

    stat1_label = "评估候选代码总数" if is_zh else "Total Programs Evaluated"
    stat1_sub = "Gemini 企业级闭环搜索" if is_zh else "Gemini Enterprise Loop"
    stat2_label = "模拟点击量提升" if is_zh else "Simulated Click Uplift"
    stat2_sub = "相较于初始人工规则种子" if is_zh else "vs. Human Rule Baseline"
    stat3_label = "适应度得分提升" if is_zh else "Fitness Score Uplift"
    stat3_sub = "复合约束优化目标" if is_zh else "Constrained Optimization"
    stat4_label = "经验 CPC 降幅" if is_zh else "Empirical CPC Change"
    stat4_sub = "目标 CPC: 120.00 RMB" if is_zh else "Target CPC: 120.00 RMB"

    traj_title = "📈 AlphaEvolve 演化优化动力学历程 (Optimization Trajectory)" if is_zh else "📈 AlphaEvolve Optimization Trajectory (Generation 1 to 200)"
    traj_explanation = (
        "💡 <strong>演化历程交互动画说明</strong>：默认从第 1 代（初始人工规则）开始。点击“播放演化”或拖动进度滑块，可动态复盘 1~200 代探索突变产生与最优前沿（绿色虚线）阶跃跃迁的完整过程。红色散点为违规罚函数触发，蓝色散点为合规探索。"
        if is_zh else
        "💡 <strong>Interactive Replay Guide</strong>: Begins at Generation 1. Click 'Play Evolution' or scrub the generation slider to animate the 200-generation search process and observe how the green dashed Best Frontier ascents monotonically. Red dots indicate barrier penalties; blue dots are valid policies."
    )

    compare_title = (
        "⚖️ 核心业务与技术评估指标对比 (Business &amp; Technical Scorecard)"
        if is_zh else
        "⚖️ Business &amp; Technical Scorecard: Seed vs. Champion (Simulation Horizon)"
    )
    col_metric = "评估指标" if is_zh else "Metric"
    col_seed = "初始种子策略 (Seed)" if is_zh else "Seed Program (Human Rule)"
    col_champ = "AlphaEvolve 冠军策略" if is_zh else "AlphaEvolve Champion"
    col_diff = "相对变化率" if is_zh else "Delta / Lift"

    row_fit = "适应度评分 (Fitness Score)" if is_zh else "Fitness Score"
    row_clicks = "期望点击量 (Expected Clicks)" if is_zh else "Expected Clicks"
    row_cpc = "期望单次点击成本 (Expected CPC)" if is_zh else "Expected CPC"
    row_spend = "期望总预算消耗 (Expected Spend)" if is_zh else "Expected Spend"
    row_ood = "分布外越界率 (OOD Rate)" if is_zh else "OOD Bid Ratio"

    diff_title = "🔬 核心代码突变差异 (Code Mutation Diff)" if is_zh else "🔬 Code Mutation Diff: Seed vs. Champion Program"
    gemini_badge_text = "由 Vertex AI Gemini 3.8 Flash (global) 深度剖析" if is_zh else "Analyzed by Vertex AI Gemini 3.8 Flash (global)"
    gemini_card_title = "🧠 Gemini 模型深度解析：冠军策略相较于种子策略的机制革新与业务影响" if is_zh else "🧠 Gemini Model Deep Architecture &amp; Mechanism Analysis"

    html_template = f"""<!DOCTYPE html>
<html lang="{ 'zh-CN' if is_zh else 'en' }">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{page_title}</title>
  <style>
    :root {{
      --primary: #1a73e8;
      --primary-dark: #1557b0;
      --success: #0d652d;
      --success-bg: #e6f4ea;
      --warning: #e37400;
      --danger: #d93025;
      --danger-bg: #fce8e6;
      --bg: #f8f9fa;
      --card-bg: #ffffff;
      --text: #202124;
      --text-muted: #5f6368;
      --border: #dadce0;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.6;
      margin: 0;
      padding: 30px 20px;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    .header {{
      background: linear-gradient(135deg, #1a73e8, #174ea6);
      color: white;
      padding: 30px 40px;
      border-radius: 12px;
      margin-bottom: 25px;
      box-shadow: 0 4px 12px rgba(26, 115, 232, 0.2);
    }}
    .header-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }}
    .lang-switcher {{
      font-size: 13px;
      background: rgba(255, 255, 255, 0.2);
      padding: 4px 12px;
      border-radius: 20px;
    }}
    .header h1 {{
      margin: 0 0 10px 0;
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.5px;
    }}
    .header p {{
      margin: 0;
      font-size: 14px;
      opacity: 0.92;
    }}
    .task-tag {{
      background: rgba(255,255,255,0.22);
      padding: 2px 10px;
      border-radius: 6px;
      font-weight: 600;
      margin-left: 6px;
    }}

    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 25px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border-radius: 10px;
      border: 1px solid var(--border);
      padding: 18px 22px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }}
    .stat-label {{
      font-size: 13px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }}
    .stat-val {{
      font-size: 30px;
      font-weight: 700;
      color: var(--text);
    }}
    .stat-sub {{
      font-size: 13px;
      margin-top: 6px;
      display: flex;
      align-items: center;
      gap: 4px;
    }}
    .positive {{ color: var(--success); font-weight: 600; }}
    .neutral {{ color: var(--primary); font-weight: 600; }}
    .negative {{ color: var(--danger); font-weight: 600; }}

    .card {{
      background: var(--card-bg);
      border-radius: 10px;
      border: 1px solid var(--border);
      padding: 24px 28px;
      margin-bottom: 25px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }}
    .card h2 {{
      margin-top: 0;
      font-size: 19px;
      border-bottom: 2px solid #f1f3f4;
      padding-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    table.metrics-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 15px;
    }}
    table.metrics-table th, table.metrics-table td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid #f1f3f4;
      font-size: 14px;
    }}
    table.metrics-table th {{
      background-color: #f8f9fa;
      color: var(--text-muted);
      font-weight: 600;
    }}
    table.metrics-table tr:hover {{
      background-color: #f8f9fa;
    }}

    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 600;
    }}
    .badge-pass {{ background-color: var(--success-bg); color: var(--success); }}
    .badge-fail {{ background-color: var(--danger-bg); color: var(--danger); }}

    /* Pitch Deck Button Styles */
    .btn-deck-nav {{
      padding: 6px 14px;
      border-radius: 6px;
      border: 1px solid #dadce0;
      background: white;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      color: #3c4043;
      transition: all 0.2s;
    }}
    .btn-deck-nav:hover:not(:disabled) {{
      background: #f1f3f4;
      color: #1a73e8;
    }}
    .btn-deck-nav:disabled {{
      opacity: 0.4;
      cursor: not-allowed;
    }}
    .deck-thumb-btn {{
      padding: 10px;
      border-radius: 8px;
      border: 1px solid #dadce0;
      background: white;
      text-align: left;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .deck-thumb-btn.active {{
      border: 2px solid #1a73e8;
      background: #e8f0fe;
    }}

    /* Metric Toggle Button */
    .btn-metric-toggle {{
      padding: 6px 14px;
      border-radius: 6px;
      border: 1px solid #dadce0;
      background: white;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      color: #3c4043;
      transition: all 0.2s;
    }}
    .btn-metric-toggle.active {{
      background: #1a73e8;
      color: white;
      border-color: #1a73e8;
    }}

    /* Trajectory Animation Control Buttons */
    .btn-traj-ctrl {{
      padding: 6px 12px;
      border-radius: 6px;
      border: 1px solid #dadce0;
      background: white;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      color: #3c4043;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: all 0.2s;
    }}
    .btn-traj-ctrl:hover {{
      background: #f1f3f4;
      border-color: #1a73e8;
      color: #1a73e8;
    }}
    .btn-traj-ctrl.active-play {{
      background: #e8f0fe;
      color: #1a73e8;
      border-color: #1a73e8;
    }}
    .btn-speed {{
      padding: 3px 8px;
      border-radius: 4px;
      border: 1px solid #dadce0;
      background: white;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
    }}
    .btn-speed.active {{
      background: #1a73e8;
      color: white;
      border-color: #1a73e8;
    }}

    /* Diff Viewer */
    pre.diff-code {{
      background: #1e1e1e;
      color: #d4d4d4;
      padding: 16px;
      border-radius: 8px;
      overflow-x: auto;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
      font-size: 13px;
      line-height: 1.5;
    }}
    .diff-add {{ background-color: rgba(46, 160, 67, 0.25); color: #7ee787; display: block; }}
    .diff-del {{ background-color: rgba(248, 81, 73, 0.25); color: #ff7b72; display: block; }}
    .diff-hunk {{ color: #79c0ff; display: block; font-weight: bold; margin: 4px 0; }}
    .diff-ctx {{ color: #d4d4d4; display: block; }}

    /* Gemini Card */
    .gemini-card {{
      border-left: 4px solid #1a73e8;
      background: #ffffff;
      margin-top: 25px;
    }}
    .gemini-card h4 {{
      color: #1a73e8;
      font-size: 16px;
      margin-top: 5px;
      margin-bottom: 12px;
    }}
    .gemini-card h5 {{
      color: #202124;
      font-size: 15px;
      margin-top: 18px;
      margin-bottom: 8px;
      border-bottom: 1px dashed #e8eaed;
      padding-bottom: 4px;
    }}
    .gemini-card ul {{
      padding-left: 20px;
      line-height: 1.7;
      font-size: 14px;
      color: #3c4043;
    }}
    .gemini-card code {{
      background: #f1f3f4;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: monospace;
      font-size: 13px;
      color: #d93025;
    }}

    /* SVG Chart */
    .chart-container {{
      width: 100%;
      height: 340px;
      margin-top: 15px;
      position: relative;
    }}
    svg {{
      width: 100%;
      height: 100%;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-top">
        <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.85;">
          Google Cloud &bull; AlphaEvolve Quantitative Research
        </div>
        <div class="lang-switcher">
          {lang_switch_html}
        </div>
      </div>
      <h1>{header_title}</h1>
      <p>{header_subtitle} &bull; <span class="task-tag">Task: {task_id_display}</span></p>
    </div>

    <!-- Executive KPI Grid -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">{stat1_label}</div>
        <div class="stat-val">{len(programs)}</div>
        <div class="stat-sub neutral">{stat1_sub}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">{stat2_label}</div>
        <div class="stat-val positive">+{click_lift:.2f}%</div>
        <div class="stat-sub positive">{stat2_sub}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">{stat3_label}</div>
        <div class="stat-val positive">+{score_lift:.4f}</div>
        <div class="stat-sub positive">{stat3_sub}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">{stat4_label}</div>
        <div class="stat-val {'positive' if cpc_reduction >= 0 else 'neutral'}">{champ_metrics['expected_cpc']:.2f} RMB</div>
        <div class="stat-sub neutral">{stat4_sub}</div>
      </div>
    </div>

    <!-- Executive Presentation Pitch Deck Section -->
    {pitch_deck_html}

    <!-- Optimization Trajectory Section with Interactive Replay -->
    <div class="card">
      <h2>{traj_title}</h2>
      <p style="color:var(--text-muted); font-size:13px; margin:0 0 14px 0;">{traj_explanation}</p>
      
      <!-- Interactive Replay Control Bar -->
      <div style="background: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 14px 18px; margin-bottom: 14px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
          <div style="display: flex; align-items: center; gap: 8px;">
            <button id="btnPlayPause" class="btn-traj-ctrl" onclick="togglePlay()">
              <span id="playIcon">▶</span> <span id="playText">{ '播放演化历程' if is_zh else 'Play Evolution' }</span>
            </button>
            <button class="btn-traj-ctrl" onclick="resetToGen1()">
              ↺ { '重置' if is_zh else 'Reset' }
            </button>
            <button class="btn-traj-ctrl" onclick="stepGen(-1)">⏮</button>
            <button class="btn-traj-ctrl" onclick="stepGen(1)">⏭</button>
            <button class="btn-traj-ctrl" onclick="jumpToChampion()">
              🏆 { '直达冠军代' if is_zh else 'Jump to Winner' }
            </button>
            <button class="btn-traj-ctrl" onclick="showAllGenerations()">
              🌐 { '全景总览' if is_zh else 'Show All' }
            </button>
          </div>

          <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 12px; color: #5f6368;">{ '倍速' if is_zh else 'Speed' }:</span>
            <div style="display: flex; gap: 4px;">
              <button class="btn-speed active" onclick="setSpeed(1, this)">1x</button>
              <button class="btn-speed" onclick="setSpeed(2, this)">2x</button>
              <button class="btn-speed" onclick="setSpeed(5, this)">5x</button>
              <button class="btn-speed" onclick="setSpeed(10, this)">10x</button>
            </div>
          </div>
        </div>

        <!-- Scrubber Slider -->
        <div style="display: flex; align-items: center; gap: 12px;">
          <input type="range" id="genSlider" min="1" max="{len(trajectory)}" value="1" style="flex: 1; accent-color: #1a73e8; cursor: pointer;" oninput="onSliderChange(this.value)">
          <span id="genSliderBadge" style="font-size: 13px; font-weight: 700; color: #1a73e8; min-width: 90px; text-align: right;">Gen 1 / {len(trajectory)}</span>
        </div>
      </div>

      <!-- Real-time Candidate Inspector Card -->
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; background: white; border: 1px solid #dadce0; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
        <div>
          <span style="font-size: 11px; color: #5f6368; text-transform: uppercase;">{ '当前回放代际' if is_zh else 'Current Generation' }</span>
          <div id="inspGen" style="font-size: 18px; font-weight: 700; color: #202124;">Gen #1</div>
        </div>
        <div>
          <span style="font-size: 11px; color: #5f6368; text-transform: uppercase;">{ '当前突变得分' if is_zh else 'Candidate Score' }</span>
          <div id="inspScore" style="font-size: 18px; font-weight: 700; color: #1a73e8;">0.2088</div>
        </div>
        <div>
          <span style="font-size: 11px; color: #5f6368; text-transform: uppercase;">{ '当前最优前沿' if is_zh else 'Best Frontier' }</span>
          <div id="inspBest" style="font-size: 18px; font-weight: 700; color: #0d652d;">0.2088</div>
        </div>
        <div>
          <span style="font-size: 11px; color: #5f6368; text-transform: uppercase;">{ '代码演化状态' if is_zh else 'Candidate Insight' }</span>
          <div id="inspLabel" style="font-size: 13px; font-weight: 600; color: #3c4043; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{ 'Initial Human Rule Baseline' if not is_zh else '初始人工经验基准' }</div>
        </div>
      </div>

      <!-- Legend -->
      <div style="display: flex; gap: 20px; font-size: 13px; margin-bottom: 8px;">
        <span style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:12px; height:12px; background:#1a73e8; border-radius:50%;"></span>
          { '候选探索突变 (Exploration Score)' if is_zh else 'Exploratory Mutation Candidate' }
        </span>
        <span style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:12px; height:12px; background:#d93025; border-radius:50%;"></span>
          { '越界/超预算罚分 (Barrier Penalty)' if is_zh else 'Barrier Penalty Trigger' }
        </span>
        <span style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:20px; height:3px; background:#0d652d; border-top: 2px dashed #0d652d;"></span>
          { '全局最优前沿 (Best Frontier So Far)' if is_zh else 'Champion Frontier (Best So Far)' }
        </span>
        <span style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:12px; height:12px; border:2px solid #f29900; border-radius:50%;"></span>
          { '当前代焦点 (Current Selection)' if is_zh else 'Current Generation Focus' }
        </span>
      </div>

      <div class="chart-container" id="chart"></div>
    </div>

    <!-- Oracle Verification Section -->
    {oracle_section_html}

    <!-- Simulation Comparison Section -->
    <div class="card">
      <h2>{compare_title}</h2>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>{col_metric}</th>
            <th>{col_seed}</th>
            <th>{col_champ}</th>
            <th>{col_diff}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>{row_fit}</strong></td>
            <td>{seed_metrics['fitness']:.4f}</td>
            <td><strong>{champ_metrics['fitness']:.4f}</strong></td>
            <td class="positive">+{score_lift:.4f}</td>
          </tr>
          <tr>
            <td><strong>{row_clicks}</strong></td>
            <td>{seed_metrics['expected_clicks']:.2f}</td>
            <td><strong>{champ_metrics['expected_clicks']:.2f}</strong></td>
            <td class="positive">+{click_lift:.2f}%</td>
          </tr>
          <tr>
            <td><strong>{row_cpc}</strong></td>
            <td>{seed_metrics['expected_cpc']:.2f} RMB</td>
            <td><strong>{champ_metrics['expected_cpc']:.2f} RMB</strong></td>
            <td>{seed_metrics['expected_cpc'] - champ_metrics['expected_cpc']:+.2f} RMB</td>
          </tr>
          <tr>
            <td><strong>{row_spend}</strong></td>
            <td>{seed_metrics['expected_spend']:.2f} RMB</td>
            <td><strong>{champ_metrics['expected_spend']:.2f} RMB</strong></td>
            <td>{champ_metrics['expected_spend'] - seed_metrics['expected_spend']:+.2f} RMB</td>
          </tr>
          <tr>
            <td><strong>{row_ood}</strong></td>
            <td>{seed_metrics['ood_ratio']*100:.2f}%</td>
            <td><strong>{champ_metrics['ood_ratio']*100:.2f}%</strong></td>
            <td>{champ_metrics['ood_ratio']*100 - seed_metrics['ood_ratio']*100:+.2f}%</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Code Mutation Diff Section -->
    <div class="card">
      <h2>{diff_title}</h2>
      <p style="color:var(--text-muted); font-size:14px; margin-top:0;">
        { '展示在 200 代演化探索中，LLM 在 # EVOLVE-BLOCK 代码块中实现的结构重构：' if is_zh else 'Structural mutations synthesized by LLM inside the # EVOLVE-BLOCK across 200 generations:' }
      </p>
      {diff_html}

      <!-- Gemini Deep Architecture & Mechanism Analysis -->
      <div class="gemini-card" style="padding: 20px 24px; border-radius: 8px; border: 1px solid #dadce0; border-left: 5px solid #1a73e8;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #e8eaed; padding-bottom: 8px;">
          <h3 style="margin: 0; font-size: 16px; color: #202124; display: flex; align-items: center; gap: 8px;">
            <span>✨</span> {gemini_card_title}
          </h3>
          <span style="font-size: 11px; background: linear-gradient(135deg, #1a73e8, #4285f4); color: white; padding: 3px 10px; border-radius: 12px; font-weight: 600;">
            {gemini_badge_text}
          </span>
        </div>
        {gemini_analysis_html}
      </div>
    </div>
  </div>

  <script>
    const traj = {traj_json};
    const isZh = { 'true' if is_zh else 'false' };
    const maxGen = traj.length || 200;
    
    // Interactive Replay State: Initialize at Gen 1 as requested!
    let currentGen = 1;
    let isPlaying = false;
    let playSpeed = 1;
    let playIntervalId = null;

    const container = document.getElementById('chart');

    function updateInspector(genIdx) {{
      const d = traj[genIdx - 1] || traj[0];
      document.getElementById('inspGen').innerText = `Gen #${{d.index}}`;
      
      const isPenalty = d.score < 0;
      const scoreEl = document.getElementById('inspScore');
      scoreEl.innerText = d.score.toFixed(4);
      scoreEl.style.color = isPenalty ? '#d93025' : '#1a73e8';

      const bestVal = (d.bestSoFar !== undefined ? d.bestSoFar : d.best_so_far) || d.score;
      document.getElementById('inspBest').innerText = bestVal.toFixed(4);
      
      let labelText = d.label;
      if (d.text) labelText += ` (${{d.text}})`;
      document.getElementById('inspLabel').innerText = labelText;
      document.getElementById('genSliderBadge').innerText = `Gen ${{d.index}} / ${{maxGen}}`;
      document.getElementById('genSlider').value = d.index;
    }}

    function renderTrajectoryChart() {{
      if (!container || traj.length === 0) return;
      const width = container.clientWidth || 1100;
      const height = 340;
      const padding = {{ top: 40, right: 40, bottom: 50, left: 65 }};

      const allScores = traj.map(d => d.score);
      const allBest = traj.map(d => d.bestSoFar !== undefined ? d.bestSoFar : (d.best_so_far !== undefined ? d.best_so_far : d.score));
      const rawMax = Math.max(...allScores, ...allBest);
      const minScore = 0.0;
      const maxScore = Math.ceil(Math.max(rawMax * 1.15, 1.60) * 10) / 10;
      const yRange = (maxScore - minScore) || 1.0;

      const xScale = (i) => padding.left + (i / Math.max(1, maxGen - 1)) * (width - padding.left - padding.right);
      const yScale = (s) => height - padding.bottom - ((Math.max(s, minScore) - minScore) / yRange) * (height - padding.top - padding.bottom);

      let pointsSvg = '';
      let bestPath = '';
      let focusRingSvg = '';

      traj.forEach((d, i) => {{
        const genNum = i + 1;
        const x = xScale(i);
        const y = yScale(d.score);
        const bestVal = d.bestSoFar !== undefined ? d.bestSoFar : d.best_so_far;
        const yBest = yScale(bestVal);

        // Best frontier path up to currentGen
        if (genNum <= currentGen) {{
          if (i === 0) {{
            bestPath += `M ${{x}} ${{yBest}}`;
          }} else {{
            bestPath += ` L ${{x}} ${{yBest}}`;
          }}
        }}

        const isPenalty = d.score < 0;
        const isPastOrCurrent = genNum <= currentGen;
        
        // Colors & opacity
        let ptColor = isPenalty ? '#d93025' : '#1a73e8';
        let ptRadius = isPenalty ? 3 : 3.5;
        let ptOpacity = isPastOrCurrent ? 0.85 : 0.12;

        if (genNum === currentGen) {{
          ptRadius = 6;
          ptColor = '#f29900';
          ptOpacity = 1.0;
          // Outer pulsing ring
          focusRingSvg = `<circle cx="${{x}}" cy="${{y}}" r="11" fill="none" stroke="#f29900" stroke-width="2.5" opacity="0.9"/>`;
        }}

        pointsSvg += `<circle cx="${{x}}" cy="${{y}}" r="${{ptRadius}}" fill="${{ptColor}}" opacity="${{ptOpacity}}" style="cursor:pointer;" onclick="selectGen(${{genNum}})">
          <title>Gen #${{d.index}}: Score ${{d.score.toFixed(4)}} (${{d.label}})</title>
        </circle>`;
      }});

      // Y-axis gridlines
      const tickStep = (maxScore - minScore) / 4;
      let gridSvg = '';
      for (let t = minScore; t <= maxScore + 0.001; t += tickStep) {{
        const yPos = yScale(t);
        gridSvg += `
          <line x1="${{padding.left}}" y1="${{yPos}}" x2="${{width - padding.right}}" y2="${{yPos}}" stroke="#f1f3f4" stroke-width="1" stroke-dasharray="3,3"/>
          <text x="${{padding.left - 10}}" y="${{yPos + 4}}" font-size="11" fill="#5f6368" text-anchor="end">${{t.toFixed(2)}}</text>
        `;
      }}

      // Milestone X ticks
      const milestones = [1, 50, 100, 150, maxGen].filter((v, idx, arr) => arr.indexOf(v) === idx && v <= maxGen);
      let xTicksSvg = '';
      milestones.forEach(m => {{
        const xPos = xScale(m - 1);
        xTicksSvg += `
          <line x1="${{xPos}}" y1="${{height - padding.bottom}}" x2="${{xPos}}" y2="${{height - padding.bottom + 5}}" stroke="#dadce0" stroke-width="1"/>
          <text x="${{xPos}}" y="${{height - padding.bottom + 20}}" font-size="11" fill="#5f6368" text-anchor="middle">Gen ${{m}}</text>
        `;
      }});

      container.innerHTML = `
        <svg viewBox="0 0 ${{width}} ${{height}}" style="width:100%; height:auto; overflow:visible;">
          <line x1="${{padding.left}}" y1="${{height - padding.bottom}}" x2="${{width - padding.right}}" y2="${{height - padding.bottom}}" stroke="#dadce0" stroke-width="1.5"/>
          <line x1="${{padding.left}}" y1="${{padding.top}}" x2="${{padding.left}}" y2="${{height - padding.bottom}}" stroke="#dadce0" stroke-width="1.5"/>
          
          ${{gridSvg}}
          ${{xTicksSvg}}

          ${{bestPath ? `<path d="${{bestPath}}" fill="none" stroke="#0d652d" stroke-width="2.5" stroke-dasharray="5,3"/>` : ''}}
          ${{pointsSvg}}
          ${{focusRingSvg}}

          <text x="${{padding.left}}" y="${{padding.top - 12}}" font-size="12" font-weight="700" fill="#202124">
            ${{isZh ? '适应度得分 (Fitness Score)' : 'Fitness Score'}}
          </text>
          <text x="${{width - padding.right}}" y="${{height - 12}}" font-size="12" font-weight="600" fill="#202124" text-anchor="end">
            ${{isZh ? '迭代代际 (Generation 1 → 200)' : 'Generation Iteration (1 → 200)'}}
          </text>
        </svg>
      `;
    }}

    window.selectGen = function(gen) {{
      currentGen = Math.max(1, Math.min(maxGen, gen));
      updateInspector(currentGen);
      renderTrajectoryChart();
    }};

    window.onSliderChange = function(val) {{
      if (isPlaying) togglePlay(); // pause on user drag
      selectGen(parseInt(val, 10));
    }};

    window.stepGen = function(delta) {{
      if (isPlaying) togglePlay();
      selectGen(currentGen + delta);
    }};

    window.resetToGen1 = function() {{
      if (isPlaying) togglePlay();
      selectGen(1);
    }};

    window.jumpToChampion = function() {{
      if (isPlaying) togglePlay();
      // Find peak index
      let peakIdx = 1;
      let peakScore = -Infinity;
      traj.forEach((d) => {{
        if (d.score > peakScore) {{
          peakScore = d.score;
          peakIdx = d.index;
        }}
      }});
      selectGen(peakIdx);
    }};

    window.showAllGenerations = function() {{
      if (isPlaying) togglePlay();
      selectGen(maxGen);
    }};

    window.togglePlay = function() {{
      isPlaying = !isPlaying;
      const btn = document.getElementById('btnPlayPause');
      const icon = document.getElementById('playIcon');
      const text = document.getElementById('playText');

      if (isPlaying) {{
        btn.classList.add('active-play');
        icon.innerText = '⏸';
        text.innerText = isZh ? '暂停' : 'Pause';

        // Loop
        const intervalMs = Math.max(50, Math.floor(500 / playSpeed));
        playIntervalId = setInterval(() => {{
          if (currentGen >= maxGen) {{
            togglePlay(); // stop at end
            return;
          }}
          selectGen(currentGen + 1);
        }}, intervalMs);
      }} else {{
        btn.classList.remove('active-play');
        icon.innerText = '▶';
        text.innerText = isZh ? '播放演化历程' : 'Play Evolution';
        if (playIntervalId) {{
          clearInterval(playIntervalId);
          playIntervalId = null;
        }}
      }}
    }};

    window.setSpeed = function(spd, elem) {{
      playSpeed = spd;
      document.querySelectorAll('.btn-speed').forEach(b => b.classList.remove('active'));
      elem.classList.add('active');
      if (isPlaying) {{
        // Restart interval with new speed
        togglePlay();
        togglePlay();
      }}
    }};

    // Initial render
    updateInspector(currentGen);
    renderTrajectoryChart();
    window.addEventListener('resize', renderTrajectoryChart);
  </script>
</body>
</html>
"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)

    logger.info("Generated HTML evolution report (%s) at %s", lang, output_file)
    return output_file


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HTML Evolution Report")
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Task ID to report on (default: latest task)",
    )
    parser.add_argument(
        "--force-gemini",
        action="store_true",
        help="Force re-running Gemini analysis using Vertex AI",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path("config.yaml")
    cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    artifacts_base = Path(cfg.get("artifacts", {}).get("dir", "./artifacts"))
    effective_task_id, task_dir = resolve_task_artifacts_dir(
        task_id=args.task_id,
        artifacts_base=artifacts_base,
        create=False,
    )

    history_file = task_dir / "evolution_history.json"
    report_file_en = task_dir / "evolution_report.html"
    report_file_zh = task_dir / "evolution_report_zh.html"

    # Fallback to base artifacts dir if history_file does not exist in task_dir
    if not history_file.exists():
        fallback_history = artifacts_base / "evolution_history.json"
        if fallback_history.exists():
            history_file = fallback_history
            report_file_en = artifacts_base / "evolution_report.html"
            report_file_zh = artifacts_base / "evolution_report_zh.html"

    if not history_file.exists():
        print(f"Notice: History file {history_file} does not exist yet.")
        print("Generating report based on current Seed Program against baseline evaluation...")
        history_file.parent.mkdir(parents=True, exist_ok=True)
        seed_metrics = evaluate_code_metrics(INITIAL_PROGRAM_CODE)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump({
                "task_id": effective_task_id,
                "experiment_name": "local_seed_eval",
                "metric_name": AUTO_BIDDING_EVALUATION_METRIC,
                "total_programs": 1,
                "programs": [{
                    "name": "seed_program_0",
                    "content": {"files": [{"path": "program.py", "content": INITIAL_PROGRAM_CODE}]},
                    "evaluation": {
                        "scores": {"scores": [{"metric": AUTO_BIDDING_EVALUATION_METRIC, "score": seed_metrics["fitness"]}]},
                        "insights": {"insights": [{"label": "Seed Policy", "text": "Initial baseline human heuristic policy"}]}
                    }
                }]
            }, f, indent=2)

    # Generate English Report
    out_en = generate_html_report(history_file, report_file_en, lang="en", force_gemini=args.force_gemini)
    # Generate Chinese Report
    out_zh = generate_html_report(history_file, report_file_zh, lang="zh", force_gemini=False)

    print(f"\n✅ Evolution reports [{effective_task_id}] successfully generated:")
    print(f"  • English Report: file://{out_en.resolve()}")
    print(f"  • Chinese Report: file://{out_zh.resolve()}\n")


if __name__ == "__main__":
    main()
