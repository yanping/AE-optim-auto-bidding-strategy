# Google Cloud AlphaEvolve RTB 智能出价策略演化框架

[English](README.md) | **简体中文**

基于 **Google Cloud AlphaEvolve (Auto-Coding AI Engine)** 与 **Vertex AI (Gemini 3.8 Flash · global)**，在计算广告业界标准基准 **iPinYou 真实实时竞价数据集 (Campaign 1458)** 上实现的自适应目标出价（Target-CPC Auto-Bidding）智能策略演化与闭环验证系统。

本项目针对现代实时竞价（Real-Time Bidding, RTB）中买方单边删失（Censored Feedback，买方仅知己方胜出时支付价格，未知落败竞价及对手底牌）的核心业务痛点，通过 Kaplan-Meier 极限定理建立高精度买方响应模型，驱动 AlphaEvolve 在 Python 代码符号空间中进行自主重构与演化，突破传统人工经验规则与强化学习调参瓶颈，在严格遵守目标 CPC 硬上限的约束下，最大化广告主真实转化与点击收益。

> 📘 **架构白皮书与客户推介材料**：
> - 完整方案宏观介绍、生产落地飞轮辨析与买方删失数据说明详见 [docs/client_presentation_guide_CN.md](docs/client_presentation_guide_CN.md)。
> - 适应度得分的数学形式与多目标运筹学推导详见 [docs/fitness_and_objective_design_CN.md](docs/fitness_and_objective_design_CN.md)。
> - iPinYou 数据集特征工程与清洗规范详见 [docs/ipinyou_real_data_report_CN.md](docs/ipinyou_real_data_report_CN.md)。

> 🌐 **在线交互式演化报告 (Live Interactive Reports)**：
> * **英文报告 (English Version)**: [https://storage.googleapis.com/auto-bidding-spartan-figure-500309-g2/auto-bidding-demo/evolution_report.html](https://storage.googleapis.com/auto-bidding-spartan-figure-500309-g2/auto-bidding-demo/evolution_report.html)
> * **中文报告 (Chinese Version)**: [https://storage.googleapis.com/auto-bidding-spartan-figure-500309-g2/auto-bidding-demo/evolution_report_zh.html](https://storage.googleapis.com/auto-bidding-spartan-figure-500309-g2/auto-bidding-demo/evolution_report_zh.html)

---

## 🏛️ 系统架构与数据闭环

### 系统总体架构拓扑
![System Architecture](docs/assets/system_architecture.png)

### 数据流与生产落地推进闭环 (Production Flywheel)
![Data Flow Pipeline](docs/assets/data_flow_pipeline.png)

---

## 1. 目录规范与文件清单

```text
ads-AE/
├── Makefile                     # 自动化工作流入口 (setup, auth, download-data, prepare-data, run, oracle, report, test, mask, unmask)
├── config.yaml                  # 🌟 唯一全局配置文件 (可见、自解释、单一真实源，已脱敏)
├── requirements.txt             # 项目核心依赖清单 (numpy, pandas, pyarrow, scipy, scikit-learn, lifelines, matplotlib, google-genai)
├── instructions.md              # 传递给 AlphaEvolve 的策略演化问题描述、接口协议与变异指引
├── README.md                    # 项目完整使用说明与操作指南 (English)
├── README_CN.md                 # 项目完整使用说明与操作指南 (简体中文)
├── alpha_evolve/                # Google Cloud AlphaEvolve 官方核心代码 (只读)
├── scripts/                     # 运维与工程化交付脚本
│   └── mask_credentials.py     # 🔒 自动化项目交付凭据脱敏与恢复工具 (纯标准库，无外部依赖)
├── src/                         # 本项目核心工程实现
│   ├── bidding/                 # 竞价执行引擎与多策略基准测试套件
│   │   ├── benchmark.py         # 四大基准策略横向对比与 Day 7 Oracle 盲测验证
│   │   ├── candidate_bidding.py # AlphaEvolve 演化程序挂载容器与动态沙箱调用
│   │   ├── heuristic_rules.py   # 人工经验阶梯规则基准策略 (Seed Candidate)
│   │   ├── linear_bidding.py    # 经典线性出价策略 (Linear Grid Search)
│   │   ├── mcpc_bidding.py      # 静态价值出价策略 (Mcpc Baseline)
│   │   └── types.py             # 竞价上下文、出价状态与出价结果强类型定义
│   ├── data/                    # 数据流提取、删失掩蔽与时序划分管线
│   │   ├── download_data.py     # 原始 iPinYou 数据集获取、下载校验与合成测试集生成
│   │   ├── ipinyou_stream_extractor.py # bz2 压缩流提取、广告活动过滤与格式对齐
│   │   ├── prepare_data.py      # 🌟 端到端数据准备管线 CLI (支持任意 Campaign 与日期区间)
│   │   ├── run_ipinyou_pipeline.py # 数据管线封装层 (兼容历史调用接口)
│   │   ├── schema.py            # 数据集字段规范与 DatasetSplit 容器
│   │   └── splitter.py          # 严格时序划分与买方视角/平台全局视角单向隔离器
│   ├── env/                     # 离线回放仿真沙盒
│   │   └── offline_simulator.py # 基于胜率生存概率与积分成本期望的离线仿真评估器
│   ├── models/                  # 预测与统计机器学习模型
│   │   ├── ctr_model.py         # 基于买方历史日志预训练的逻辑回归 pCTR 预估器
│   │   └── market_model.py      # 分片非参数化 Kaplan-Meier 删失市场响应胜率估计模型
│   ├── utils/                   # 通用工具组件
│   │   ├── gemini_analyzer.py   # Vertex AI (Gemini 3.8 Flash · global) 演化机理剖析器
│   │   └── task_manager.py      # 演化实验生命周期与隔离目录管理器
│   ├── evaluate_bidding.py      # AlphaEvolve 标准评估入口 (计算适应度得分)
│   ├── report.py                # 现代化自包含中英双语 HTML 演化交互式报告生成引擎
│   └── run_evolution.py         # Google Cloud AlphaEvolve 演化主控入口 CLI
├── data/                        # 数据集目录 (已配置 .gitignore，仅保留目录结构)
│   ├── raw/                     # 原始 7z 压缩包与解压后的 bz2 日志
│   └── processed/               # 按 Campaign 隔离的时序切分 Parquet 数据集
├── artifacts/                   # 演化产物输出目录 (按任务时间戳隔离，已配置 .gitignore)
│   └── task_YYYYMMDD_HHMMSS/    # 单次演化产物 (演化历史、冠军代码、双语 HTML 报告)
├── docs/                        # 项目白皮书、架构图与演示资源
│   ├── assets/                  # 高清架构与生产闭环矢量图 (SVG / PNG)
│   ├── client_presentation_guide.md # 客户沟通推介指南与系统架构白皮书
│   ├── fitness_and_objective_design.md # 适应度函数数学形式与多目标运筹推导
│   └── ipinyou_real_data_report.md     # 数据集分析与探索报告
└── tests/                       # 全量自动化单元与集成测试套件 (24 项测试，100% 通过)
```

---

## 2. 前置准备与环境设置

### 2.1 操作系统与基础软件要求
- **操作系统**: macOS (Apple Silicon / Intel) 或 Linux (Ubuntu / Debian / CentOS)。
- **Python**: Python 3.10+ (推荐 Python 3.11 或 3.14)。
- **GNU Make**: 自动化构建工作流工具 (`make --version`)。
- **Google Cloud SDK**: `gcloud` CLI（用于 GCP 凭证 ADC 鉴权与 API 交互）。

### 2.2 创建 Python 虚拟环境与安装依赖 (首次运行必须执行)
> [!IMPORTANT]
> **交付前置说明**：为避免依赖冲突与体积冗余，本项目在 Git 仓库和交付包中不包含 `venv/` 虚拟环境。在您首次运行任何演化任务或测试前，**必须先在项目根目录下创建虚拟环境并安装依赖**！

提供了以下两种方式（任选其一即可）：

**方式一：通过 Makefile 一键自动化初始化（强烈推荐）**
```bash
make setup
```
该命令会自动检测并在当前根目录下创建 `.venv` 虚拟环境、升级 pip、安装 `requirements.txt` 中的全部依赖包并校验 `config.yaml`。

**方式二：手动命令行分步执行**
```bash
# 1. 创建虚拟环境
python3 -m venv .venv

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 升级 pip 并安装项目核心依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 GCP 账号与 Gemini Enterprise APP 设置
1. **准备 Google Cloud 项目**：
   - 准备一个已开通结算账号的 Google Cloud Project（记下您的 `PROJECT_ID`）。
2. **启用 Discovery Engine API**：
   ```bash
   gcloud services enable discoveryengine.googleapis.com --project=<YOUR_GCP_PROJECT_ID>
   ```
3. **获取 Gemini Enterprise APP ID**：
   - 在 Google Cloud Console 的 Gemini Enterprise 控制台中创建或查看应用，获取 Engine / App ID（记下您的 `GE_APP_ID`）。
4. **IAM 权限配置**：
   - 运行账号需具备 **Discovery Engine Editor** (`roles/discoveryengine.editor`) 或管理员角色。
   - 官方权限设置指导可参考：[AlphaEvolve 环境与 API 访问设置文档](https://docs.cloud.google.com/gemini/enterprise/docs/alphaevolve/developer-guide/environment-and-api-access-setup?hl=zh-cn)。
5. **本地凭据鉴权 (ADC)**：
   ```bash
   make auth
   # 或手动运行：gcloud auth application-default login
   ```

---

## 3. 数据集获取与准备管线 (`make download-data` & `make prepare-data`)

本项目支持从零开始自动化下载原始数据并处理，支持提取任意 Campaign ID 与指定日期区间。

### 3.1 获取原始 iPinYou 数据集
原始数据集压缩包为 `ipinyou.contest.dataset.7z`（压缩包约 6.30 GB，解压后约 14 GB）。

> ⚠️ **数据源可用性提示**：原 UCL 学术源（`data.computational-advertising.org`）服务器已永久下线。本项目已接入学术社区权威仓库 [`wnzhang/make-ipinyou-data`](https://github.com/wnzhang/make-ipinyou-data) 官方维护更新（PR #11）验证可用的全量镜像源。

```bash
# 选项 1：一键通过官方验证镜像源自动断点续传下载（推荐，无需手动打开浏览器）：
make download-data SOURCE=dropbox

# 或者直接在终端使用 curl 断点续传下载：
curl -L -C - --retry 5 -o data/raw/ipinyou.contest.dataset.7z "https://www.dropbox.com/s/txz0ms0axqf7jrl/ipinyou.contest.dataset.7z?dl=1"

# 选项 2：从 Kaggle 社区镜像手动或通过 CLI 下载：
# Web 页面: https://www.kaggle.com/datasets/lastsummer/ipinyou
kaggle datasets download -d lastsummer/ipinyou -p data/raw/ --unzip

# 选项 3：快速合成测试集（无需下载 6.3GB 即可立即运行完整演化流程）：
.venv/bin/python -m src.data.download_data --generate-sample
```

下载完成后，请确保原始文件位于 `data/raw/ipinyou.contest.dataset.7z`（或解压至 `data/raw/season2/`）。

### 3.2 准备与校准数据 (端到端数据管线)
```bash
# 1. 默认处理 Campaign 1458 (提取全部 7 天并自动切分训练/验证/测试集，拟合生存模型):
make prepare-data

# 2. 处理指定 Campaign (例如 Campaign 3358):
make prepare-data CAMPAIGN=3358

# 3. 指定时间区间过滤 (例如只提取 2013-06-06 至 2013-06-10):
make prepare-data CAMPAIGN=1458 START_DATE=20130606 END_DATE=20130610
```

数据处理完成后，会在 `data/processed/{CAMPAIGN}/` 下生成严格物理隔离的 Parquet 数据集：
- `train_advertiser.parquet`: 买方视角训练集 (落败竞价价格严格删失掩蔽为 NaN，无全局真实市场价)
- `val_advertiser.parquet`: 买方视角验证集 (用于 AlphaEvolve 快速评估搜索)
- `test_oracle.parquet`: 平台全局视角测试集 (含未脱敏真实出清价，仅用于最终 Day 7 真实市场盲测)
并在 `data/` 目录下生成生存模型校准曲线图 `calibration_curve_ipinyou_{CAMPAIGN}.png` ($R^2 \ge 0.99$)。

---

## 4. 全局配置文件说明 (`config.yaml`)

项目所有设置统一在根目录 `config.yaml` 中管理，无隐藏配置文件。首次运行时请在 `config.yaml` 中填入您自己的 GCP 凭据：

```yaml
# 1. Google Cloud 凭证与 Gemini Enterprise 服务配置
gcp:
  project_id: "<YOUR_GCP_PROJECT_ID>"  # 请在此填入您的 Google Cloud Project ID
  location: "global"
  collection: "default_collection"
  ge_app_id: "<YOUR_GE_APP_ID>"        # 请在此填入您的 Gemini Enterprise App/Engine ID
  assistant: "default_assistant"
  base_url: "discoveryengine.googleapis.com"
  alpha_evolve_path: "./alpha_evolve"

# 2. 竞价业务指标与预算配置 (针对 Campaign 1458)
bidding:
  campaign_id: "1458"
  target_cpc: 120.0                    # 目标 CPC 硬约束 (RMB)
  val_budget: 15000.0                  # 验证集预算上限 (RMB)
  test_budget: 25000.0                 # Day 7 测试集预算上限 (RMB)
  val_data_path: "data/processed/1458/val_advertiser.parquet"
  test_oracle_path: "data/processed/1458/test_oracle.parquet"

# 3. AlphaEvolve 演化参数配置
evolution:
  max_programs_generated: 200          # 演化代际总预算
  concurrency: 1                       # 串行演化确保精确度
  problem_description_path: "instructions.md"

# 4. Gemini 大模型生成权重配置 (使用最新 Gemini 3.8 Flash 全球端点)
models:
  - name: "gemini-3.8-flash"
    weight: 1.00
```

---

## 5. 快速上手与运行流程 (Quick Start)

完整的策略发现与验证仅需依次执行以下标准步骤：

```bash
# 步骤 1: 安装依赖并初始化虚拟环境
make setup

# 步骤 2: 登录 Google Cloud ADC 凭证
make auth

# 步骤 3: 准备数据集 (默认 Campaign 1458)
make prepare-data

# 步骤 4: 启动 AlphaEvolve 演化循环 (例如演化 200 代)
make run programs=200

# 步骤 5: 运行 Day 7 Held-out Oracle 真实市场盲测验证
make oracle

# 步骤 6: 编译并生成交互式中英双语 HTML 报告
make report

# 步骤 7: 运行全量单元测试套件
make test
```

---

## 6. 核心技术亮点与算法突破

### 1. 突破买方单边删失壁垒 (Kaplan-Meier Survival Modeling)
在真实 RTB 场景中，买方仅知己方胜出时的第二名出清价，绝大多数流量因落败而呈现**右删失（Right Censored）**特征。本项目首创引入生物医学与可靠性工程的 **分片非参数化 Kaplan-Meier 生存分析**，将竞价胜率形式化为生存函数：

$$
P(\mathrm{win} \mid b) = 1 - S(b) = 1 - \prod_{t_i \le b} \left(1 - \frac{d_i}{n_i}\right)
$$

在无需对手报价的前提下，仅凭买方历史日志即实现 $R^2 = 0.9935$ 的胜率预估精度，筑牢离线高保真演化基石。

### 2. 纯代码符号化演化 (Code-Level Symbolic Evolution)
区别于传统强化学习调参的“黑盒黑箱”，AlphaEvolve 借助 Gemini 3.8 Flash 直接在 Python 代码空间（AST 抽象语法树）中探索：
- **可读可解释**：发现的代码均为纯原生 Python 函数，业务专家可逐行审核并精准理解其背后的运筹机理；
- **自适应涌现**：自动衍生出对数预算平滑、三次双曲刹车与微价值拦截等高阶自适应控制论机制。

### 3. Day 7 Held-out Oracle 真实市场盲测验证
在完全未参与演化的第 7 天 447,493 次全量真实市场拍卖流中回放：
- **真实点击量**：斩获 **288 个真实点击**，相较静态价值基线提升 **+16.60%**，相较人工经验规则提升 **+14.74%**；
- **合规硬约束**：实际点击成本（CPC）为 **78.76 RMB**，严格压低在 120.00 RMB 安全红线以内，预算消耗率达 90.72%。

### 4. 完全自包含现代化交互式 HTML 双语报告
演化过程自动编译生成独立的 `evolution_report.html` 与 `evolution_report_zh.html`：
- **商业汇报幻灯片**：内置 5 页投影视角 Slide，支持键盘 `←` / `→` 键演讲；
- **200 代动力学历程演示**：支持播放、重置、单步、直达冠军、速度调节与实时代码突变探测；
- **策略横向柱状图**：动态切换点击量与 CPC/安全硬上限；
- **Gemini 深度剖析**：内嵌 Vertex AI (Gemini 3.8 Flash · global) 针对冠军策略 6 大控制论机制的深度解读。

---

## 7. 项目交付脱敏与隐私保护 (`make mask` & `make unmask`)

为保障企业级隐私安全，在将代码开源、分享给客户或推送到公共 GitHub 仓库前，请务必执行凭据脱敏工具。

### 7.1 脱敏命令
```bash
# 1. 预览模式 (查看哪些文件会被修改，不写盘)
make mask-dry

# 2. 正式执行脱敏 (自动替换 project_id 与 ge_app_id 为占位符，并保存本地恢复凭据)
make mask
```

脱敏工具会自动扫描代码库中的 `config.yaml`、文档和脚本，将您的真实凭证替换为 `<YOUR_GCP_PROJECT_ID>` 和 `<YOUR_GE_APP_ID>`，并在 `config.yaml` 中生成填写指导注释。同时会将您的原凭证安全暂存于本地 `.credentials.backup`（已被 `.gitignore` 保护，绝不会被推送到 Git）。

### 7.2 恢复凭证命令
```bash
# 1. 本地一键恢复 (自动读取 .credentials.backup 恢复为您本人的凭据):
make unmask

# 2. 或指定自定义凭证恢复:
make unmask PROJECT_ID=my-gcp-project-123 APP_ID=gemini-enterprise-999999
```

---

## 8. 许可证与致谢

本项目基于 Apache 2.0 许可证开源。
- 感谢 **Google Cloud AlphaEvolve Team** 提供的代码演化引擎支持；
- 感谢 **iPinYou Research** 提供的 RTB 计算广告真实竞价基准数据集。
