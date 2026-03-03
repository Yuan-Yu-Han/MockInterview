# MockInterview —— AI 模拟面试系统

基于 **agno** 框架 + 本地 **vLLM（Qwen3-8B）** + 可选 **RAG 检索**，
实现「面试官 Agent ↔ 求职者 Agent」多角色博弈的自动化模拟面试。

---

## 功能特性

- **AI 完整模拟**：面试官 Agent 与求职者 Agent 全程自动对话
- **人工练习模式**：面试官由 AI 驱动，你亲自作答，实时获得评分反馈
- **智能追问**：每道主问题自动生成 1 次深挖追问
- **匹配度分析**：开始前自动计算简历与 JD 的技能匹配度
- **结构化评估**：EvaluatorAgent 对每轮问答独立打分（0–10）
- **最终报告**：生成综合得分、录用建议、技能缺口和改进方向
- **RAG 增强**（可选）：接入本地 MODULAR-RAG-MCP-SERVER，支持从向量库检索简历/JD 原文

---

## 项目结构

```
MockInterview/
├── run_interview.py              # CLI 入口
├── run_vllm.sh                   # 启动 vLLM 服务
├── start_rag_server.sh           # 启动 RAG MCP 服务器
├── requirements.txt
├── data/
│   ├── uploads/                  # 放你的简历和 JD 文件
│   └── output/                   # 生成的面试报告（JSON + TXT）
├── interview/
│   ├── config.py                 # 全局配置（vLLM 地址、路径等）
│   ├── models.py                 # Pydantic 数据模型
│   ├── prompts.py                # 四个 Agent 的系统提示词
│   ├── utils.py                  # JSON 解析、token 裁剪工具
│   ├── runner.py                 # 面试主流程（含 RAG / 非 RAG 两路）
│   ├── report.py                 # 报告导出（JSON + TXT）
│   └── agents/
│       └── factory.py            # 四个 Agent 的工厂函数
└── MODULAR-RAG-MCP-SERVER/       # RAG 子系统（独立项目）
```

---

## 四个 Agent 的职责

| Agent | 知道什么 | 职责 |
|---|---|---|
| **ParseAgent** | 简历 + JD + 可选 RAG | 解析文档、计算匹配度、生成最终报告 |
| **InterviewerAgent** | 简历 + JD + 可选 RAG | 生成主问题和追问 |
| **IntervieweeAgent** | **仅自己的简历** | 模拟候选人回答（信息刻意隔离） |
| **EvaluatorAgent** | 问题 + 回答 | 客观打分，结果对候选人不可见 |

> **IntervieweeAgent 信息隔离**是核心设计——它不知道 JD 要求，不知道评分标准，
> 与真实候选人拥有的信息完全对等。这也使得它可以无缝替换为真人参与（`--mode human`）。

---

## 快速开始

### 1. 进入环境

```bash
module load Miniconda3
source activate vllm
```

### 2. 安装依赖

```bash
cd /home/yuan0165/MockInterview
pip install -r requirements.txt
```

### 3. 启动 vLLM 服务（另开终端）

```bash
bash run_vllm.sh
```

服务启动后监听 `http://localhost:8000`，模型名 `Qwen3-8B`。

### 4. 准备简历和 JD 文件

将简历和职位描述保存为文本文件，放入 `data/uploads/`。
可使用内置样例直接测试：

```
data/uploads/sample_resume.txt
data/uploads/sample_jd.txt
```

### 5. 运行面试

**AI 完整模拟（默认）：**
```bash
python run_interview.py \
    --resume data/uploads/sample_resume.txt \
    --jd data/uploads/sample_jd.txt
```

**人工练习模式（你来回答）：**
```bash
python run_interview.py \
    --resume data/uploads/sample_resume.txt \
    --jd data/uploads/sample_jd.txt \
    --mode human
```

**自定义题目数量 + 保存报告：**
```bash
python run_interview.py \
    --resume data/uploads/sample_resume.txt \
    --jd data/uploads/sample_jd.txt \
    --questions 3 \
    --save
```

---

## 命令行参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--resume` | 简历文件路径（必填） | — |
| `--jd` | 职位描述文件路径（必填） | — |
| `--mode` | `ai`（AI 模拟）或 `human`（人工练习） | `ai` |
| `--questions` | 主问题数量（每题含 1 次追问） | `5` |
| `--rag` | 启用 RAG 检索（需配置 MCP 服务器） | 关闭 |
| `--save` | 保存报告到 `data/output/` | 关闭 |
| `--quiet` | 只显示最终报告，不打印实时问答 | 关闭 |

---

## 面试流程

```
简历 + JD
    ↓
[ParseAgent]  解析文档 → ResumeInfo / JobDescription / MatchAnalysis
    ↓
[第 1 轮]
  InterviewerAgent  ──→  生成问题
  IntervieweeAgent 或你  ──→  回答
  EvaluatorAgent    ──→  打分（面试过程中不展示给候选人）
  InterviewerAgent  ──→  追问
  IntervieweeAgent 或你  ──→  回答
  EvaluatorAgent    ──→  打分
    ↓
[第 2~N 轮]  重复上述步骤
    ↓
[ParseAgent]  汇总所有问答 → InterviewReport
    ↓
打印报告 / 保存文件
```

---

## 启用 RAG 模式

RAG 模式通过 agno 的 `MCPTools`（stdio 传输）调用本地 MODULAR-RAG-MCP-SERVER，
让 ParseAgent 和 InterviewerAgent 可以检索向量库中的简历/JD 内容。

> **注意**：IntervieweeAgent 刻意不连接 RAG，保持信息隔离原则。

### 配置步骤

1. 编辑 `MODULAR-RAG-MCP-SERVER/config/settings.yaml`，配置 embedding 模型和向量库
2. 将简历/JD PDF 文件 ingest 进向量库（参考 MODULAR-RAG-MCP-SERVER 的文档）
3. 运行时加上 `--rag` 参数：

```bash
python run_interview.py --resume resume.txt --jd jd.txt --rag
```

---

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `VLLM_BASE_URL` | vLLM 服务地址 | `http://localhost:8000/v1/` |
| `VLLM_MODEL_NAME` | 模型名称 | `Qwen3-8B` |
| `VLLM_API_KEY` | API 密钥（本地不需要） | `not-needed` |

---

## 技术栈

| 组件 | 技术 |
|---|---|
| Agent 框架 | [agno](https://docs.agno.com) |
| 推理引擎 | [vLLM](https://github.com/vllm-project/vllm) |
| 模型 | Qwen3-8B（本地部署） |
| RAG 检索 | MODULAR-RAG-MCP-SERVER（MCP stdio 协议） |
| 数据验证 | Pydantic v2 |
